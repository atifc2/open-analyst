import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import os
import ollama

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="AI Data Analyst", layout="wide")

# Initialize session state
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_cleaning_report' not in st.session_state:
    st.session_state.show_cleaning_report = False

def clean_data(df):
    """Clean the dataframe and return cleaning report"""
    original_shape = df.shape
    cleaning_steps = []
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna(how='all')  # Drop rows where all values are missing
    df = df.dropna(axis=1, how='all')  # Drop columns where all values are missing
    missing_after = df.isnull().sum().sum()
    
    if missing_before > 0:
        cleaning_steps.append(f"Removed {missing_before - missing_after} missing values")
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        cleaning_steps.append(f"Removed {duplicates} duplicate rows")
    
    # 3. Convert date columns
    date_columns = df.select_dtypes(include=['object']).apply(
        lambda x: pd.to_datetime(x, errors='coerce').notna().any()
    )
    for col in date_columns[date_columns].index:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        cleaning_steps.append(f"Converted '{col}' to datetime")
    
    # 4. Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        # Remove extra whitespace
        df[col] = df[col].astype(str).str.strip()
        # Convert empty strings to NaN
        df[col] = df[col].replace('', np.nan)
    
    cleaning_steps.append(f"Cleaning complete. Shape changed from {original_shape} to {df.shape}")
    
    return df, cleaning_steps

def generate_chart_suggestions(df):
    """Generate chart suggestions based on data"""
    try:
        # Get data summary for LLM
        data_summary = f"""
        Data Shape: {df.shape}
        \nColumn Types:\n{df.dtypes}
        \nFirst 5 rows:\n{df.head().to_string()}
        """
        
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Suggest 1-2 best chart types for this data and explain why. Be concise."},
                {"role": "user", "content": f"Suggest visualizations for this data:\n{data_summary}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error generating chart suggestions: {str(e)}"

def analyze_data_with_llm(df, question=None):
    """Generate analysis using Ollama"""
    try:
        data_summary = f"""
        Data Shape: {df.shape}
        \nColumn Names: {', '.join(df.columns)}
        \nColumn Types:\n{df.dtypes}
        \nSummary Statistics:\n{df.describe(include='all').to_string()}
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful data analyst. Provide clear, concise analysis."}
        ]
        
        if question:
            messages.append({
                "role": "user",
                "content": f"Question: {question}\n\nData Summary:{data_summary}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"""Analyze this data and provide:
1. Data overview (shape, types, missing values)
2. Key trends and patterns
3. Potential outliers or data quality issues
4. One business recommendation based on the data

Be concise and to the point. Use bullet points.

Data Summary:{data_summary}"""
            })
        
        response = ollama.chat(
            model="llama3",
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def display_dataframe_info(df):
    """Display dataframe information and preview"""
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Basic info
    st.subheader("📋 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values, 
                 delta=f"{missing_values/df.size*100:.1f}% of total" if df.size > 0 else "0%")
    
    # Data types
    st.subheader("🔍 Data Types")
    st.write(df.dtypes.astype(str))

def display_charts(df):
    """Display suggested charts"""
    st.subheader("📈 Chart Suggestions")
    with st.spinner("Generating chart suggestions..."):
        chart_suggestions = generate_chart_suggestions(df)
        st.write(chart_suggestions)
        
        # Simple auto-plotting for common column patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
        
        if len(numeric_cols) > 0:
            if len(date_cols) > 0:
                # If we have dates and numbers, create a time series plot
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                try:
                    fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} over time")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
            
            # Create a histogram for the first numeric column
            try:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

def main():
    st.title("🤖 AI Data Analyst (Local LLM)")
    
    # Sidebar for file upload and history
    with st.sidebar:
        st.header("📤 Upload File")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", 
                                      type=["csv", "xlsx", "xls"],
                                      key="file_uploader")
        
        # Display upload history
        if st.session_state.upload_history:
            st.header("📚 File History")
            for idx, file_info in enumerate(st.session_state.upload_history):
                if st.button(f"📄 {file_info['name']} ({file_info['shape'][0]}x{file_info['shape'][1]})", 
                           key=f"history_{idx}",
                           use_container_width=True):
                    st.session_state.current_file = file_info['name']
            st.markdown("---")
        
        # Data cleaning options
        if st.session_state.current_df is not None:
            if st.button("🧹 Clean Data", use_container_width=True):
                with st.spinner("Cleaning data..."):
                    cleaned_df, cleaning_report = clean_data(st.session_state.current_df.copy())
                    st.session_state.current_df = cleaned_df
                    st.session_state.show_cleaning_report = True
                    st.session_state.cleaning_report = cleaning_report
                    st.rerun()
            
            if st.session_state.get('show_cleaning_report', False):
                with st.expander("🧽 Cleaning Report"):
                    for step in st.session_state.get('cleaning_report', []):
                        st.write(f"✅ {step}")
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.current_df = df
            st.session_state.current_file = uploaded_file.name
            
            # Add to upload history if not already there
            file_info = {
                'name': uploaded_file.name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'shape': df.shape
            }
            
            if not any(f['name'] == uploaded_file.name for f in st.session_state.upload_history):
                st.session_state.upload_history.insert(0, file_info)
            
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Display current data if available
    if st.session_state.current_df is not None:
        df = st.session_state.current_df
        
        # Display file info and preview
        st.header(f"📂 {st.session_state.current_file}")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Data Overview", "AI Analysis", "Ask Questions"])
        
        with tab1:
            display_dataframe_info(df)
            display_charts(df)
        
        with tab2:
            st.subheader("🤖 AI-Powered Analysis")
            if st.button("Generate Analysis", key="analyze_btn"):
                with st.spinner("🧠 Analyzing data with local LLM..."):
                    analysis = analyze_data_with_llm(df)
                    st.markdown(analysis)
        
        with tab3:
            st.subheader("💬 Chat with your data")
            
            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your data"):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = analyze_data_with_llm(df, prompt)
                        st.write(response)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    else:
        # Show welcome message if no file uploaded
        st.markdown("""
        ## Welcome to AI Data Analyst! 👋
        
        This version runs completely locally using Ollama and Llama 3.
        
        ### How to use:
        1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
        2. Pull the model: `ollama pull llama3`
        3. Upload your CSV/Excel file using the sidebar
        4. Explore the data in the "Data Overview" tab
        5. Get AI analysis in the "AI Analysis" tab
        6. Ask questions in the "Ask Questions" tab
        
        ### Sample Data:
        Try uploading the Mall_Customers.csv file to get started.
        """)

if __name__ == "__main__":
    main()
