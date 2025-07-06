# Secure Google Drive Upload with Streamlit

This application allows users to securely upload files to their Google Drive using OAuth 2.0 authentication. The app doesn't store any user tokens permanently.

## Features

- Secure Google OAuth 2.0 authentication
- Upload files directly to your Google Drive in a folder named "AI-Analyst-TEMP"
- No server-side token storage
- Simple and intuitive interface

## Prerequisites

- Python 3.7+
- Google Cloud Platform account with OAuth 2.0 credentials

## Setup

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Google Cloud Project:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the Google Drive API
   - Create OAuth 2.0 credentials (Web application)
   - Add `http://localhost:8501/` as an authorized redirect URI
   - Download the credentials and update the `.env` file

4. Create a `.env` file in the project root with your credentials:
   ```
   GOOGLE_CLIENT_ID=your_client_id
   GOOGLE_CLIENT_SECRET=your_client_secret
   REDIRECT_URI=http://localhost:8501/
   ```

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and go to `http://localhost:8501`
3. Click "Sign in with Google" and authenticate
4. Upload files to your Google Drive

## Security Notes

- The application never stores your Google credentials
- All OAuth tokens are stored in the browser's session storage
- The app only requests the minimum required permissions
- Your Google Drive files remain private and secure

## License

MIT
