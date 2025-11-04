#!/usr/bin/env python3
"""
Authentication setup for video transcriber
Run this once to get Google Docs write permissions
"""

import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Scopes needed for video transcription with calendar integration
SCOPES = [
    'https://www.googleapis.com/auth/documents',  # Create/edit docs
    'https://www.googleapis.com/auth/drive.file',  # Manage created files
    'https://www.googleapis.com/auth/calendar.readonly'  # Read calendar events
]

def authenticate():
    """Authenticate and save credentials"""
    creds = None

    # Try to load existing credentials
    if os.path.exists('token_video.pickle'):
        print("Found existing credentials, checking if valid...")
        with open('token_video.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            # Look for credentials.json in multiple locations
            creds_file = None
            possible_locations = [
                'credentials.json',
                os.path.expanduser('~/Development/Scripts/Blog idea generator/credentials.json'),
            ]

            for loc in possible_locations:
                if os.path.exists(loc):
                    creds_file = loc
                    print(f"Using credentials from: {loc}")
                    break

            if not creds_file:
                print("\n❌ Error: credentials.json not found")
                print("\nPlease copy credentials.json from your Qwilo project:")
                print("  cp ~/Development/Scripts/Blog\\ idea\\ generator/credentials.json .")
                return False

            print("\n🔐 Opening browser for authentication...")
            print("Please authorize the app to create Google Docs")

            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for future use
        with open('token_video.pickle', 'wb') as token:
            pickle.dump(creds, token)

        print("\n✅ Authentication successful!")
        print("Credentials saved to: token_video.pickle")
        return True

    print("✅ Valid credentials found!")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("VIDEO TRANSCRIBER - AUTHENTICATION SETUP")
    print("=" * 60)
    print()

    if authenticate():
        print("\n✅ Setup complete! You can now run:")
        print("   python3 video_transcriber.py /path/to/videos --credentials token_video.pickle")
    else:
        print("\n❌ Setup failed. Please check the error above.")
