import os
from typing import Optional
from PIL import Image
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

# Import the browser automation module
from browser import accept_oath_url


# Custom flow class that changes the authorization URL message
class CustomInstalledAppFlow(InstalledAppFlow):
    def run_local_server(self, **kwargs):
        """Override to modify the auth URL display message and handle OAuth."""
        # Start the local server first (parent method)
        original_authorization_url = self.authorization_url
        
        def wrapped_authorization_url(*args, **kwargs):
            auth_url, state = original_authorization_url(*args, **kwargs)
            print(f"OAUTH URL: {auth_url}")
            
            # Use browser automation to handle the OAuth flow
            try:
                accept_oath_url(auth_url)
            except Exception as e:
                print(f"Browser automation failed: {e}")
                print("Continuing with manual authorization...")
                
            return auth_url, state
            
        # Replace the authorization_url method temporarily
        self.authorization_url = wrapped_authorization_url
        
        # Call the parent method which will use our wrapped authorization_url
        result = super().run_local_server(**kwargs)
        
        # Restore the original method
        self.authorization_url = original_authorization_url
        
        return result


UPLOAD_PUBLIC = True  # Set to True for immediate public upload, False for private with 24h delay

class YouTubeUploader:
    """Handles authentication and video upload to YouTube."""

    API_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = "client_secrets.json"
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",  # Add this scope for comments
    ]
    THUMBNAIL_FILE = "images/thumbnail.png"
    SHORT_TAGS = ["Shorts"]  # Añadido para shorts

    def __init__(self):
        """Initialize YouTubeUploader."""
        self.credentials: Optional[Credentials] = None
        self.youtube = None

    def authenticate(self) -> None:
        """Handle OAuth2 authentication flow with persistent refresh token."""
        try:
            if os.path.exists("token.json"):
                self.credentials = Credentials.from_authorized_user_file(
                    "token.json", self.SCOPES
                )

            if not self.credentials or not self.credentials.valid:
                if (
                    self.credentials
                    and self.credentials.expired
                    and self.credentials.refresh_token
                ):
                    try:
                        self.credentials.refresh(Request())
                        # Save refreshed credentials
                        with open("token.json", "w", encoding="utf-8") as token:
                            token.write(self.credentials.to_json())
                    except Exception as e:
                        self._handle_invalid_token()
                else:
                    self._handle_invalid_token()

            self.youtube = build(
                self.API_NAME, self.API_VERSION, credentials=self.credentials
            )
        except Exception as e:
            print(f"Authentication error: {e}")
            self._handle_invalid_token()

    def _handle_invalid_token(self) -> None:
        """Handle invalid token by initiating a new OAuth flow."""
        if os.path.exists("token.json"):
            os.remove("token.json")
        
        flow = CustomInstalledAppFlow.from_client_secrets_file(
            self.CLIENT_SECRETS_FILE, 
            self.SCOPES,
            # Enable offline access to get refresh token
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        flow.run_local_server(
            port=0,
            prompt='consent',  # Force consent screen to ensure refresh token
            access_type='offline'  # Request refresh token
        )
        self.credentials = flow.credentials
        
        # Save new credentials
        with open("token.json", "w", encoding="utf-8") as token:
            token.write(self.credentials.to_json())

    def refresh_token_if_needed(self) -> None:
        """Ensure token is valid, refresh or reauthenticate if needed."""
        try:
            if not self.credentials or not self.credentials.valid:
                if os.path.exists("token.json"):
                    os.remove("token.json")
                flow = CustomInstalledAppFlow.from_client_secrets_file(
                    self.CLIENT_SECRETS_FILE, self.SCOPES
                )
                self.credentials = flow.run_local_server(port=0)
                with open("token.json", "w", encoding="utf-8") as token:
                    token.write(self.credentials.to_json())
                self.youtube = build(
                    self.API_NAME, self.API_VERSION, credentials=self.credentials
                )
        except Exception as e:
            print(f"Error refreshing token: {e}")
            self.authenticate()

    def upload_video(
        self, video_file: str, title: str, description: str, hashtags: list[str]
    ) -> tuple[str, datetime]:
        """Upload video to YouTube with metadata from text files."""
        try:
            # Prepare the video upload request
            if UPLOAD_PUBLIC:
                # Upload as public immediately
                body = {
                    "snippet": {
                        "title": title,
                        "description": description,
                        "tags": hashtags,
                        "categoryId": "24",  # Entertainment category
                        "defaultLanguage": "en",
                        "defaultAudioLanguage": "en",
                    },
                    "status": {
                        "privacyStatus": "public",  # Set to public immediately
                        "selfDeclaredMadeForKids": False,
                    },
                }
                publish_time = datetime.now(timezone.utc)
            else:
                # Upload as private with 24h delay
                publish_time = datetime.now(timezone.utc) + timedelta(days=1)
                body = {
                    "snippet": {
                        "title": title,
                        "description": description,
                        "tags": hashtags,
                        "categoryId": "24",  # Entertainment category
                        "defaultLanguage": "en",
                        "defaultAudioLanguage": "en",
                    },
                    "status": {
                        "privacyStatus": "private",  # Set to private initially
                        "selfDeclaredMadeForKids": False,
                        "publishAt": publish_time.isoformat(),
                    },
                }

            # Create MediaFileUpload object
            media = MediaFileUpload(
                video_file,
                mimetype="video/mp4",
                chunksize=5 * 1024 * 1024,
                resumable=True,
            )

            # Execute the upload request
            print(f"Starting upload of video: {title}")
            request = self.youtube.videos().insert(
                part=",".join(body.keys()), body=body, media_body=media
            )

            # Handle the upload with progress reporting
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Uploaded {int(status.progress() * 100)}%")

            # Upload thumbnail
            self.youtube.thumbnails().set(
                videoId=response["id"],
                media_body=MediaFileUpload(
                    self.THUMBNAIL_FILE, chunksize=5 * 1024 * 1024, resumable=True
                ),
            ).execute()

            print(f"Upload Complete! Video ID: {response['id']}")
            return response["id"], publish_time

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

    def upload_short(self, video_file: str, title: str, description: str, hashtags: list[str]) -> str:
        """Upload video as a YouTube Short."""
        try:
            # Reuse the token from video uploading
            self.refresh_token_if_needed()
            # Prepare the video upload request
            body = {
                "snippet": {
                    "title": title,
                    "description": f"{description.split('\n')[0].strip()}\n\n👁️ Did you find it interesting? Then you can continue watching the video on my channel — don't miss it!\n\n➡️ Full video on my channel https://youtube.com/@my-burning-thoughts",
                    "tags": hashtags,
                    "categoryId": "27",  # Education category (could also use "24" for Entertainment)
                },
                "status": {
                    "privacyStatus": "public",  # Set to public for Shorts
                    "selfDeclaredMadeForKids": False,
                },
            }

            # Create MediaFileUpload object
            media = MediaFileUpload(
                video_file,
                mimetype="video/mp4",
                chunksize=5 * 1024 * 1024,
                resumable=True,
            )

            # Execute the upload request
            print(f"Starting upload of Short: {title}")
            request = self.youtube.videos().insert(
                part=",".join(body.keys()), body=body, media_body=media
            )

            # Handle the upload with progress reporting
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Uploaded {int(status.progress() * 100)}%")

            # After upload, update the video with vertical metadata
            self.youtube.videos().update(
                part="status",
                body={
                    "id": response["id"],
                    "status": {
                        "selfDeclaredMadeForKids": False,
                        "madeForKids": False,
                        "license": "youtube",
                        "embeddable": True,
                        "publicStatsViewable": True,
                    },
                },
            ).execute()

            print(f"Short Upload Complete! Video ID: {response['id']}")
            return response["id"]

        except Exception as e:
            print(f"An error occurred while uploading the Short: {str(e)}")
            raise

    def post_comment(self, video_id: str, comment_text: str) -> str:
        """Post a comment on a specific video."""
        try:
            # Ensure token is valid before posting
            self.refresh_token_if_needed()

            comment_insert_response = (
                self.youtube.commentThreads()
                .insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "videoId": video_id,
                            "topLevelComment": {
                                "snippet": {"textOriginal": comment_text}
                            },
                        }
                    },
                )
                .execute()
            )

            comment_id = comment_insert_response["id"]
            print(f"Comment posted successfully! Comment ID: {comment_id}")
            return comment_id

        except Exception as e:
            if "invalid_grant" in str(e):
                print("Token expired, refreshing authentication...")
                self.authenticate()
                # Retry the comment post once after re-authentication
                return self.post_comment(video_id, comment_text)
            print(f"An error occurred while posting the comment: {str(e)}")
            raise


def save_video_details(
    now: datetime, title: str, description: str, comment: str
) -> None:
    """
    Save video details to a text file.

    Args:
        video_id: YouTube video ID
        title: Video title
        description: Video description
        publish_time: Scheduled publish time
    """
    details = {
        "VIDEO": now.strftime("%d-%m-%Y_%H-%M-%S"),
        "TITLE": title,
        "DESCRIPTION": description,
        "COMMENT": comment,
    }

    filename = f"text/video.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for key, value in details.items():
            f.write(f"[{key}]\n{value}\n\n")


def upload_video(
    video_file: str, title: str, description: str, comment: str, hashtags: list[str]
) -> None:
    uploader = YouTubeUploader()
    try:
        # Authenticate
        uploader.authenticate()
        
        # Upload video
        video_id, publish_time = uploader.upload_video(
            video_file, title, description, hashtags
        )
        print(
            f"Video uploaded successfully! URL: https://youtube.com/watch?v={video_id}"
        )

        # Convert UTC to local timezone
        local_publish_time = publish_time.astimezone()
        wait_until = local_publish_time + timedelta(minutes=10)
        wait_seconds = (wait_until - datetime.now().astimezone()).total_seconds()

        if wait_seconds > 0:
            print(
                f"Waiting until {wait_until.strftime('%d/%m/%Y %H:%M:%S')} local time to post comment..."
            )
            time.sleep(wait_seconds)
        else:
            print("Target time has already passed, posting comment immediately...")

        print("Posting comment...")
        uploader.post_comment(video_id, comment)

        # Upload the Short version if it exists
        short_path = str(Path(video_file).parent / "short.mp4")
        if os.path.exists(short_path):
            print("Uploading Short version...")
            short_id = uploader.upload_short(short_path, title, description, hashtags)
            print(f"Short uploaded successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
