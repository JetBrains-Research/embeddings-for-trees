import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def upload_file(file_name: str):
    google_auth = GoogleAuth()
    # client_secrets.json need to be in the same directory as the script
    google_auth.CommandLineAuth()
    drive = GoogleDrive(google_auth)
    file = drive.CreateFile({'title': os.path.basename(file_name)})
    file.SetContentFile(file_name)
    file.Upload()
