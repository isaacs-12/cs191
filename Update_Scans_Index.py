from Google import Create_Service
import os
import io
from googleapiclient.http import MediaIoBaseDownload

# Placeholder code -- current implemetation is to download from Google Drive manually when a notification of a file upload is received


file_id = '0BwwA4oUTeiV1UVNwOHItT0xfa2M'
request = drive_service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print "Download %d%%." % int(status.progress() * 100)