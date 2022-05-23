import os
from apiclient import discovery
from httplib2 import Http
import oauth2client
from oauth2client import file, client, tools
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from tqdm import tqdm
import datetime

folder_path = "./results"
drive_folder_id = "1gsciAbuYI9BgwyE4b95v30qgSBJgxh5_"

obj = lambda: None

lmao = {"auth_host_name": 'localhost', 'noauth_local_webserver': 'store_true', 'auth_host_port': [8080, 8090],
        'logging_level': 'ERROR'}
for k, v in lmao.items():
    setattr(obj, k, v)

# authorization code
SCOPES = 'https://www.googleapis.com/auth/drive'
store = file.Storage('token.json')
creds = store.get()

if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
    creds = tools.run_flow(flow, store, obj)

drive_service = build('drive', 'v3', credentials=creds)

for file_path in tqdm([os.path.join(folder_path, f) for f in os.listdir(folder_path) if ".xlsx" in f][950:]):
    createdTime = datetime.datetime.fromtimestamp(os.stat(file_path).st_ctime)
    createdTime = (createdTime - datetime.timedelta(microseconds=createdTime.microsecond)).isoformat() + ".00Z"
    file_metadata = {'name': os.path.basename(file_path), 'parents': [drive_folder_id], 'createdTime': createdTime}
    media = MediaFileUpload(file_path, mimetype='application/zip')
    try:
        file = drive_service.files().update(body=file_metadata, media_body=media, fields='id').execute()
    except:
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
