#RENAME THIS FILE TO 'settings.py'

#AWS
AWS_SERVER_PUBLIC_KEY='sample'
AWS_SECRET_ACCESS_KEY='sample'

#Datset
METADATA_FILE = 'trail_cam_ims.json'
BUCKET = 'us-west-2.opendata.source.coop'
PREFIX = 'agentmorris/lila-wildlife/nz-trailcams/'
ROOT_ELEMENT = 'images'
ID_PROPERTY = 'file_name'
CLASS_PROPERTY = 'species'
NUM_CLASSES = 97

#Data Saving
MODELS_PATH = 'Models/' 
