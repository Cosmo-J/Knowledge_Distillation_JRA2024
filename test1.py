import DataSetLoader
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pickle
from PIL import Image


# Define transformations if necessary
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create the dataset
metadata_file = 'trail_cam_ims.json'
bucket = 'us-west-2.opendata.source.coop'
prefix = 'agentmorris/lila-wildlife/nz-trailcams/'
dataset = DataSetLoader.S3Dataset(bucket=bucket, metadata_file=metadata_file,prefix=prefix, transform=transform)
sample_weight_pkl = 'trail_cam_ims_sample_weights.pkl'

# Loads pre-made sample weights
with open(sample_weight_pkl, 'rb') as f:
    sample_weights = pickle.load(f)

wSampler = WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)

data_loader = DataLoader(dataset, batch_size=10, sampler=wSampler)

# Usage example: Iterate over the data loader
for images, labels in data_loader:
    print(images.shape, labels)
