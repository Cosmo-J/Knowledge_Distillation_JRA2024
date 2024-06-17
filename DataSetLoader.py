import boto3
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io

# Define a custom dataset class
class S3Dataset(Dataset):
    def __init__(self, bucket,prefix, metadata_file, transform=None):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix        
        self.transform = transform

        # Load metadata file
        with open(metadata_file, 'r') as file:
            metadata_json = json.load(file)

        # Store images metadata
        self.images = metadata_json['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Access the metadata for the specific image
        image_metadata = self.images[idx]
        image_key = f"{self.prefix}{image_metadata['file_name']}"  # Construct the full key with the prefix

        # Get image from S3
        img_obj = self.s3.get_object(Bucket=self.bucket, Key=image_key)
        image = Image.open(io.BytesIO(img_obj['Body'].read()))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Label can be species or any other metadata
        label = image_metadata['species']  # Adjust according to the needs, can be 'location', etc.

        return image, label

