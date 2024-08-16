import boto3
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
import pickle
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import settings 
import numpy as np
import ClassWeightCalculator
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm 
import random


from collections import Counter

def worker_init_fn(worker_id):
    global s3
    import boto3
    session = boto3.Session(
        aws_access_key_id=settings.AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    s3 = session.client('s3')


class S3Dataset(Dataset):
    def __init__(self, bucket,prefix, metadata_file, transform=None):
        self.bucket = bucket
        self.prefix = prefix        
        self.transform = transform
        self.metadata_file = metadata_file
        self.label_file_name = ""


        with open(metadata_file, 'r') as file:
            metadata_json = json.load(file)
        self.images = metadata_json['images']
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(list({key['species'] for key in metadata_json['images']}))
        self.label_mapping = {label: int(index) for label, index in zip(le.classes_, le.transform(le.classes_))}

        suffix = '_classlabels.json'
        self.label_file_name = self.metadata_file.split('.')[0] + suffix
        with open(self.label_file_name, 'w') as file:
            json.dump(self.label_mapping, file, indent=4)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_metadata = self.images[idx]
        image_key = f"{self.prefix}{image_metadata['file_name']}" 

        image = self.download_image(image_key=image_key,idx=idx,img_metadata=image_metadata)

        if self.transform:
            image = self.transform(image)
        label = image_metadata['species'] 
        return image, label

    def download_image(self,image_key,idx,img_metadata):
        attempts = 0
        allowedAttempts = 3
        done = False

        while attempts < allowedAttempts:
            try:
                img_obj = s3.get_object(Bucket=self.bucket, Key=image_key)
                image = Image.open(io.BytesIO(img_obj['Body'].read()))
                return image
            except:
                attempts+=1

        # if it cant be found looks for another image of the same species
        for img in self.images:
            if img['species'] == img_metadata['species'] and img != img_metadata:
                image = self.download_image(f"{self.prefix}{img_metadata['file_name']}",idx,img_metadata)
                break
        return image

    




default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

metadata_file = settings.METADATA_FILE
bucket = settings.BUCKET
prefix = settings.PREFIX


def load_dataset(transform=default_transform, batch_size=10, test_split=0.2, random_seed=42,dataset_percentage=1,prefetch_factor=None,num_workers=1):
    
    sample_weights = ClassWeightCalculator.GenerateSampleWeights(dataset_percentage)
    dataset = S3Dataset(bucket=bucket, metadata_file=metadata_file, prefix=prefix, transform=transform)


    train_indices, test_indices = train_test_split(list(range(len(sample_weights))), test_size=test_split, random_state=random_seed)
 
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_weights = [sample_weights[i] for i in train_indices]
    test_weights = [sample_weights[i] for i in test_indices]

    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True)

    print(f"Train Sampler: {train_sampler}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,prefetch_factor=prefetch_factor,num_workers=num_workers,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, sampler=test_sampler,prefetch_factor=prefetch_factor,num_workers=num_workers,worker_init_fn=worker_init_fn)



    return train_loader, test_loader, dataset.label_file_name

