import boto3
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import pickle
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import settings 
import numpy as np

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
        # Access the metadata for the specific image
        image_metadata = self.images[idx]
        image_key = f"{self.prefix}{image_metadata['file_name']}"  # Construct the full key with the prefix

        # Get image from S3
        img_obj = s3.get_object(Bucket=self.bucket, Key=image_key)
        image = Image.open(io.BytesIO(img_obj['Body'].read()))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Label can be species or any other metadata
        label = image_metadata['species']  # Adjust according to the needs, can be 'location', etc.

        return image, label


default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

metadata_file = settings.METADATA_FILE
bucket = settings.BUCKET
prefix = settings.PREFIX


def load_dataset(transform=default_transform, metadata_file=metadata_file, bucket=bucket, prefix=prefix, batch_size=10, test_split=0.2, random_seed=42,dataset_percentage=1,prefetch_factor=None,num_workers=1):
    sample_weight_pkl = 'trail_cam_ims_sample_weights.pkl'

    dataset = S3Dataset(bucket=bucket, metadata_file=metadata_file, prefix=prefix, transform=transform)

    # Split indices for training and testing
    dataset_size = int(len(dataset)*dataset_percentage)
    indices = list(range(dataset_size))
    subset_indices = np.random.choice(indices, dataset_size, replace=False)

    train_indices, test_indices = train_test_split(subset_indices, test_size=test_split, random_state=random_seed)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    with open(sample_weight_pkl, 'rb') as f:
        sample_weights = pickle.load(f)

    train_weights = [sample_weights[i] for i in train_indices]
    test_weights = [sample_weights[i] for i in test_indices]

    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,prefetch_factor=prefetch_factor,num_workers=num_workers,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, sampler=test_sampler,prefetch_factor=prefetch_factor,num_workers=num_workers,worker_init_fn=worker_init_fn)

    return train_loader, test_loader, dataset.label_file_name

