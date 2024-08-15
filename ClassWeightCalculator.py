import json
from collections import Counter
import pickle
from tqdm import tqdm
from pathlib import Path
import settings
import random

metadata_file = settings.METADATA_FILE
root_element = settings.ROOT_ELEMENT
id_property = settings.ID_PROPERTY
class_property = settings.CLASS_PROPERTY

def GenerateSampleWeights():
    outfile_name = Path(metadata_file).stem + '_sample_weights.pkl'


    with open(metadata_file, 'r') as file:
        metadata = json.load(file)

    class_list = [img[class_property] for img in metadata[root_element]]
    class_counts = Counter(class_list)
    total_samples = sum(class_counts.values())

    class_weights = {species:total_samples/count for species, count in class_counts.items()}

    #TODO normalising class weights

    sample_weights = [class_weights[sample[class_property]] for sample in tqdm(metadata[root_element], desc="Calculating Sample Weights")]

    with open(outfile_name, 'wb') as f:
        pickle.dump(sample_weights, f)

    return class_weights

def GenerateSampleWeights(subset: float = 1.0):
    if not (0.0 < subset <= 1.0):
        raise ValueError("Subset parameter must be between 0.0 and 1.0.")

    outfile_name = Path(metadata_file).stem + '_sample_weights.pkl'

    with open(metadata_file, 'r') as file:
        metadata = json.load(file)

    images = metadata[root_element]

    total_images = len(images)
    subset_size = int(total_images * subset)

    subset_images = random.sample(images, subset_size)

    class_list = [img[class_property] for img in subset_images]
    class_counts = Counter(class_list)
    total_samples = sum(class_counts.values())

    class_weights = {species: total_samples / count for species, count in class_counts.items()}

    sample_weights = [class_weights[img[class_property]] for img in tqdm(subset_images, desc="Calculating Sample Weights")]

    return sample_weights
