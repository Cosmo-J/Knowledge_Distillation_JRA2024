import json
from collections import Counter
import pickle
from tqdm import tqdm
from pathlib import Path

metadata_file = 'trail_cam_ims.json'
root_element = 'images'
id_property = 'file_name'
class_property = 'species'

outfile_name = Path(metadata_file).stem + '_sample_weights.pkl'


with open(metadata_file, 'r') as file:
    metadata = json.load(file)

class_list = [img[class_property] for img in metadata[root_element]]
class_counts = Counter(class_list)
total_samples = sum(class_counts.values())

class_weights = {species:total_samples/count for species, count in class_counts.items()}
print("class weights")    

sample_weights = [class_weights[sample[class_property]] for sample in tqdm(metadata[root_element], desc="Calculating Sample Weights")]

with open(outfile_name, 'wb') as f:
    pickle.dump(sample_weights, f)
