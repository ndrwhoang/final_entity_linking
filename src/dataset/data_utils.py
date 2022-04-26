import os
import json
import glob
from tqdm.auto import tqdm

def clean_path(path):
    return os.path.join(*path.split('\\'))
    
def read_line_delim_json(path):
    path = clean_path(path)
    out = []
    
    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            out.append(sample)
    
    return out     

def collect_files(dir):
    dir += '\*.json'
    dir = clean_path(dir)
    file_path = glob.glob(dir)
    
    return file_path

def read_knowledge(path):
    document_files = collect_files(path)
    
    documents = {}
    for file in tqdm(document_files):
        with open(file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                documents[sample['document_id']] = {
                    'title': sample['title'],
                    'text': sample['text']
                }

    return documents