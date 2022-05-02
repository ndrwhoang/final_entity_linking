import os
import tarfile
import json
import glob
from tqdm.auto import tqdm
from typing import NamedTuple, List

import torch

def _clean_path(path):
    return os.path.join(*path.split('\\'))

def extract_raw_bz2(path):
    path = _clean_path(path)
    path_out = os.path.join(*path.split('\\')[:-1])
    tar = tarfile.open(path, 'r:bz2')
    tar.extractall(path_out)
    tar.close()
    
def read_line_delim_json(path):
    path = _clean_path(path)
    out = []
    
    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            out.append(sample)
    
    return out     

def _collect_files(dir):
    dir += '\*.json'
    dir = _clean_path(dir)
    file_path = glob.glob(dir)
    
    return file_path

def _read_knowledge(document_path):
    document_files = _collect_files(document_path)
    
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

def _read_data(sample_path):   
    candidate_path = sample_path.replace('zeshel', '').replace('mentions', 'tfidf_candidates')
    sample_path = _clean_path(sample_path)
                
    mentions = []
    with open(sample_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            mentions.append(sample)
                
    candidates = {}
    with open(candidate_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            candidates[sample['mention_id']] = sample['tfidf_candidates']   
    
    return mentions, candidates

class Sample(NamedTuple):
    input_str: str
    candidate_strs: List[str]

class InputSample(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    segment_ids: torch.Tensor

def process_sample(sample, documents, candidates, tokenizer, n_negatives=3):
    # Getting mention keys and indices
    context_document_id = sample['context_document_id']
    gold_document_id = sample['label_document_id']
    mention_id = sample['mention_id']
    start_idx = sample['start_index']
    end_idx = sample['end_index']
    
    # Extracing mention string
    context_str = documents[context_document_id]['text']
    mention_str = ' '.join(context_str.split()[start_idx : end_idx+1])
    input_str = f' {tokenizer.sep_token} '.join([mention_str, context_str])
    
    # Gathering candidates, filter outduplicate gold, adding gold document at head
    candidate_ids = candidates[mention_id]
    if not candidate_ids: return None
    candidate_ids = [cand for cand in candidate_ids if cand != gold_document_id]
    candidate_ids.insert(0, gold_document_id)
    candidate_ids = candidate_ids[:n_negatives+1]
    
    # Building list of hard negatives
    candidate_strs = []
    for candidate_id in candidate_ids:
        candidate_str = f' {tokenizer.sep_token} '.join([mention_str, documents[candidate_id]['text']])
        candidate_strs.append(candidate_str)
        
        # candidate_document = tokenizer.tokenize(candidate)
        
        # input_tokens = f' {tokenizer.sep_token} '.join([gold_document, candidate_document])
        # tokenizer_out = tokenizer.convert_tokens_to_ids(input_tokens)
        # input_ids = tokenizer_out['input_ids']
        # attention_mask = tokenizer_out['attention_masks']
        # segment_ids = [0]*(len(gold_document)+2) + [1]*(len(candidate_document)+1)
        
    return Sample(input_str, candidate_strs) 
    
def make_traning_samples(documents, samples, candidates):
    out = []
    
    for sample in samples:
        processed_sample = process_sample(sample, documents, candidates)
        out.append(processed_sample)
    
    return out



if __name__ == '__main__':
    print('hello world')
    
    # path = 'data\\raw\zeshel.tar.bz2'    
    # extract_raw_bz2(path)
    
    document_path = 'data\\raw\zeshel\documents'
    sample_path = 'data\\raw\zeshel\mentions\\val.json'
    # candidate_path = 'data\\raw\\tfidf_candidates'
    # documents = _read_knowledge(document_path)
    # samples, candidates = _read_data(sample_path)
    
    # print(len(documents))
    # print(len(samples))
    # print(len(candidates))
    
    
    
    