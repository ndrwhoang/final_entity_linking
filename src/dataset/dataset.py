from tqdm.auto import tqdm
import json
import logging
from typing import NamedTuple, List
from itertools import chain

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from src.dataset.data_utils import (clean_path, 
                                    read_knowledge)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Batch(NamedTuple):
    context_ids: torch.Tensor
    context_attention_mask: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_attention_mask: torch.Tensor

class ELDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['pretrained_name'])
        documents = read_knowledge(self.config['data_path']['document_path'])
        mentions, candidates = self._read_data(data_path)
        
        self.input_strs, self.candidate_strs = \
            self._convert_to_samples(documents, mentions, candidates)
        self.n_samples = len(self.input_strs)
        logger.info(f'Loaded data from {data_path}, n_samples: {self.n_samples}')
        
    def _read_data(self, sample_path):
        candidate_path = sample_path.replace('zeshel', '').replace('mentions', 'tfidf_candidates')
        sample_path = clean_path(sample_path)
        candidate_path = clean_path(candidate_path)
                    
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
    
    def _convert_to_samples(self, documents, mentions, candidates):
        def process_sample(mention, documents, candidates):
            # Getting mention keys and indices
            context_document_id = mention['context_document_id']
            gold_document_id = mention['label_document_id']
            mention_id = mention['mention_id']
            start_idx = mention['start_index']
            end_idx = mention['end_index']
        
            # Extracing mention string
            context_str = documents[context_document_id]['text']
            mention_str = ' '.join(context_str.split()[start_idx : end_idx+1])
            input_str = f' {self.tokenizer.sep_token} '.join([mention_str, context_str])
            
            # Gathering candidates, filter outduplicate gold, adding gold document at head
            candidate_ids = candidates[mention_id]
            if len(candidate_ids) == 0: return None, None
            candidate_ids = [cand for cand in candidate_ids if cand != gold_document_id]
            candidate_ids.insert(0, gold_document_id)
            candidate_ids = candidate_ids[:int(self.config['data']['n_hard_negs'])+1]
            padding = candidate_ids[-1]
            while len(candidate_ids) < 1 + int(self.config['data']['n_hard_negs']):
                candidate_ids.append(padding)
            
            # Building list of hard negatives
            candidate_strs = []
            for candidate_id in candidate_ids:
                candidate_str = f' {self.tokenizer.sep_token} '.join([mention_str, documents[candidate_id]['text']])
                candidate_strs.append(candidate_str)
        
            return input_str, candidate_strs
            
                
        input_strs, candidate_strs = [], []
        n_fails = 0
        failed_mention_ids = []
        
        for mention in tqdm(mentions):
            input_str, cands = process_sample(mention, documents, candidates)
            # try:
            #     input_str, cands = process_sample(mention, documents, candidates)
            # except:
            #     n_fails += 1
            #     failed_mention_ids.append(mention['mention_id'])
            #     input_str, cands = None, None
            #     continue
            if cands:
                input_strs.append(input_str)
                candidate_strs.append(cands)
            
        logger.info(f' Failed to parse {n_fails} samples: {failed_mention_ids}')
            
        return input_strs, candidate_strs
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.input_strs[index], self.candidate_strs[index]
    
    def collate_fn(self, batch):
        input_strs, candidate_strs = zip(*batch)
        candidate_strs = list(chain.from_iterable(candidate_strs))
        
        input_encodings = self.tokenizer(list(input_strs),
                                         truncation=True,
                                         padding=True,
                                         return_attention_mask=True,
                                         return_tensors='pt',
                                         max_length=512)
        candidate_encodings = self.tokenizer(list(candidate_strs),
                                             truncation=True,
                                             padding=True,
                                             return_attention_mask=True,
                                             return_tensors='pt',
                                             max_length=512)
        
        return Batch(input_encodings['input_ids'], 
                     input_encodings['attention_mask'],
                     candidate_encodings['input_ids'],
                     candidate_encodings['attention_mask'])
        
if __name__ == '__main__':
    print('hello world')
    import os
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    tokenizer = 'microsoft/deberta-v3-xsmall'
    dataset = ELDataset(config, 'data\\raw\zeshel\mentions\\val.json')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        print(batch.context_ids.size())
        print(batch.context_attention_mask.size())
        print(batch.candidate_ids.size())  
        print(batch.candidate_attention_mask.size())  
        print('============================')
    