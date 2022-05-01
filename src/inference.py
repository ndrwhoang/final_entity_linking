import os
import logging
import random
random.seed(123)
from tqdm.auto import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Inference:
    def __init__(self, config, accelerator, model, dataset):
        self.config = config
        self.model = model
        self.accelerator = accelerator
        self.dataset = dataset
        
        self._set_seed()
        self._init_model(self.config['data_path']['ckpt_path'])
        self._init_dataloader(dataset)
        
    def _set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _init_model(self, model_path):
        if model_path:
            path = os.path.join(*model_path.split('\\'))
            self.model.load_state_dict(torch.load(path))
            logger.info(f' Loaded checkpoint from {model_path}')
        
        self.model.eval()
        self.model = self.accelerator.prepare(self.model)
        
    def _init_dataloader(self, dataset):
        loader = DataLoader(dataset,
                            batch_size=int(self.config['training']['bsz_test']),
                            collate_fn=dataset.collate_fn,
                            shuffle=False,
                            drop_last=False
                            )
        self.loader = self.accelerator.prepare(loader)
    
    def _make_label_vec(self, ctx_bs, cand_bs, device):
        cand_group = int(cand_bs / ctx_bs)
        labels = torch.arange(0, cand_bs, cand_group)
        labels = labels.to(device)
        
        return labels  
    
    @torch.no_grad()
    def run_test(self):
        pbar = tqdm(self.loader)
        preds, labels = [], []
        n_samples = self.dataset.__len__()
        
        for i, batch in enumerate(pbar):            
            scores = self.model(batch, 'scores')
            pred = torch.argmax(scores, dim=1)
            label = self._make_label_vec(batch.context_ids.size(0), 
                                         batch.candidate_ids.size(0),
                                         batch.context_ids.device)
            
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
            
        n_rights = sum([1 for pred, label in zip(preds, labels) if pred == label])
        
        logger.info(f' Test set accuracy {n_rights/n_samples}')

if __name__ == '__main__':
    import configparser
    
    from accelerate import Accelerator
    
    from src.dataset.dataset import ELDataset
    from src.model.biencoder import Reranker
    from src.trainer import Trainer
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    data_path = 'data\\raw\zeshel\mentions\\val.json'
    
    accelerator = Accelerator(mixed_precision='no', cpu=True)
    dataset = ELDataset(config, data_path)
    model = Reranker(config)
    
    inference = Inference(config, accelerator, model, dataset)
    inference.run_test()