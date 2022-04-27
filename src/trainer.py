import os
import logging
import configparser
import json
from tqdm.auto import tqdm
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_constant_schedule_with_warmup as warmup_scheduler
from accelerate import Accelerator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class Trainer:
    def __init__(self, 
                 config, 
                 model, 
                 accelerator,
                 train_dataset, 
                 val_dataset=None
                 ):
        self.config = config
        self.model = model
        self.accelerator = accelerator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset if val_dataset != None else train_dataset 

        self._set_seed()
        self._init_model()
        self._get_dataloaders()
        self._get_optimizer()
        
    def _set_seed(self):
        self.seed = 69420
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _init_model(self):
        if self.config['data_path']['ckpt_path'] != 'None':
            path = os.path.join(*self.config['data_path']['ckpt_path'].split('\\'))
            self.model.load_state_dict(torch.load(path))
        
        # self.model.to(self.device)
        self.model = self.accelerator.prepare(self.model)
        
    def _get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset,
                                batch_size=int(self.config['training']['bsz_train']),
                                collate_fn=self.train_dataset.collate_fn,
                                shuffle=True,
                                drop_last=True
                                )
        val_loader = DataLoader(self.val_dataset,
                                batch_size=int(self.config['training']['bsz_val']), 
                                collate_fn=self.val_dataset.collate_fn,
                                shuffle=False, 
                                drop_last=False)
        # self.train_loader, self.val_loader = train_loader, val_loader
        self.train_loader, self.val_loader = self.accelerator.prepare(train_loader, val_loader)

    def _get_optimizer(self):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias']
        optimized_params = [
            {
                'params':[p for n, p in model_params if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }   
        ]
        optimizer = AdamW(optimized_params, lr=float(self.config['training']['lr']))
        lr_scheduler = warmup_scheduler(optimizer, 
                                        int(self.config['training']['n_warmup_steps']))
        # self.optimizer = optimizer
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)

    def run_train(self):
        best_loss = self.run_validation()
        
        for epoch in range(int(self.config['training']['n_epochs'])):
            pbar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)
            self.model.train()
            self.model.zero_grad(set_to_zero=None)
            
            for i, batch in enumerate(pbar):
                batch_loss = self._training_step(batch)
                
                if i % int(self.config['training']['grad_accum_steps']):
                    self.optimizer.step()
                    self.model.zero_grad(set_to_none=True)
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.lr_scheduler.step()
                        
                pbar.set_description(f'(Training) Epoch: {epoch} Loss: {batch_loss:.4f}')
                
            val_loss = self.run_validation()
            
            if val_loss < best_loss:
                logger.info(f'New best validation loss at {val_loss:.4f}, saving checkpoint')
                best_loss = val_loss
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                ckpt_path = os.path.join(self.config['training']['output_dir'], 'model_ckpt.pt')
                torch.save(unwrapped_model.state_dict(), ckpt_path)
                # torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f'New checkpoint saved at {ckpt_path}')
                
            if (val_loss >= best_loss or epoch > 5) and self.model.frozen == True:
                logger.info(f'Unfreeze encoder at epoch {epoch}')
                self.model.frozen = False
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.00002
                for param in self.model.biencoder.parameters():
                    param.requires_grad = True
    
    def run_validation(self):
        pbar = tqdm(self.val_loader, disable=not self.accelerator.is_local_main_process)
        self.model.eval()
        val_loss = 0
        
        for i, batch in enumerate(pbar):
            batch_loss = self._prediction_step(batch)
            val_loss += batch_loss
            pbar.set_description(f'(Validating) Loss: {batch_loss:.4f}')
            
        logger.info(f' Validation loss: {val_loss:.4f}')
        
        return val_loss
    
    def _training_step(self, batch):
        loss = self.model(batch, return_type='loss')
        # loss.backward()
        loss = loss / int(self.config['training']['grad_accum_steps'])
        self.accelerator.backward(loss)
        
        return loss.detach()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        loss = self.model(batch)
        
        return loss.detach()
    
if __name__ == '__main__':
    print('hello world')
    import configparser
    
    from src.model.biencoder import Reranker
    from src.dataset.dataset import ELDataset
    
    # from transformers import AutoTokenizer, AutoModel
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    data_path = 'data\\raw\zeshel\mentions\\val.json'
    pretrained_model = 'microsoft/deberta-v3-xsmall'
    
    accelerator = Accelerator(mixed_precision='no', cpu=False)
    dataset = ELDataset(config, data_path)
    model = Reranker(config)
    
    trainer = Trainer(config, model, accelerator, dataset)
    trainer.run_train()