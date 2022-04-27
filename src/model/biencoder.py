import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import logging as t_logging
t_logging.set_verbosity_error()

class BiEncoderModule(nn.Module):
    def __init__(self, config):
        super(BiEncoderModule, self).__init__()
        pretrained_name = config['model']['pretrained_name']
        self.context_enc = AutoModel.from_pretrained(pretrained_name)
        self.candidate_enc = AutoModel.from_pretrained(pretrained_name)
        
        freeze_layers = [int(l) for l in config['model']['freeze_layers'].split(',')]
        self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, freeze_layers):
        for layer in freeze_layers:
            for param in self.context_enc.encoder.layer[layer].parameters():
                param.require_grad = False
            for param in self.candidate_enc.encoder.layer[layer].parameters():
                param.require_grad = False
        for param in self.context_enc.embeddings.parameters():
            param.require_grad = False
        for param in self.candidate_enc.embeddings.parameters():
            param.require_grad = False
    
    def forward(self, contexts, context_mask , candidates, candidate_mask):
        context_emb = self.context_enc(input_ids=contexts, 
                                       attention_mask=context_mask).last_hidden_state[:, 0, :]
        candidate_emb = self.candidate_enc(input_ids=candidates, 
                                           attention_mask=candidate_mask).last_hidden_state[:, 0, :]
        
        return context_emb, candidate_emb
    
class Reranker(nn.Module):
    def __init__(self, config):
        super(Reranker, self).__init__()
        self.config = config
        self.biencoder = BiEncoderModule(self.config)
        if self.config['model']['freeze_layers'] != 'None':
            self.frozen = True
    
    def _make_label_vec(self, ctx_bs, cand_bs, device):
        cand_group = int(cand_bs / ctx_bs)
        labels = torch.arange(0, cand_bs, cand_group)
        labels = labels.to(device)
        
        return labels        
    
    def loss_fn(self, scores, labels):
        return F.cross_entropy(scores, labels)
    
    def score_candidates(self, ctx_vecx, cand_vecs):
        return torch.matmul(ctx_vecx, torch.transpose(cand_vecs, 0, 1))
    
    def forward(self, batch, return_type='loss'):
        assert return_type in ['loss', 'scores']
        
        context_emb, candidate_emb = self.biencoder(batch.context_ids,
                                                    batch.context_attention_mask,
                                                    batch.candidate_ids,
                                                    batch.candidate_attention_mask)
        scores = self.score_candidates(context_emb, candidate_emb)
        
        if return_type == 'loss':
            device = context_emb.device
            labels = self._make_label_vec(context_emb.size(0), 
                                        candidate_emb.size(0),
                                        device)
            try:
                loss = self.loss_fn(scores, labels)
            except ValueError:
                print(batch.context_ids.size())
                print(batch.candidate_ids.size())
                print(scores.size())
                print(labels.size())
                print(labels)
            return loss
        elif return_type == 'scores':
            return scores
        

if __name__ == '__main__':
    import os
    import torch
    import configparser
    
    from src.dataset.dataset import Batch
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    batch_size = 4
    input_len = 128
    context_ids = torch.randint(0, 2000, size=(batch_size, input_len), dtype=torch.int)
    context_attention_mask = (torch.rand(size=(batch_size, input_len)) < 0.25).int()
    candidate_ids = torch.randint(0, 2000, size=(batch_size*4, input_len), dtype=torch.int)
    candidate_attention_mask = (torch.rand(size=(batch_size*4, input_len)) < 0.25).int()
    
    batch = Batch(context_ids, context_attention_mask, candidate_ids, candidate_attention_mask)
    model = Reranker(config)
    # loss = model(batch, return_type='loss')
    # scores = model(batch, return_type='scores')
    
    # print(loss)
    # print(scores.size())
    
    print(model.biencoder.context_enc)
    