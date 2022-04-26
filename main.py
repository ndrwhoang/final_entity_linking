
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_train():
    import os
    import configparser
    
    from accelerate import Accelerator
    
    from src.model.biencoder import Reranker
    from src.dataset.dataset import ELDataset
    from src.trainer import Trainer
    
    # from transformers import AutoTokenizer, AutoModel
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    data_path = 'data\\raw\zeshel\mentions\\val.json'
    pretrained_model = config['model']['pretrained_name']
    
    accelerator = Accelerator(mixed_precision='fp16', cpu=False)
    dataset = ELDataset(config, data_path)
    model = Reranker(config)
    
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR) 
    logger.info('Accelerator settings')
    logger.info(accelerator.state)
    logger.info(f' Pretrained Encoder: {pretrained_model}')
    logger.info(f" Freezing layers: {config['model']['freeze_layers']}")
    
    trainer = Trainer(config, model, accelerator, dataset)
    trainer.run_train()
    

if __name__ == '__main__':
    print('hello world')