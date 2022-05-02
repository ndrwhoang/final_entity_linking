import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_test():
    import os
    import configparser
    
    from accelerate import Accelerator
    
    from src.model.biencoder import Reranker
    from src.dataset.dataset import ELDataset
    from src.inference import Inference
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.ini'))
    
    data_path = 'data\\raw\zeshel\mentions\\val.json'
    ckpt_path = config['data_path']['ckpt_path']
    
    accelerator = Accelerator(mixed_precision='no')
    dataset = ELDataset(config, data_path, 'test')
    model = Reranker(config)
    
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR) 
    logger.info('Accelerator settings')
    logger.info(accelerator.state)
    logger.info(f' Pretrained Encoder: {ckpt_path}')
    
    inference = Inference(config, accelerator, model, dataset)
    inference.run_test()

if __name__ == '__main__':
    print('hello world')
    run_test()