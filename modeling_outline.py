model_fn = model_fn_builder()
train_input_fn = file_based_input_fn_builder()
estimator.train(train_input_fn)

def model_fn_builder():
    def model_fn(features, labels):
        input_ids = features.['input_ids']
        input_mask = features['input_mask']
        mention_ids = features['mention_id']
        label_ids = features['label_id']
        
        (loss, logits) = create_zeshel_model(input_ids, input_mask, mention_ids, label_ids)
        output = TPUEstimatorSpec()
        
        return output
        
    return model_fn
    
def create_zeshel_model(input_ids, input_mask, mention_ids, label_ids):
    model = BertModel(input_ids, input_mask, mention_ids, segment_ids)
    model_out = model.get_pooled_output()
    loss = ce_loss(model_out, label_ids)
    
    return loss, logits

    