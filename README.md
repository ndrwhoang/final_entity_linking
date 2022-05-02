# Final Project: Zero Shot Entity Linking with Bi-Encoder
This is a re-implementation of [this model](https://arxiv.org/pdf/1911.03814.pdf) on the dataset of [this paper](https://arxiv.org/pdf/1906.07348.pdf). The model is a Bi-Encoder that scores a context with named entity (the string for the entity is marked) against 64 candidate entities (assumed to have already been provided through some candidate generation process, BM25 in this case). 

# How to Run
Install dependencies from `requirements.txt`, run `train.py` for training and `test.py` for testing. File paths and other model details can be found in `configs/config.ini`. See `nlp_entity_linking.ipynb` for example run on Google Colab, replace data paths with data from [here](https://github.com/lajanugen/zeshel)

Training note: On Colab's T4, 1 epoch takes ~2h50m

# Results
Accuracy on test set: 0.677
(Prediction deemed accurate when the model scores the gold entity candidate higher than all other candidates)

# Various Implementation/Formulation Details
- The encoder is DeBERTa xsmall (chosen with regards to the need of having 2 encoders and a large-ish batch size for in-batch negatives with restrictive resources).
- Assuming a gold candidate always exist. Otherwise, this can be handled by using a common 'unknown' gold candidate for NIL. (Also assuming a set of candidates of at least 1 exist)
- Trained with gradual unfreezing, gradient accumulation, and a pseudo early stopping (saving checkpoint on new lowest validation loss) 
- Trained with [in-batch negatives](https://www.sbert.net/examples/unsupervised_learning/CT_In-Batch_Negatives/README.html). Additional to hard negatives (other top-64 candidates for that entity), we also optimize the scoring against candidates of other entities of the same batch (assuming no overlap and duplicates, so no situation where a model has to score gold candidates against each other). Better explanation [here](https://github.com/facebookresearch/DPR/issues/110#issuecomment-800289075).

