import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = '[CLS] I want to [MASK] the car because it is cheap . Malaria is a [MASK] disease [SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the segments tensors.
segments_ids = [0] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]']
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)



for i in masked_index:
      #masked_index = tokenized_text.index(word)
      predicted_index = torch.argmax(predictions[0, 1]).item()
      predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
      print(predicted_token)