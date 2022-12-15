from transformers import BertTokenizer
import params
import torch
from torch import nn

# same as in bert_format
def tokenize(input, tokenizer, token = False): 

    device = torch.device(params.device)
    batch_tokenized = tokenizer(input, return_tensors="pt", padding=True, add_special_tokens=True) # get tokens id for each token (word) in the dialog
    input_ids = batch_tokenized["input_ids"].to(device) # list of token ids of dialogs in batch 
    attention_masks = batch_tokenized["attention_mask"].to(device) # cuda
    token_type_ids = batch_tokenized["token_type_ids"].to(device)
    tokens = []

    if token : 
        for t in batch_tokenized["input_ids"]:
            tokens += [tokenizer.convert_ids_to_tokens(t)]
    else : tokens = None

    return input_ids, attention_masks, token_type_ids, tokens

def encode_data(data, test = False):
    ''' This version of the function works for the linear data structure'''
    max_distance = params.max_distance

    # build the samples and targets :
    input_text, labels, raw = [], [], []
    for i in range(len(data)):
        raw_text = [j["speaker"][:6] + ": " + j["text_raw"][:-1] for j in data[i]["edus"] ]
        raw += [raw_text]
        
        temp = [[ [i, cand, y, 0, -1 ] for cand in range(y)] for y in range(1,len(data[i]["edus"]))]
        for rel in data[i]['relations']:
            temp[rel['y']-1][rel['x']][3] = 1 
            temp[rel['y']-1][rel['x']][4] = rel['type'] 
        
        labels += temp
        input_text += [[[raw_text[k-cand],raw_text[k]] for cand in range(k,0,-1)] for k in range(1,len(raw_text))]

    # positions matrix 
    tokenizer = BertTokenizer.from_pretrained(params.model_name, use_fast=True)
    ids = [tokenizer(raw[i], return_tensors="pt", padding=True, add_special_tokens=True)['input_ids'] for i in range(len(raw))]

    # compute position matrix
    positions = []
    for dialog in ids :
        temporary = []
        counter = 0
        for i in range(len(dialog)):
            position_vector = [counter+j for j in range(1, len(dialog[i])) if dialog[i][j] != 0]
            counter += len(position_vector)
            temporary += [position_vector]
        positions += [temporary]
        
    
    if test :
        return input_text, labels, positions
    
    # delete elements with distance > max_distance
    labels = [temp[-max_distance:] for temp in labels]
    input_text = [temp[-max_distance:] for temp in input_text]
    
    return input_text, labels, positions

def extract_labels(input_text, labels, positions, token = False):
    ''' Puts the data in the correct format for Linear.py'''
    input_ids, attention_masks, token_type_ids, tokens, position_ids = [], [], [], [], []
    tokenizer = BertTokenizer.from_pretrained(params.model_name, use_fast=True)

    for i in range(len(input_text)):
        input_ids_, attention_masks_, token_type_ids_, tokens_ = tokenize(input_text[i], tokenizer, token)
        position_ids_ = []
        for e, label in enumerate(labels[i]) : 
            position_ids_vector_ = [0]
            position_ids_vector_ += positions[label[0]][label[1]]
            position_ids_vector_ += positions[label[0]][label[2]]
            position_ids_vector_ = [t-position_ids_vector_[1]+1 if t != 0  else 0 for t in position_ids_vector_]
            position_ids_vector_ += [0 for i in range(len(input_ids_[e])-len(position_ids_vector_))]
            position_ids_ += [position_ids_vector_]
        position_ids_ = torch.tensor(position_ids_)
        input_ids += [input_ids_]
        attention_masks += [attention_masks_]
        token_type_ids += [token_type_ids_]
        tokens += [tokens_]
        position_ids += [position_ids_] 
        
    return input_ids, attention_masks, token_type_ids, tokens, position_ids

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear = nn.Sequential(
            nn.Dropout(p=0.3), 
            nn.Linear(params.hidden_size, params.hidden_size_1), 
            nn.Dropout(p=0.3),
            nn.Tanh(), 
            nn.Linear(params.hidden_size_1, 1))
        
        
    def forward(self, x):
        logits = self.linear(x)
        return logits

def get_batches(len_data, batch_size):
    indices = [i for i in range(len_data)]
    batches = []
    for i in range(len_data // batch_size + bool(len_data) % batch_size):
        batches.append(indices[i * batch_size:(i + 1) * batch_size])
    return batches