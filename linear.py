from linear_format import encode_data, extract_labels, NeuralNetwork, get_batches
from utils import  load_data
import params
import random, torch,pickle
from torch import nn
import numpy as np
from transformers import BertForSequenceClassification
from tqdm import tqdm

device = torch.device(params.device)

data = load_data(params.data_path + "train_data.json", map_relations=params.map_relations)

train_data = data[params.valid_size:]
valid_data = data[:params.valid_size]
        
input_text_train, labels_train, positions_train = encode_data(train_data)
input_ids_train, attention_masks_train, token_type_ids_train, tokens_train, position_ids_train = extract_labels(input_text_train, labels_train, positions_train, token = True)
position_ids_train = [position_id.clone().detach().to(device) for position_id in position_ids_train]

input_text_valid, labels_valid, positions_valid = encode_data(valid_data)
input_ids_valid, attention_masks_valid, token_type_ids_valid, tokens_valid, position_ids_valid = extract_labels(input_text_valid, labels_valid, positions_valid, token = True)
position_ids_valid = [position_id.clone().detach().to(device) for position_id in position_ids_valid]

model_path = params.model_path + params.bert_name + '.pth'
embedder = BertForSequenceClassification.from_pretrained(
    params.model_name, 
    output_attentions = False,
    output_hidden_states = True, attention_probs_dropout_prob=0, hidden_dropout_prob=0
)

checkpoint = torch.load(model_path, map_location=params.device)
embedder.load_state_dict(checkpoint['model_state_dict'])
embedder.to(device)
print('Loaded finetuned Bert model')

meta_data_train = []
for i in range(len(labels_train)):
    meta_data_train.append([[lb[2], lb[2]-lb[1]] for lb in labels_train[i]])

meta_data_valid = []
for i in range(len(labels_valid)):
    meta_data_valid.append([[lb[2], lb[2]-lb[1]] for lb in labels_valid[i]])

batches_train = get_batches(len(input_ids_train), params.batch_size_linear)
batches_valid = get_batches(len(input_ids_valid), params.batch_size_linear)


linear = NeuralNetwork().to(device)
linear.train()
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.AdamW(params=linear.parameters(), lr=params.lr_linear)


for epoch in range(params.epochs_linear) : 
    loss_sum_train = 0
    linear.train()
    for batch in tqdm(batches_train) : 
        loss = None
        for i in batch:
            output = embedder(input_ids_train[i].to(device), 
                                 token_type_ids=token_type_ids_train[i].to(device), 
                                 attention_mask=attention_masks_train[i].to(device),
                                 position_ids = position_ids_train[i].to(device),
                                 return_dict=True)
            H_embed = torch.stack([torch.cat((output.hidden_states[-1][cand][0],torch.tensor(meta_data_train[i][cand]).to(device)),0) for cand in range(len(output.hidden_states[0]))])            
            H_embed = H_embed.to(device)
            logits = linear(H_embed).unsqueeze(0) 

            logits = logits.squeeze(-1)
            target = torch.tensor([[float(lab[3]) for lab in labels_train[i]]]).to(device)
            
            if loss :
                loss += criterion(input=logits, target=target)
            else:
                loss = criterion(input=logits, target=target)

        loss = loss/len(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum_train += loss.item()

    linear.eval()
    loss_sum_valid = 0
    for batch in tqdm(batches_valid):
        loss = None
        for i in batch :
            with torch.no_grad():  
                output = embedder(input_ids_valid[i].to(device), 
                         token_type_ids=token_type_ids_valid[i].to(device), 
                         attention_mask=attention_masks_valid[i].to(device),
                         position_ids = position_ids_valid[i].to(device),
                         return_dict=True)
            H_embed = torch.stack([torch.cat((output.hidden_states[-1][cand][0],torch.tensor(meta_data_valid[i][cand]).to(device)),0) for cand in range(len(output.hidden_states[0]))])           
            H_embed = H_embed.to(device)
            with torch.no_grad():  
                logits = linear(H_embed).unsqueeze(0) 

            logits = logits.squeeze(-1)
            target = torch.tensor([[float(lab[3]) for lab in labels_valid[i]]]).to(device)
                

            # compute loss
            if loss :
                loss += criterion(input=logits, target=target)
            else:
                loss = criterion(input=logits, target=target)
        loss = loss/len(batch)
        loss_sum_valid += loss.item()

output_model = params.model_path + params.linear_name + '.pth' 

print('finished_training, saving to : ', output_model)
        
torch.save({
    'model_state_dict': linear.state_dict(),
}, output_model)