import os
from utils import load_data
import torch, random, time
import numpy as np
import params
from bert_format import input_format, position_ids_compute, undersample, format_time, flat_accuracy 
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
from torch import nn
from multitask_format import Task, MultiTaskModel

device = torch.device(params.device)

num_labels = 18

data = load_data(params.data_path + "train_data.json", map_relations=params.map_relations)

if params.data_path == "data/stac_data/"  :
    data[343]['relations'].remove({'type': 14, 'x': 18, 'y': 78})

if params.data_path == "data/stac_squished_data/" :

    data[177]['relations'].remove({'type': 17, 'x': 5, 'y': 41})

    data[178]['relations'].remove({'type': 3, 'x': 40, 'y': 76})

    
    data[322]['relations'].remove({'type': 17, 'x': 4, 'y': 164})
    data[322]['relations'].remove({'type': 1, 'x': 53, 'y': 151})
        
    data[474]['relations'].remove({'type': 11, 'x': 14, 'y': 59})
    data[474]['relations'].remove({'type': 2, 'x': 18, 'y': 59})
    
#     data[560]['relations'].remove({'type': 9, 'x': 18, 'y': 43})
    data[562]['relations'].remove({'type': 4, 'x': 34, 'y': 66})
    data[578]['relations'].remove({'type': 17, 'x': 44, 'y': 83})
    data[581]['relations'].remove({'type': 17, 'x': 19, 'y': 63})
        
    data[884]['relations'].remove({'type': 14, 'x': 23, 'y': 83})

train_data = data[params.valid_size:]
valid_data = data[:params.valid_size]

# attachment prediction data 
input_ids, attention_masks, token_type_ids, tokens, labels_attach, labels, raw = input_format(train_data, token = False)
position_ids = position_ids_compute(input_ids, raw, labels)
attach_labels_complete, attach_labels, attach_input_ids, attach_attention_masks, attach_token_type_ids, attach_position_ids = undersample(params.n_keep_multi, labels, labels_attach, input_ids, attention_masks, token_type_ids, position_ids)
attach_task_ids = torch.tensor([0 for i in range(len(attach_labels))])

# relation prediction data
relation_input_ids, relation_attention_masks, relation_token_type_ids, relation_tokens, relation_labels, relation_labels_complete, relation_raw = input_format(train_data, relations=True, token = False)
relation_position_ids = position_ids_compute(relation_input_ids, relation_raw, relation_labels_complete)
relation_task_ids = torch.Tensor([1 for i in range(len(relation_labels))])

# regroup the attach and relation datasets 
pad_value = np.shape(attach_input_ids)[1]-np.shape(relation_input_ids)[1]
relation_input_ids = nn.functional.pad(input=relation_input_ids, pad=(0,pad_value), mode='constant', value=0)
relation_attention_masks = nn.functional.pad(input=relation_attention_masks, pad=(0,pad_value), mode='constant', value=0)
relation_token_type_ids = nn.functional.pad(input=relation_token_type_ids, pad=(0,pad_value), mode='constant', value=0)
relation_position_ids = nn.functional.pad(input=relation_position_ids, pad=(0,pad_value), mode='constant', value=0)

attach_task = Task(id = 0, name = 'attach prediction', type = "seq_classification", num_labels=2)
relation_task = Task(id = 1, name = 'relation prediction', type = "seq_classification", num_labels = num_labels)
tasks = [attach_task, relation_task]

input_ids = torch.cat((attach_input_ids,relation_input_ids))
attention_masks = torch.cat((attach_attention_masks, relation_attention_masks))
token_type_ids = torch.cat((attach_token_type_ids, relation_token_type_ids))
position_ids = torch.cat((attach_position_ids, relation_position_ids)) 
labels = torch.cat((attach_labels ,relation_labels))
task_ids = torch.cat((attach_task_ids ,relation_task_ids))

dataset = TensorDataset(input_ids, attention_masks, token_type_ids, position_ids, labels, task_ids)

train_dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset),
            batch_size = params.batch_size_bert
        )

model = MultiTaskModel(params.model_name, tasks)
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = params.lr_bert, 
                  eps = params.eps_bert 
                )

training_stats = []

seed_val = params.seed

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if params.device == 'cuda' :
    torch.cuda.manual_seed_all(seed_val)

total_t0 = time.time()

for epoch_i in range(params.epoch_multitask):
    
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params.epochs_bert))
    
    t0 = time.time()

    total_train_loss = 0
    
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
        model.zero_grad()
        outputs, embed = model(input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device),
                token_type_ids=batch[2].to(device),
                position_ids=batch[3].to(device),
                labels=batch[4].to(device),
                task_ids=batch[5].to(device)
                        )
        loss = outputs[0]

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)       
        training_time = format_time(time.time() - t0)

torch.save({
    'model_state_dict': model.state_dict(),
}, params.model_path + 'bert_multitask.pth')

