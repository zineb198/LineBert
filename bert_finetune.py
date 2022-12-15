from utils import load_data
import torch, random, time
import numpy as np
import params
from bert_format import input_format, position_ids_compute, undersample, format_time, flat_accuracy
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification

device = torch.device(params.device)

data = load_data(params.data_path + "train_data.json", map_relations=params.map_relations)

train_data = data[params.valid_size:]
valid_data = data[:params.valid_size]

input_ids, attention_masks, token_type_ids, tokens, labels_attach, labels, raw = input_format(train_data, token = False)
position_ids = position_ids_compute(input_ids, raw, labels)
labels, labels_attach, input_ids, attention_masks, token_type_ids, position_ids = undersample(params.n_keep, labels, labels_attach, input_ids, attention_masks, token_type_ids, position_ids)

input_ids_valid, attention_masks_valid, token_type_ids_valid, tokens_valid, labels_attach_valid, labels_valid, raw_valid = input_format(valid_data, token = False)
position_ids_valid = position_ids_compute(input_ids_valid, raw_valid, labels_valid)

train_dataset = TensorDataset(input_ids, attention_masks, token_type_ids, position_ids, labels_attach)

val_dataset = TensorDataset(input_ids_valid, attention_masks_valid, token_type_ids_valid, position_ids_valid, labels_attach_valid)

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = params.batch_size_bert 
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = params.batch_size_bert 
        )

model = BertForSequenceClassification.from_pretrained(
    params.model_name, 
    output_attentions = False,
    output_hidden_states = True, attention_probs_dropout_prob=0, hidden_dropout_prob=0)

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = params.lr_bert, 
                  eps = params.eps_bert 
                )

total_steps = len(train_dataloader) * params.epochs_bert

random.seed(params.seed)
if params.device == 'cuda' :
    torch.cuda.manual_seed_all(params.seed)

total_t0 = time.time()

for epoch_i in range(params.epochs_bert):
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params.epochs_bert))
    
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        model.zero_grad()  

        result = model(batch[0].to(device), 
                       token_type_ids=batch[2].to(device), 
                       attention_mask=batch[1].to(device), 
                       position_ids = batch[3].to(device),
                       labels=batch[4].to(device),
                       return_dict=True)

        loss = result.loss
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)       
    training_time = format_time(time.time() - t0)
    
    print("  Training Loss: ",avg_train_loss)
    print("  Training took: ", training_time)
    print("Running Validation")
    t0 = time.time()

    # Evaluation step
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        
        with torch.no_grad():        
            result = model(batch[0].to(device), 
                           token_type_ids=batch[2].to(device), 
                           attention_mask=batch[1].to(device),
                           position_ids = batch[3].to(device),
                           labels=batch[4].to(device),
                           return_dict=True)

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        if params.device == 'cuda' : 
            logits = logits.detach().cpu().numpy()
            label_ids = batch[4].to(device).to('cpu').numpy()

        # Compute the accuracy
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: ", avg_val_accuracy)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: ", avg_val_loss)
    print("  Validation took: ",validation_time)
print("Training complete!")

torch.save({
    'model_state_dict': model.state_dict(),
}, params.model_path + params.bert_name + '.pth')