# LineBert
This repository contains the code associated to the LineBert paper. It indroduces a simple method for discourse parsing in dialogue.
Attempt | Stac| Stac_squished|
--- | --- | --- 
Attachments | 73.06 | X 
Relations | X|X

# Data sets 
Stac (link)
Stac Squished (link)
(minecraft)  

# Setup
- All the requirement for LineBert can be found in `requirements.txt`.
- Add the data sets in the corresponding folder.
- For training the following commands
```
python bert_finetune.py
python linear.py
python multitask.py
```
- For evaluation add the trained models (link) to the `models` folder.
add the models in link to the corresponding model folder. run the notebook linear_test.py before running multitask_test.py to get f1 score on relation prediction.
