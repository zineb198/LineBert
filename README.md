# LineBert
This repository contains the code associated to the LineBert paper. It indroduces a simple method for discourse parsing in dialogue.
Attempt | Stac| Stac_squished|
--- | --- | --- 
Attachments | 73.06 | X 
Relations | X|X

# Data sets 
Stac (link).  
Stac Squished (link).  
(minecraft).  

# Setup
All the requirement for LineBert can be found in `requirements.txt`. For training or testing, the data sets should be put in the corresponding folder.  
For training, run the following commands:
```
python bert_finetune.py
python linear.py
python multitask.py
```
- For evaluation add the trained models (link) to the `models` folder, or use the .pth obtained after training, and run the the notebooks `bert_finetune_test.ipynb` and `multitask_test.ipynb`.

