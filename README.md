# LineBert
This repository contains the code associated to the LineBert paper. It indroduces a simple method for discourse parsing in dialogue.
 Scores | Attachments| Relations|
--- | --- | --- 
STAC | 73.06 | 56.25
STAC-squished | 79.93|71.22

# Data sets 
The two data sets, stac and stac_squished can be found [here](https://drive.google.com/drive/folders/1tFBhq3rRbQGcq5J6Lga8-YPSaNb_pfmd?usp=sharing)

# Setup
All the requirement for LineBert can be found in `requirements.txt`. For training or testing, the data sets should be put in the corresponding folder.  
For training, run the following commands:
```
python bert_finetune.py
python linear.py
python multitask.py
```
- For evaluation add the trained [models](https://drive.google.com/drive/folders/1S7ICsu5QUAjouOuDCmO_CcFAr3RnsUdr?usp=sharing) to the `models` folder, or use the .pth obtained after training, and run the the notebooks `bert_finetune_test.ipynb` and `multitask_test.ipynb`.

