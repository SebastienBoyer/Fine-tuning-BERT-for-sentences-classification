import config
import pandas as pd
import dataset
import torch
import torch.nn as nn
import engine
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import seaborn as sns
import json


from sklearn import model_selection
from sklearn.utils.class_weight import compute_class_weight
from model import BERTGoEmotion #don't forget to change the name of the model here if you changed it in model.py
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from operator import itemgetter



plt.rc("font", size=15)
pylab.rcParams['figure.figsize'] = 10, 10




"""
The train_eval dataset should be a csv file. The columns header in this csv states what are our features and targets.
So in this particular case "sentences" column regroup my features and "emotion" column is my target.
"""


def run():
	df = pd.read_csv(config.DATA_FILE)
	df_train, df_valid = model_selection.train_test_split(
		df,
		test_size=0.2,
		random_state=42,
		stratify= df.emotion.values
	)# the train validation splitting occurs here and is stratified by our target (here emotion)

	adresse = "/".join(config.DATA_FILE.split("/")[:-1])+"/"

	df_train.to_csv(adresse+"val_train/train.csv")#keep a trace of the training and validation set
	df_valid.to_csv(adresse+"val_train/val.csv")

	w = compute_class_weight('balanced', np.unique(df_train.emotion.values), df_train.emotion.values)
	weights_classes = torch.tensor(w,dtype=torch.float32)

	df_train = df_train.reset_index(drop=True)
	
	df_valid = df_valid.reset_index(drop=True)
	
	train_dataset = dataset.BERTDataset(
		sentences=df_train.sentences.values,
		target=df_train.emotion.values
	)
	
	train_data_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=config.TRAIN_BATCH_SIZE,
		num_workers=16 #change that accordingly to the number of cpu you have and how much they can handle and your batch size
	)

	
	valid_dataset = dataset.BERTDataset(
		sentences=df_valid.sentences.values,
		target=df_valid.emotion.values
	)

	valid_data_loader = torch.utils.data.DataLoader(
		valid_dataset,
		batch_size=config.VALIDATION_BATCH_SIZE,
		num_workers=16 #change that accordingly to the number of cpu you have and how much they can handle and your batch size
	)

	device = torch.device("cuda")

	model = BERTGoEmotion(len(set(df.emotion)))
	model.to(device)

	if config.MULTI_GPUS==True:
		model = nn.DataParallel(model)
	
	num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

	#Moreover here all the weights on the model are optimized, even the one from BERT pre trained (their initial states are
	#the pre trained one so after 4 epochs or so you should not be too far from them, but still they change). In some case
	# you might want to optimize only the weights on the layer(s) you added. For that add to the code.
	
	param_optimizer = list(model.named_parameters())
	param_opti = [v for v in open(config.PARAM_OPTI).read().splitlines()]
	if len(param_opti)!=0:
		param_to_freeze=[ v[1] for v in param_optimizer if v[0] not in param_opti]
		for param in param_to_freeze:
				param.requires_grad = False  

	#print(param_optimizer)# Uncomment that to make sure which layers are on not frozen
	param_optimizer = list(model.named_parameters())
	
	#I am not too sure why some parameters should not be regularized. You can try to put wathever you think make sense
	#here.You don't need to put the exact name of the layer, since it is looking for keyword in the layers.
	no_decay = [v for v in open(config.PARAM_DECAY).read().splitlines()]# Weights that should not be affected by weight decay
	#I am not too sure why some parameters should not be regularized, but there is an example like that on Huggingface
	#["bias","LayerNorm.bias","LayerNorm.weight"]. You can try to put wathever you think make sense to you here.
	optimizer_parameters = [
		{"params":[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},#basically the L2 strength for the coef of the linear regression (play with that)
		{"params":[p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
	]
	optimizer = AdamW(optimizer_parameters, lr=5e-5)#You should play with the learning rate
	
	

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=num_train_steps
	)# You can try some warm up step but it seems a bit strange as the usual number of training step in fine tunning is small. Just try it if you feel like it.

	best_f1 = 0

	f1_list=[]
	f1_train_list=[]

	loss_train_list=[]
	loss_list=[]

	
	for epoch in range(config.EPOCHS):
		loss_train, outputs_train, target_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler,weights_classes)
		loss, outputs, targets = engine.eval_fn(valid_data_loader, model, device,weights_classes)
		
		_, preds = torch.max(torch.tensor(outputs),dim=1)
		
		_, preds_train = torch.max(torch.tensor(outputs_train),dim=1)

		f1 = metrics.f1_score(targets, preds,average='weighted')# I used weighted f1 score as it deal nicely with imbalanced dataset.
		f1_train = metrics.f1_score(target_train, preds_train,average='weighted')# Change to whatever you prefer and make sense to you.

		print(f"F1 score = {f1},   F1 train score = {f1_train}",flush=True)
		print(f"Loss = {loss},   Loss train score = {loss_train}",flush=True)

		f1_list.append(f1)
		f1_train_list.append(f1_train)

		loss_list.append(loss)
		loss_train_list.append(loss_train)

		if f1 > best_f1:#model saving is based on validation set best score...Work whatever fits you here.
			torch.save(model.state_dict(), config.MODEL_PATH)
			best_f1 = f1
			final_preds = preds
			print(f"Best epoch so far = {epoch}",flush=True)

	#list(model.named_parameters())

	# in the following I plot and save as a pdf the loss fuction and score as well as the confusion matrix for the validation set
	# under the kept model
	ee=np.arange(0,config.EPOCHS,1)
	plt.plot(ee,loss_list,'r-',label='Loss_val')
	plt.plot(ee,loss_train_list,'b-',label='Loss_train')
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig('loss.pdf')
	
	plt.close()
	plt.plot(ee, f1_list,'r-',label='F1_val')
	plt.plot(ee,f1_train_list,'b-',label='F1_train')
	plt.ylabel("F1_score")
	plt.xlabel("Epochs")
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig('f1.pdf')
	
	plt.close()

	with open(adresse+config.DICO_NAME) as json_file:
        	dico_emo1 = json.load(json_file)

	dico_emo={dico_emo1[s]:s for s in dico_emo1.keys()}
	confusion_mc_c = confusion_matrix(targets, final_preds)
	df_cm_c = pd.DataFrame(confusion_mc_c, 
                     index = [dico_emo[i] for i in range(len(dico_emo))], columns = [dico_emo[i] for i in range(len(dico_emo))])

	plt.figure(figsize=(10,10))
	sns.heatmap(df_cm_c, annot=True)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig('confusion.pdf')
	
	plt.close()
	print(classification_report(targets,final_preds),flush=True)

		
if __name__=="__main__":
	run()














