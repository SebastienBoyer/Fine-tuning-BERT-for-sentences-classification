import config
import pandas as pd
import dataset
import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import os

from tqdm import tqdm
from model import BERTGoEmotion #don't forget to change the name of the model here if you changed it in model.py

ap = argparse.ArgumentParser()
ap.add_argument("-d",'--directoryname',required=True,help="path to the directory containing all the files to predict")
args=vars(ap.parse_args())

adresse="/".join(config.DATA_FILE.split("/")[:-1])+"/"

with open(adresse+config.DICO_NAME) as json_file:
	dico_emo1 = json.load(json_file)

dico_emo={dico_emo1[s]:s for s in dico_emo1.keys()}
print(dico_emo)

def predict_fn(data_loader, model, device):

	model.eval()
	
	fin_outputs = []
	

	with torch.no_grad():
		for bi, d in tqdm(enumerate(data_loader),total=len(data_loader)):
			#sent=d.get["sentences"]
			ids = d["ids"]
			token_type_ids = d["token_type_ids"]
			mask = d["mask"]
			

			ids = ids.to(device, dtype=torch.long)
			token_type_ids = token_type_ids.to(device, dtype=torch.long)
			mask = mask.to(device, dtype=torch.long)
			

			
			outputs= model(
				ids=ids,
				mask=mask,
				token_type_ids=token_type_ids
			)
			


			sf = nn.Softmax(dim=1)
			fin_outputs.extend(sf(outputs).cpu().detach().numpy().tolist())
			#text.append(sent)

	return fin_outputs#,text

n_emotion=len(dico_emo)

	

device = torch.device("cuda")

model = BERTGoEmotion(n_emotion)
if config.MULTI_GPUS==True:
	model = nn.DataParallel(model)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)
#model.eval()
#with torch.no_grad():
#if True:
listfile=os.listdir(args["directoryname"])
listfile=[p for p in listfile if ".csv" in p]
for o in listfile:
	df = pd.read_csv(args["directoryname"]+o)
	print(o,flush=True)
	print("df",df.shape,flush=True)

	pairs_list = []
	df_dataset = dataset.BERTDataset(
		sentences=df.sentences.values,
		target=df.emotion.values
	)

	df_data_loader = torch.utils.data.DataLoader(
		df_dataset,
		batch_size=config.TRAIN_BATCH_SIZE,
		num_workers=16 #change that accordingly to the number of cpu you have and how much they can handle and your batch size
	)


	outputs=predict_fn(df_data_loader, model, device)
	print(np.array(outputs).shape,flush=True)
	#_,preds=torch.max(torch.tensor(outputs),dim=1)
	for i,s in enumerate(outputs):
		#print(df.sentences[i],preds[i],df.emotion[i])
		pairs_list.append(s+[df.sentences.values[i]])
	


	df_predict=pd.DataFrame([v for v in pairs_list],index=np.arange(0,len(pairs_list),1),columns=[dico_emo[o] for o in range(n_emotion)]+["sentences"])
	df_predict.to_csv("/".join(config.DATA_FILE.split("/")[:-2])+"/"+"predictions/"+o[:-4]+"_eval.csv")

