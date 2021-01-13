from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

def loss_fn(outputs, targets,w):
	return nn.CrossEntropyLoss(weight=w)(outputs,targets)

def train_fn(data_loader, model, optimizer, device, scheduler,w):
	model.train()
	fin_targets = []
	fin_outputs = []
	losses = []

	for bi, d in tqdm(enumerate(data_loader),total=len(data_loader)):
		ids = d["ids"]
		token_type_ids = d["token_type_ids"]
		mask = d["mask"]
		targets = d["targets"]

		ids = ids.to(device, dtype=torch.long)
		token_type_ids = token_type_ids.to(device, dtype=torch.long)
		mask = mask.to(device, dtype=torch.long)
		targets = targets.to(device, dtype=torch.long)
		w = w.to(device)

		
		outputs= model(
			ids=ids,
			mask=mask,
			token_type_ids=token_type_ids
		)

		loss = loss_fn(outputs, targets,w).to(device)

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)# this is up to you
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		fin_targets.extend(targets.cpu().detach().numpy().tolist())
		sm = nn.Softmax(dim=1)
		fin_outputs.extend(sm(outputs).cpu().detach().numpy().tolist())#we softmax the outcome of our linear regression
		losses.append(loss.item())

	return np.mean(losses),fin_outputs, fin_targets




def eval_fn(data_loader, model, device,w):

	model.eval()
	fin_targets = []
	fin_outputs = []
	losses = []

	with torch.no_grad():
		for bi, d in tqdm(enumerate(data_loader),total=len(data_loader)):
			ids = d["ids"]
			token_type_ids = d["token_type_ids"]
			mask = d["mask"]
			targets = d["targets"]

			ids = ids.to(device, dtype=torch.long)
			token_type_ids = token_type_ids.to(device, dtype=torch.long)
			mask = mask.to(device, dtype=torch.long)
			targets = targets.to(device, dtype=torch.long)
			w = w.to(device)
			
			outputs= model(
				ids=ids,
				mask=mask,
				token_type_ids=token_type_ids
			)
			loss = loss_fn(outputs, targets, w).to(device)


			fin_targets.extend(targets.cpu().detach().numpy().tolist())
			sm = nn.Softmax(dim=1)
			fin_outputs.extend(sm(outputs).cpu().detach().numpy().tolist())
			losses.append(loss.item())

	return np.mean(losses), fin_outputs, fin_targets



