import transformers
import torch.nn as nn
import config




class BERTGoEmotion(nn.Module):
	def __init__(self,n_emotions):
		super(BERTGoEmotion,self).__init__()
		self.n_emotions = n_emotions
		self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)#if you were working with the BERTweet model this could be 
		#from transformers import AutoModel
		#AutoModel.from_pretrained("vinai/bertweet-base")
		#Also here in that configuration only the last hidden state layers are accessible. If you want access
		#to all the 12 the hidden state layers for BERT base, you should add output_hidden_states=True
		self.bert_drop = nn.Dropout(0.3)#You can play with thee drop out rate
		self.out = nn.Linear(config.FINAL_LAYER_SIZE,self.n_emotions)#here we just use a linear regression on the outcome
		#of BERT embedding. You can complexify as much as you want here. Let's say this is the basic method.

	def forward(self, ids, mask, token_type_ids):
		outputs = self.bert(
			ids,
			attention_mask=mask,
			token_type_ids=token_type_ids
		)
		#print(o1,flush=True)
		bo = self.bert_drop(outputs[1])#by choosing 1 we settle for the pooled_output of the last layer of BERT. 0
		# would give you access to the hidden sates of the last layer, which can be usefull in many other occasion like Part of Speech recognition
		# if you modified output_hidden_states=True before then outputs[2] give you access to the hidden states of the other
		# layers . Check the doc for more info at huggingface.co
		output = self.out(bo)
		return output
