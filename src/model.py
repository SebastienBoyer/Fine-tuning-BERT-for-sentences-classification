import transformers
import torch.nn as nn
import config

# Here is what is worth for you play with : in the different layers you can add after BERT embedding
# As well as in the different output of BERT you wan to work with.


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
		#if we only wanted to add that basic layer we could actually have used one of the huggingface function like in this case
		#BertForSequenceClassification which already contain this linear layer at the end. So really what's interesting here
		#is that you are allowed to do more if you want.

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
		# layers . Check the doc for more info at huggingface.co. Again that would not have been possible if we were using one 
		#of the pre coded function from Huggingface.
		output = self.out(bo)
		return output
