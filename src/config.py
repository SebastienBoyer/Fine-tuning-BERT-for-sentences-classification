import transformers

MAX_LEN = 128 # This is how big (in term of tokens which really roughly speaking translate to words : sometimes a word can be split in multiple tokens)
#the sentences or group of sentences you are going to feed the model can be at max with BERT base at least you can go up to 512. But it will cost you in memory.
#It is also possible that your data is not that demanding in terms of number of tokens and so 512 is an unecessary amount of padding
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE= 32
EPOCHS = 4 # Usually not much more than that for fine tunning.
BERT_PATH = #path to your pre trained BERT model (in this demo the directory bert-base-uncased)
MODEL_PATH = #path to were your pre trained model will be saved 
	#: it should be outside of your folders, at the same level as input src and predictions
DATA_FILE = #path to your train_val csv : it should be in your input directory
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True) 
# in this particular case the BERT I am providing is uncased, so the do_lower_case is necessary
# This is also where you should change if you have a special tokenizer. For example I used BERTweet for another project
# I had to 
# from transformers import BertweetTokenizer
# BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
FINAL_LAYER_SIZE = 768 # this is the size of your BERT base pooled_output layer, which is what usually fine tunning use.
						# if you intend to use another layer or BERT_small or large. You will have to find out what that layer size is
DICO_NAME = "dico_sentiment.json"
PARAM_OPTI = # A text file, with path, in which each line corresponds to the name of the layer that you would like to optimize (layers that you don't want frozen).
# If file empty then all layers, that also means BERT layers, will be optimized.
PARAM_DECAY = # A text file, with path, in which each line corresponds to the a key word referring to groups layer, or a single layer if you right the full name of the layer,
# that you don't want to regularize.
MULTI_GPUS = False # change to True if multiple gpus available.
