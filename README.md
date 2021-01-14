# Fine tunning BERT for sentences classification

Here you will find all the description needed to finetune the classical BERT model pretrained from [initial BERT paper](https://arxiv.org/pdf/1810), [models in huggingface](https://huggingface.co/models), [models from BERT](https://github.com/google-research/bert) on a subset of the emotion annotated data from [stanford go emotions](https://github.com/google-research/google-research/tree/master/goemotions).


## What's in here:
- BERT_env_cluster.yml enables the building of the environment I used on the sciCORE cluster. 
- BERT_env_RTX_3080.yml enables the building of the environment I used with more recent gpu.
- Directories_tree.txt explains the role of the different directories and files in running/training the model.
- BERT_base_uncased contains the pre-trained BERT model.


## What you need:
- a json file that will link the name of your classes to an int : all the int should start at 0 and be contiguous (check example)
- a train_validation set (training validation spliting is taking care of in train.py) of the csv format. Check the example : where our features are "sentences" and our targets are "emotion" (int).
- train.py needs a name for your json dictionary (-d) and a True or False statement for multiple gpus usage (-p)
- predict.py needs, a path toward a directory in which all the files are going to be predicted (-f),a name for your json dictionary (-d) and a True or False statement for multiple gpus usage (-p)


## What's under the hood:
It is a pretty common/straightforward sentences classifier.

Basically we use a pre-trained BERT model (of our choice) to embedd the words in a sentence or a block of sentences.

For that we use only the last layer hidden-state of the first token of the sequence (classification token) of BERT. In the commented code of model.py I discuss a little why and how we could get the full hidden states of the last layers or of all the layers.

From that sentences embedding we add some regularization thanks to a Dropout. It then proceeds to fit a linear regression to this BERT output thanks to a linear layer. The linear layer matrix coefficient projects the BERT output into a base made of your classes.

The training is made so that we find the good weights of the linear regression (good in the sense that the highest weight\*embedd_sentence predicts the class of the sentence). Finally we use a softmax layer and take the max softmax(weight\*embedd_sentence) as being our label. In predict.py the output is the softmax without taking the max because I think it gives more information.

## 
The code is commented and explains choices made as well as where you should definitly tweek to build your own sentences classifier. So it is highly recommended to read it carefully.

## Important
**The code is heavily (I insist on heavily) inspired** by [abhishekkrthakur](https://github.com/abhishekkrthakur/bert-sentiment/) and [venelinvalkov](https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb) (they both have great youtube channels with videos that go throught their own implementation of the code). So please if you use this code, acknowledge them too.
