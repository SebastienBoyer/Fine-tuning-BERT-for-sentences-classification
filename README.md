# Fine tunning BERT for sentences classification

Here you will find all the description needed to finetune the classical BERT model pretrained from [initial BERT paper](https://arxiv.org/pdf/1810), [models in huggingface](https://huggingface.co/models), [models from BERT](https://github.com/google-research/bert) on a subset of the emotion annotated data from [stanford go emotions](https://github.com/google-research/google-research/tree/master/goemotions).


## What's in here:
Check Directories_tree.txt. But briefly:
- BERT_env_cluster.yml enables the building of the environment I used on the sciCORE cluster. 
- BERT_env_RTX_3080.yml enables the building of the environment I used with more recent gpu.
- Directories_tree.txt explains the role of the different directories and files in running/training the model.
- BERT_base_uncased contains the pre-trained BERT model.
- The codes in the src directory.
- An example of dataset format for training, and an example of predict.py output.


## What you need:
Check config.py in src directory, or Directories_tree.txt for what you need. But briefly:
- a json file that will link the name of your classes to an int : all the int should start at 0 and be contiguous (check example).
- a train_validation set (training validation spliting is taking care of in train.py) of the csv format. Check the example : where our features are "sentences" and our targets are "emotion" (int).
- predict.py needs a path toward a directory in which all the files are going to be predicted (-d).


## What's under the hood:
It is a pretty common/straightforward sentences classifier.

Basically we use a pre-trained BERT model (of our choice) to embedd the words in a sentence or a block of sentences.

For that we use only the last layer hidden-state of the first token of the sequence (classification token) of BERT. In the commented code of model.py I discuss a little why and how we could get the full hidden states of the last layers or of all the layers. Indeed, this implementation is only really interesting because you can modify (it is pointed out in the commented code) the simple linear regression as well as the type of classification you intend. If you want to train a straightforward sequence classifier, like the one this example describes, you are better of using HUggingface implementations of different types of fine tunning (they have standard one for many purpose : sequence classification, question answering, etc...).

From this embedding we add some regularization thanks to a Dropout. It then proceeds to fit a linear regression to this BERT output thanks to a linear layer. The linear layer matrix coefficient projects the BERT output into a base made of your classes.

The training is made so that we find the good weights of the linear regression (good in the sense that the highest weight\*embedd_sentence predicts the class of the sentence). Finally we use a softmax layer and take the max softmax(weight\*embedd_sentence) as being our label. In predict.py the output is the softmax without taking the max because I think it gives more information.

This implementation offers you the choice of which the layers will be affected by weight decay regularization : another source of regularization than Dropout. Modify the no_decay.txt text file for that.

It also allows you to choose the depth of your fine tunning. Indeed you can choose to tune any different layers of your model, freezing the others. You could imagine tuning all the weights in your model : the weights of BERT pre trained and of your subsequent layers that you added. Or for example you can decide to freeze entierly BERT and only tune the layers you added.Modify the to_optimize.txt text file for that. 

## The code is full of information:
The code is commented and explains choices made as well as where you should definitly tweek to build your own sentences classifier. So it is highly recommended to read it carefully.

## Important
**The code is heavily (I insist on heavily) inspired** by [abhishekkrthakur](https://github.com/abhishekkrthakur/bert-sentiment/) and [venelinvalkov](https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb) (they both have great youtube channels with videos that go throught their own implementation of the code). So please if you use this code, acknowledge them too.
