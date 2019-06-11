# Voice identification with one shot learning

an implementation based on several ideas introduced on Googles's Generalized End-to-End Loss for Speaker Verification (https://google.github.io/speaker-id/publications/GE2E/)

the objective of the project is being able to make a voice embedding for a given phrase (independent of what is said), and capture some voice characteristics in the embeddings that will let us recognize the voice author 

## Dataset and preprocessing

The dataset used to train the model was the open source LibriSpeech dataset (http://www.openslr.org/12), the dataset contains several short phrases from audibooks, where it's author is given, making for a great dataset for the given problem

### Preprocessing

each of the short sentences were divided in multiple smaller sentences by sliding window, making multiple 1.6 seconds sentence asociated with it's author.
This sentences will get grouped in batched of 5 from the same author, and wil be converted to it's spectogram
![](https://i.ibb.co/x1yTQr2/spectro.png)

## Model Architecture

### Base model

A base embedding model is defined which given a spectogram returns an 64 values vector which will be the embedding for that spectogram

![](https://i.ibb.co/hmHxbQc/architecture.jpg)

### Average model
this model will recieve the 5 spectogram batch and by using the base model plus an average layer, will return an averaged embedding for the given group


### Speech model
this model will recieve a triplet of spectograms groups, an anchor example, a positive example. And by using the previous models will get each embedding and output them.

Also a triplet loss function will be asigned to this model, which will calculate the gradient based on the resultant embeddings

## Authors

* **Sebastián Delgado**
* **Juan Fernando Rincón** 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* Coursera deep learning especialization (https://www.coursera.org/specializations/deep-learning)
* Original paper (https://google.github.io/speaker-id/publications/GE2E/)
