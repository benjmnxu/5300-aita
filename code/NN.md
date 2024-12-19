## Neural Network Outline

The code expects the following files (which are all in our Github repo):

```data/train.csv```: Training data (alternatively, can use ```data/new_train.csv```)

```data/dev.csv```: Development (validation) data (alternatively, can use ```data/new_dev.csv```)

To run this script, simply run ```python strong_baseline_classifier_NN.py```. 

Since the embedding generation portion of this script takes a while to run (15-20 minutes in our experience, though sometimes much longer), we've also created a script that utilizes preloaded embeddings that are saved to tensors.

To run this script, simply run ```python strong_baseline_classifier_NN_w_saved_tensors.py```. 

Note: By default, this script uses the embeddings generated from the synthetic dataset (```data/synthetic_tensors.py```). If you want to instead use the embeddings generated from the original dataset (saved under ```data/tensors.py```), you will need to swap this in in line 24 of the script and additionally set ```df_train``` and ```df_dev``` to be ```data/train.csv``` and ```data/dev.csv```respectively).  
