''' MeSH Sequence Learning from Chest X-ray Images: Experiments

The steps in the experiments are as follows:

    1. Extract image feature embeddings from pre-trained CNN model (resnet50/vgg16 trained on ImageNet or resnet50/vgg16 trained on NIH Chest X-ray8)
    2. If using NIH, also extract predictions
    3. Learn MeSH sequence conditioned on image embeddings through two models: rnn1 or rnn2
    4. Evaluate using BLEU1,2,3,4
    
 - ImageNet, rnn2, resnet50
'''

import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import sys
import math
sys.path.append("../..")
from utils import data_proc_tools as dpt
from utils import plot_tools as pt
from utils.multi_label_text_models import MultiLabelTextCNN
from utils.multi_label_image_models import MultiLabelImageCNN
from utils.multi_label_image_models import SequenceImageCNN
from utils.custom_metrics import recall, precision, binary_accuracy
from utils.custom_metrics import recall_np, precision_np, binary_accuracy_np, multilabel_confusion_matrix
from sklearn import metrics
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
import nltk
from nltk.translate.bleu_score import sentence_bleu
from collections import OrderedDict
# import tensorflow as tf
# tf.keras.backend.clear_session()
# tf.reset_default_graph()

import random
random.seed(42)
np.random.seed(42)
random_state=1000

dir = '/vol/medic02/users/ag6516/radiology_image_report_generation/'
data_dir = dir + 'data/chestx/'
sample_size = 'all'
data_type = 'processed'
data_output_dir = dir + 'data/chestx/{}/'.format(data_type)
dicts_dir = dir + 'data/chestx/{}/dicts/'.format(data_type)

image_dir = dir + 'data/chestx/jpeg_images/'
pretrained_model_dir = dir + 'trained_models/pretrained_cnn/'

sample_size = 'all'

print('Running experiment for sample size {}'.format(sample_size))

# load sampled sentence-entities df
print('Loading data...')
# train_df_predicted = pd.read_pickle(data_output_dir + 'train_{0}/train_{0}_balanced_uncollapsed.pkl'.format(1000))
train_df = pd.read_pickle(data_output_dir + 'train_1000/train_pred.pkl')
val_df = pd.read_pickle(data_output_dir + 'val/val_uncollapsed.pkl')
test_df = pd.read_pickle(data_output_dir + 'test/test_uncollapsed.pkl')

print('Training size: {}, Val size: {}, Test size: {}'.format(len(train_df), len(val_df), len(test_df)))
# vectorise train_df/val_df/test_df MeSH terms
print('Vectorising training and validation entities...')
max_mesh_length = 5
start_token = 'start'

train_vectoriser = dpt.Vectoriser(data_output_dir+'train_{}/'.format(sample_size), load_dicts=True, dicts_dir=dicts_dir)
val_vectoriser = dpt.Vectoriser(data_output_dir+'val/', load_dicts=True, dicts_dir=dicts_dir)
test_vectoriser = dpt.Vectoriser(data_output_dir+'test/', load_dicts=True, dicts_dir=dicts_dir)

train_mesh_captions = list(train_df.reduced_single_mesh)
val_mesh_captions = list(val_df.reduced_single_mesh)
test_mesh_captions = list(test_df.reduced_single_mesh)

# append start_token
train_padded_mesh = [[start_token] + mesh for mesh in train_mesh_captions]
val_padded_mesh = [[start_token] + mesh for mesh in val_mesh_captions]
test_padded_mesh = [[start_token] + mesh for mesh in test_mesh_captions]

# pad mesh captions with end tokens
train_padded_mesh = [dpt.pad_entities(mesh, max_mesh_length+1) for mesh in train_padded_mesh]
val_padded_mesh = [dpt.pad_entities(mesh, max_mesh_length+1) for mesh in val_padded_mesh]
test_padded_mesh = [dpt.pad_entities(mesh, max_mesh_length+1) for mesh in test_padded_mesh]

# vectorize mesh captions
train_mesh_ids_array = train_vectoriser.entities_to_vectors(train_padded_mesh)
val_mesh_ids_array = val_vectoriser.entities_to_vectors(val_padded_mesh)
test_mesh_ids_array = test_vectoriser.entities_to_vectors(test_padded_mesh)

mesh_to_id = train_vectoriser.ent_to_id
id_to_mesh = train_vectoriser.id_to_ent

train_mesh_vectors = np.array(train_mesh_ids_array)
val_mesh_vectors = np.array(val_mesh_ids_array)
test_mesh_vectors = np.array(test_mesh_ids_array)

# prepare df for image generator
train_df['filename'] = train_df.imageid.apply(lambda row: row+'.jpg')
val_df['filename'] = val_df.imageid.apply(lambda row: row+'.jpg')
test_df['filename'] = test_df.imageid.apply(lambda row: row+'.jpg')
train_df['mesh_vectors'] = train_mesh_vectors.tolist()
val_df['mesh_vectors'] = val_mesh_vectors.tolist()
test_df['mesh_vectors'] = test_mesh_vectors.tolist()
train_df['index'] = train_df.index
val_df['index'] = val_df.index
test_df['index'] = test_df.index

# load image cnn model and extract embeddings
print('Extracting image embeddings...')
cnn_model = 'resnet50'
epochs = 100
dense_dim = 2048
bce_weight = 0.0
recall_weight = 0.0

# load embeddings and predictions
# train_image_features = pickle.load(open(data_output_dir + 'train_{}/train_image_features_{}_imagenet_{}_{}_{}.pkl'\
#                                         .format(sample_size,cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))
val_image_features = pickle.load(open(data_output_dir + 'val/val_image_features_{}_imagenet_{}_{}_{}.pkl'\
                                      .format(cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))
test_image_features = pickle.load(open(data_output_dir + 'test/test_image_features_{}_imagenet_{}_{}_{}.pkl'\
                                      .format(cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))

train_image_features_dict = pickle.load(open(data_output_dir + 'train_{}/train_image_features_{}_imagenet_{}_{}_{}_dict.pkl'\
                                        .format('all',cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))
val_image_features_dict = pickle.load(open(data_output_dir + 'val/val_image_features_{}_imagenet_{}_{}_{}_dict.pkl'\
                                      .format(cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))
test_image_features_dict = pickle.load(open(data_output_dir + 'test/test_image_features_{}_imagenet_{}_{}_{}_dict.pkl'\
                                      .format(cnn_model,dense_dim,bce_weight,recall_weight), 'rb'))

train_image_features = []

for i, exam in train_df.iterrows():
    train_if = train_image_features_dict[exam.filename]
    train_image_features.append(train_if)
    
train_image_features = np.array(train_image_features)

# split sequence into multiple image,mesh term pars
train_X2, train_X3, train_y = list(), list(), list()
for image_feat, mesh_caption in list(zip(train_image_features, train_mesh_vectors)):
    for i in range(1,len(mesh_caption)):
        in_seq, out_seq = mesh_caption[:i], mesh_caption[i]
        in_seq = dpt.pad_entities(in_seq, max_mesh_length+1, end_token=mesh_to_id['end'])
        out_seq = dpt.one_hot_encode([[out_seq]], len(mesh_to_id))
        train_X2.append(image_feat)
        train_X3.append(in_seq)
        train_y.append(out_seq)

train_X2 = np.array(train_X2)
train_X3 = np.array(train_X3)
train_y = np.array(train_y)
train_y = train_y.reshape((train_y.shape[0],train_y.shape[-1]))

val_X2, val_X3, val_y = list(), list(), list()
for image_feat, mesh_caption in list(zip(val_image_features, val_mesh_vectors)):
    for i in range(1, len(mesh_caption)):
        in_seq, out_seq = mesh_caption[:i], mesh_caption[i]
        in_seq = dpt.pad_entities(in_seq, max_mesh_length+1, end_token=mesh_to_id['end'])
        out_seq = dpt.one_hot_encode([[out_seq]], len(mesh_to_id))
        val_X2.append(image_feat)
        val_X3.append(in_seq)
        val_y.append(out_seq)

val_X2 = np.array(val_X2)
val_X3 = np.array(val_X3)
val_y = np.array(val_y)
val_y = val_y.reshape((val_y.shape[0],val_y.shape[-1]))

test_X2, test_X3, test_y = list(), list(), list()
for image_feat, mesh_caption in list(zip(test_image_features, test_mesh_vectors)):
    for i in range(1, len(mesh_caption)):
        in_seq, out_seq = mesh_caption[:i], mesh_caption[i]
        in_seq = dpt.pad_entities(in_seq, max_mesh_length+1, end_token=mesh_to_id['end'])
        out_seq = dpt.one_hot_encode([[out_seq]], len(mesh_to_id))
        test_X2.append(image_feat)
        test_X3.append(in_seq)
        test_y.append(out_seq)

test_X2 = np.array(test_X2)
test_X3 = np.array(test_X3)
test_y = np.array(test_y)
test_y = test_y.reshape((test_y.shape[0],test_y.shape[-1]))

# train RNN
print('Training Sequence CNN-RNN...')
#             cnn_model = 'resnet50'
#             cnn_model_weights = pretrained_model_dir + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

#     cnn_model_weights = pretrained_model_dir + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#     image_input_shape = (224,224,3)

rnn_model = 'rnn1'
data_type = 'imagenet'
image_output_shape = train_image_features.shape[1]
sequence_length = train_mesh_vectors.shape[1]
image_embedding_dim = 1024
word_embedding_dim = 1024
rnn_hidden_dim = 512
vocab_size = len(mesh_to_id)
loss = 'categorical_crossentropy'
epochs = 40
batch_size = 128

new_experiment = SequenceImageCNN(rnn_model=rnn_model,
                               data_type=data_type,
                               cnn_model=cnn_model,
                               image_output_shape=image_output_shape,
                               sequence_length=sequence_length,
                               image_embedding_dim=image_embedding_dim,
                               word_embedding_dim=word_embedding_dim,
                               rnn_hidden_dim=rnn_hidden_dim,
                               vocab_size=vocab_size,
                               batch_size=batch_size,
                               loss = 'categorical_crossentropy',
                               epochs=epochs)
new_experiment.build_rnn_model1()
new_experiment.model.summary()

new_experiment.run_experiment_imagenet(train_X2, train_X3, train_y, val_X2, val_X3, val_y)

model_output_dir = dir + 'trained_models/chestx/image_rnn_{}/train_{}/{}/predicted_{}_{}_{}'\
.format(cnn_model,sample_size,rnn_model,dense_dim,bce_weight,recall_weight)
new_experiment.save_weights_history(model_output_dir)

######################################################### Evaluate ###############################################################
print('Evaluating...')
def generate_desc(seq_model, image_feat, max_length):
    # seed the generation process
    in_text = ['start']
    for i in range(max_mesh_length):
        # pad input
        sequence = dpt.pad_entities(in_text, max_mesh_length+1)
        # integer encode
        sequence = train_vectoriser.entities_to_vectors([sequence])
        # predict next word
        yhat = seq_model.predict([image_feat,sequence],verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = id_to_mesh[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        # stop if we predict the end of the sequence
        if word == 'end':
            break
        in_text.append(word)
    return in_text[1:]

train_imageids = train_df.imageid.tolist()
val_imageids = val_df.imageid.tolist()
test_imageids = test_df.imageid.tolist()

def evaluate_model(model, df, image_features_dict, max_length):
    actual, predicted = list(), list()
    bleu1, bleu2, bleu3, bleu4 = list(), list(), list(), list()

    for filename in df.filename.unique():
        image_feati = image_features_dict[filename]
        yhat = generate_desc(model,
                            image_feati.reshape(1,image_feati.shape[0]), 
                            max_mesh_length)
        
        reference = list(df[df.filename==filename].iloc[0].reduced_single_mesh)
        # calculate BLEU score
        bleu1.append(sentence_bleu([reference], yhat, weights=(1.0, 0, 0, 0)))
        bleu2.append(sentence_bleu([reference], yhat, weights=(0.5, 0.5, 0, 0)))
        bleu3.append(sentence_bleu([reference], yhat, weights=(0.3, 0.3, 0.3, 0)))
        bleu4.append(sentence_bleu([reference], yhat, weights=(0.25, 0.25, 0.25, 0.25)))
    
        # store actual and predicted
#         actual.append(reference)
#         predicted.append(yhat)
        
    print('BLEU1: ', np.mean(bleu1)*100)
    print('BLEU2: ', np.mean(bleu2)*100)
    print('BLEU3: ', np.mean(bleu3)*100)
    print('BLEU4: ', np.mean(bleu4)*100)

print('Train')
evaluate_model(new_experiment.model, train_df, train_image_features_dict, max_mesh_length)
print('Val')
evaluate_model(new_experiment.model, val_df, val_image_features_dict, max_mesh_length)
print('Test')
evaluate_model(new_experiment.model, test_df, test_image_features_dict, max_mesh_length)