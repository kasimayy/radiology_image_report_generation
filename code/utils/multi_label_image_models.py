import keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Dropout, RepeatVector, Embedding, Lambda, Reshape, concatenate, add, Concatenate, RepeatVector, TimeDistributed, BatchNormalization
from utils.custom_losses import binary_recall_specificity_loss, combined_loss
import json
from keras.models import model_from_json
import os
import numpy as np

class SingleLabelImageCNN(object):
    def __init__(self, **kwargs):
        default_params = {
            "cnn_model" : None,                      # use a pre-defined keras cnn model ['resnet50']
            "cnn_model_weights" : None,              # load pre-trained weights filename
            "image_input_shape" : (224, 224, 3),     # input shape into cnn
            "classes" : 80,
            "set_layers_trainable" : True,          # for setting layers trainable in pre-defined keras cnn
            "dense_dim" : 1024,
            "epochs" : 50,
            "bce_weight" : 0.5,
            "recall_weight" : 0.5,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'categorical_crossentropy',
            "loss_name" : 'categorical_crossentropy',
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        self.image_input_shape = tuple(self.image_input_shape)
        
        if self.loss_name == 'custom_recall_spec':
            self.loss = 'custom_recall_spec'

        elif self.loss_name == 'combined_loss':
            self.loss = 'combined_loss'
            
    def build_model(self):
        if self.cnn_model == 'resnet50':
            cnnmodel = keras.applications.resnet50.ResNet50(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
                
        elif self.cnn_model == 'vgg16':
            cnnmodel = keras.applications.vgg16.VGG16(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
                
        else:
            print('Specify CNN model type')
                
        last = cnnmodel.layers[-1].output
        x = Dense(self.dense_dim, activation='relu')(last)
        x = Dropout(0.5)(x)
        x = Dense(self.classes, activation="softmax")(x)
        self.model = Model(cnnmodel.input, x)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def run_experiment(self, train_batches, num_train_steps, val_batches, num_val_steps):
        self.build_model()
        self.history = self.model.fit_generator(train_batches, 
                               steps_per_epoch=num_train_steps, 
                               epochs=self.epochs, 
                               use_multiprocessing=True,
                               workers=10,
                               #callbacks=[early_stopping, checkpointer], 
                               validation_data=val_batches, 
                               validation_steps=num_val_steps,
                               verbose=True)

    def get_params(self):
        self.params = {}
        keys = ['epochs', 'classes', 'image_input_shape', 'set_layers_trainable', 'cnn_model', 'cnn_model_weights',
                'loss_name', 'dense_dim']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
            
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_loss_{}_dense_dim.json'\
        .format(self.epochs, self.loss_name, self.dense_dim)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_loss_{}_dense_dim.json'\
        .format(self.epochs, self.loss_name, self.dense_dim)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_loss_{}_dense_dim.h5'\
        .format(self.epochs, self.loss_name, self.dense_dim)
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):

        history_fn = 'history_cnn_{}_epochs_{}_loss_{}_dense_dim.json'\
        .format(self.epochs, self.loss_name, self.dense_dim)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_loss_{}_dense_dim.h5'\
        .format(self.epochs, self.loss_name, self.dense_dim)
        self.model.load_weights(output_dir + weights_fn)

class MultiLabelImageCNN(object):
    def __init__(self, **kwargs):
        default_params = {
            "cnn_model" : None,                      # use a pre-defined keras cnn model ['resnet50']
            "cnn_model_weights" : None,              # load pre-trained weights filename
            "image_input_shape" : (224, 224, 3),     # input shape into cnn
            "classes" : 80,
            "set_layers_trainable" : True,          # for setting layers trainable in pre-defined keras cnn
            "dense_dim" : 1024,
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'binary_crossentropy',
            "loss_name" : 'binary_crossentropy',
            "bce_weight" : 0.5,
            "recall_weight" : 0.5,
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        self.image_input_shape = tuple(self.image_input_shape)
        if self.loss_name == 'custom_recall_spec':
            self.loss = 'custom_recall_spec'

        elif self.loss_name == 'combined_loss':
            self.loss = 'combined_loss'
            
    def build_model(self):
        if self.cnn_model == 'resnet50':
            cnnmodel = keras.applications.resnet50.ResNet50(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
                
        elif self.cnn_model == 'vgg16':
            cnnmodel = keras.applications.vgg16.VGG16(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
                
        else:
            print('Specify CNN model type')
                
        last = cnnmodel.layers[-1].output
        x = Dense(self.dense_dim, activation='relu')(last)
        x = Dropout(0.5)(x)
        x = Dense(self.classes, activation="sigmoid")(x)
        self.model = Model(cnnmodel.input, x)
        
        if self.loss == 'custom_recall_spec':
            self.loss_name = 'custom_recall_spec'
            custom_loss = binary_recall_specificity_loss(self.recall_weight)
            self.loss = custom_loss

        elif self.loss == 'combined_loss':
            self.loss_name = 'combined_loss'
            custom_loss = combined_loss(self.bce_weight, self.recall_weight)
            self.loss = custom_loss

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def run_experiment(self, train_batches, num_train_steps, val_batches, num_val_steps):
        self.build_model()
        self.history = self.model.fit_generator(train_batches, 
                               steps_per_epoch=num_train_steps, 
                               epochs=self.epochs, 
                               use_multiprocessing=True,
                               workers=10,
                               #callbacks=[early_stopping, checkpointer], 
                               validation_data=val_batches, 
                               validation_steps=num_val_steps,
                               verbose=True)

    def get_params(self):
        self.params = {}
        keys = ['epochs', 'classes', 'image_input_shape', 'set_layers_trainable', 'cnn_model', 'cnn_model_weights',
                'loss_name', 'bce_weight', 'recall_weight', 'dense_dim']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
            
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_loss_{}_dense_dim_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.loss_name, self.dense_dim, self.bce_weight, self.recall_weight)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_loss_{}_dense_dim_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.loss_name, self.dense_dim, self.bce_weight, self.recall_weight)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_loss_{}_dense_dim_{:.1f}_bce_{:.1f}_recall.h5'\
        .format(self.epochs, self.loss_name, self.dense_dim, self.bce_weight, self.recall_weight)
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):

        history_fn = 'history_cnn_{}_epochs_{}_loss_{}_dense_dim_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.loss_name, self.dense_dim, self.bce_weight, self.recall_weight)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_loss_{}_dense_dim_{:.1f}_bce_{:.1f}_recall.h5'\
        .format(self.epochs, self.loss_name, self.dense_dim, self.bce_weight, self.recall_weight)
        self.model.load_weights(output_dir + weights_fn)
        
class SequenceImageCNN(object):
    def __init__(self, **kwargs):
        default_params = {
            "cnn_model" : None,                      # use a pre-defined keras cnn model ['resnet50']
            "rnn_model" : 'rnn2',                    # rnn0, rnn1, rnn2
            "data_type" : 'imagenet',                # nih or imagenet, sets whether cnn class predictions are used or not
            "cnn_model_weights" : None,              # load pre-trained weights filename
            "image_input_shape" : (224, 224, 3),     # input shape into cnn
            "image_output_shape" : 4096,             # dense output shape of cnn model
            "image_pred_shape" : 15,                 # class pred shape of cnn model
            "sequence_length" : 10,
            "word_embedding_dim" : 512,              # word embedding dim for rnn1 and rnn2
            "image_embedding_dim" : 512,             # image embedding dim for rnn1 and rnn2
            "joint_embedding_dim" : 512,             # image-text joint embedding dim for rnn0
            "vocab_size" : 100,                      # vocab size
            "rnn_hidden_dim" : 512,                  # hidden dim
            "set_layers_trainable" : False,          # for setting layers trainable in pre-defined keras cnn
            "temp" : 1,                              # between 0 and 1, reduces confidence in RNN prediction
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'categorical_crossentropy',
            "loss_name" : 'categorical_crossentropy',
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        self.image_input_shape = tuple(self.image_input_shape)
        
    def build_image_model(self):
        if self.cnn_model == 'resnet50':
            cnnmodel = keras.applications.resnet50.ResNet50(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
            last = cnnmodel.layers[-1].output
            
        elif self.cnn_model == 'vgg16':
            cnnmodel = keras.applications.vgg16.VGG16(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
            last = cnnmodel.layers[-1].output
            
        elif self.cnn_model == 'finetuned_resnet50':
            cnnmodel = keras.applications.resnet50.ResNet50(
                  input_tensor=None, 
                  input_shape=self.image_input_shape, 
                  pooling=None)
            if self.cnn_model_weights:
                cnnmodel.load_weights(self.cnn_model_weights)

            cnnmodel.layers.pop()
            for layer in cnnmodel.layers:
                layer.trainable=self.set_layers_trainable
            _last = cnnmodel.layers[-1].output
            x = Dense(self.dense_dim, activation='relu')(_last)
            last = Dense(self.classes, activation="sigmoid")(x)
                
        else:
            print('Specify CNN model type')
                
        # last = cnnmodel.layers[-1].output
        #x = Dense(self.embedding_dim, activation="sigmoid")(last)
        
        self.image_model = Model(cnnmodel.input, last)
        #self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    def build_rnn_model0(self):
        # image pred input
        if self.data_type == 'nih':
            input_pred = Input(shape=(self.image_pred_shape,))
        
        # image feat input
        input_image = Input(shape=(self.image_output_shape,))
        
        if self.data_type == 'nih':
            input_image = concatenate([input_pred, input_image])
            
        image_dense = Dense(units=self.joint_embedding_dim)(input_image)  # FC layer
        image_embedding = RepeatVector(1)(image_dense)

        input_text = Input(shape=(self.sequence_length,))
        word_embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=self.joint_embedding_dim)(input_text)

        # concat image+text embeddings
        sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])

        # lstm
        # sequence_input = BatchNormalization(axis=-1)(sequence_input)
        lstm_out = LSTM(units=self.rnn_hidden_dim,
                      return_sequences=False,
                      dropout=0.5,
                      recurrent_dropout=0.5)(sequence_input)
        
        outputs = Dense(self.vocab_size, activation='softmax')(lstm_out)
        # output = TimeDistributed(Dense(units=self.vocab_size))(lstm_out)
        
        # combine into model
        if self.data_type == 'nih':
            self.model = Model(inputs=[input_pred, input_image, input_text], outputs=outputs)
        else:
            self.model = Model(inputs=[input_image, input_text], outputs=outputs)
            
        # compile model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def build_rnn_model1(self):
        if self.data_type == 'nih':
            # image pred input
            input_pred = Input(shape=(self.image_pred_shape,))
        
        # image feat input
        input_image = Input(shape=(self.image_output_shape,))
        f1 = Dropout(0.5)(input_image)
        f2 = Dense(self.image_embedding_dim, activation='relu',name='image_features')(f1)
        
        # rnn sequence model
        input_text = Input(shape=(self.sequence_length,))
        t1 = Embedding(self.vocab_size, self.word_embedding_dim, mask_zero=True)(input_text)
        t2 = Dropout(0.5)(t1)
        t3 = LSTM(self.rnn_hidden_dim, return_sequences=False, name='text_features')(t2)
        
        #decoder model
        if self.data_type == 'nih':
            decoder1 = concatenate([input_pred, f2, t3])
        else:
            decoder1 = concatenate([f2, t3])
        decoder2 = Dense(self.joint_embedding_dim, activation='relu')(decoder1)
        decoder2 = Lambda(lambda x: x / self.temp)(decoder2)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        
        # combine into model
        if self.data_type == 'nih':
            self.model = Model(inputs=[input_pred, input_image, input_text], outputs=outputs)
        else:
            self.model = Model(inputs=[input_image, input_text], outputs=outputs)
            
        # compile model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    def build_rnn_model2(self):
        if self.data_type == 'nih':
            # image pred input
            input_pred = Input(shape=(self.image_pred_shape,))
            g1 = RepeatVector(self.sequence_length)(input_pred)
        
        # image input
        input_image = Input(shape=(self.image_output_shape,))
        f1 = Dropout(0.5)(input_image)
        f2 = Dense(self.image_embedding_dim, activation='relu',name='image_features')(f1)
        f3 = RepeatVector(self.sequence_length)(f2)
        
        # rnn sequence model
        input_text = Input(shape=(self.sequence_length,))
        t1 = Embedding(self.vocab_size, self.word_embedding_dim, mask_zero=True)(input_text)
        
        # concat pred+image+text input
        if self.data_type == 'nih':
            merged = concatenate([g1,f3,t1])
        else:
            merged = concatenate([f3,t1])
            
        # sequence model
        lstm = LSTM(self.rnn_hidden_dim, return_sequences=False, name='text_features')(merged)
        lstm = Lambda(lambda x: x / self.temp)(lstm)
        outputs = Dense(self.vocab_size, activation='softmax')(lstm)
        
        # combine into model
        if self.data_type == 'nih':
            self.model = Model(inputs=[input_pred, input_image, input_text], outputs=outputs)
        else:
            self.model = Model(inputs=[input_image, input_text], outputs=outputs)
        
        # compile model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    def run_experiment_imagenet(self, train_image_feat, train_ents, train_y, val_image_feat, val_ents, val_y):

        if self.rnn_model == 'rnn0':
            self.build_rnn_model0
        elif self.rnn_model == 'rnn1':
            self.build_rnn_model1()
        elif self.rnn_model == 'rnn2':
            self.build_rnn_model2()
            
        self.history = self.model.fit([train_image_feat, train_ents], train_y, 
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               #callbacks=[early_stopping, checkpointer], 
                               validation_data=([val_image_feat, val_ents],val_y),
                               verbose=True)
        
    def run_experiment_nih(self, train_image_pred, train_image_feat, train_ents, train_y, val_image_pred, val_image_feat, val_ents, val_y):

        if self.rnn_model == 'rnn0':
            self.build_rnn_model0
        elif self.rnn_model == 'rnn1':
            self.build_rnn_model1()
        elif self.rnn_model == 'rnn2':
            self.build_rnn_model2()
            
        self.history = self.model.fit([train_image_pred, train_image_feat, train_ents], train_y, 
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               #callbacks=[early_stopping, checkpointer], 
                               validation_data=([val_image_pred, val_image_feat, val_ents],val_y),
                               verbose=True)
        
    def get_params(self):
        self.params = {}
        keys = ['epochs', 'image_input_shape', 'image_output_shape', 'cnn_model', 'rnn_model', 'data_type', 'cnn_model_weights',
                'sequence_length', 'image_embedding_dim', 'word_embedding_dim', 'joint_embedding_dim', 'vocab_size', 
                'rnn_hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
            
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_cnn_{}_imagedim_{}_worddim_{}_jointdim_{}_hiddendim.json'\
        .format(self.epochs, self.cnn_model, self.image_embedding_dim, self.word_embedding_dim, self.joint_embedding_dim, self.rnn_hidden_dim)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_cnn_{}_imagedim_{}_worddim_{}_jointdim_{}_hiddendim.json'\
        .format(self.epochs, self.cnn_model, self.image_embedding_dim, self.word_embedding_dim, self.joint_embedding_dim, self.rnn_hidden_dim)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_cnn_{}_imagedim_{}_worddim_{}_jointdim_{}_hiddendim.h5'\
        .format(self.epochs, self.cnn_model, self.image_embedding_dim, self.word_embedding_dim, self.joint_embedding_dim, self.rnn_hidden_dim)
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):

        history_fn = 'history_cnn_{}_epochs_{}_cnn_{}_imagedim_{}_worddim_{}_jointdim_{}_hiddendim.json'\
        .format(self.epochs, self.cnn_model, self.image_embedding_dim, self.word_embedding_dim, self.joint_embedding_dim, self.rnn_hidden_dim)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_cnn_{}_imagedim_{}_worddim_{}_jointdim_{}_hiddendim.h5'\
        .format(self.epochs, self.cnn_model, self.image_embedding_dim, self.word_embedding_dim, self.joint_embedding_dim, self.rnn_hidden_dim)
        self.model.load_weights(output_dir + weights_fn)