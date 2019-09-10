import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
import tensorflow as tf

class KerasModel(object):
    def __init__(self, name, input_dim, batchSize, epochs, loss, optimizer):
        self.name = name
        self.batchSize = batchSize
        self.epochs = epochs
        self.inputs = Input(shape=(input_dim,), name = name+'_input') 
        self.loss = loss
        self.optimizer = optimizer
        

    def compile_model(self):
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model.summary()   

def get_model(name, input_dim, output_dim):
    model_50_D2_25_D2_25_D2 = KerasModel('model_50_D2_25_D2_25_D2', input_dim, 2048, 10, 'binary_crossentropy', 'adam')
    x = Dense(50, name = model_50_D2_25_D2_25_D2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2.inputs)
    x = Dropout(0.2)(x)
    x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_2', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_3', activation='relu')(x)
    x = Dropout(0.2)(x)
    model_50_D2_25_D2_25_D2.outputs = Dense(output_dim, name = model_50_D2_25_D2_25_D2.name+'_output',  activation='softmax')(x)

    models = {
        "model_50_D2_25_D2_25_D2": model_50_D2_25_D2_25_D2,
    }
    
    return models[name]
