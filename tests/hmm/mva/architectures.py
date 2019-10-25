from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
from keras import backend as K

def model_purdue_old_architecture(label, input_dim):
    inputs = Input(shape=(input_dim,), name = label+'_input')
    x = Dense(50, name = label+'_layer_1', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(25, name = label+'_layer_2', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(25, name = label+'_layer_3', activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
    return inputs, outputs

def model_caltech_architecture(label, input_dim):
    inputs = Input(shape=(input_dim,), name = label+'_input')
    x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
    return inputs, outputs

def model_caltech_multi_architecture(label, input_dim):
    inputs = Input(shape=(input_dim,), name = label+'_input')
    x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(5, name = label+'_output',  activation='softmax')(x)
    return inputs, outputs

def model_caltech_resweights_architecture(label, input_dim):
    inputs = Input(shape=(input_dim,), name = label+'_input')
    x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    out = Dense(1, name = label+'_output',  activation='sigmoid')(x)

    lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(inputs)
    def slicer(x):
        return x[:,0:1]  
    lambdaLayer = Lambda(slicer)(lambdaLayer)
    outputs = Concatenate()([out, lambdaLayer]) # order is important

    return inputs, outputs

architectures = {
        "model_purdue_old": model_purdue_old_architecture,
        "caltech_model": model_caltech_architecture,
        "caltech_multi": model_caltech_multi_architecture,
        "caltech_resweights": model_caltech_resweights_architecture
    }


def weighted_binary_crossentropy(y_true_, y_pred_) :
        y_true = y_true_[:,0]
        y_pred = y_pred_[:,0]
        weight = y_true_[:,1]
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        logloss = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(logloss, axis=-1)

losses = {
    "resweights": weighted_binary_crossentropy,
}
