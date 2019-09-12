from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization


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
    x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
    return inputs, outputs

architectures = {
        "model_purdue_old": model_purdue_old_architecture,
        "caltech_model": model_caltech_architecture,
    }
