import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

class MVAModel(object):
    def __init__(self, name, feature_set, binary):
        self.name = name
        self.feature_set = feature_set
        self.binary = binary

class KerasModel(MVAModel):
    def __init__(self, name, feature_set, input_dim, batchSize, epochs, loss, optimizer, binary = False):
        super().__init__(name, feature_set, binary)
        self.batchSize = batchSize
        self.epochs = epochs
        self.inputs = Input(shape=(len(self.feature_set),), name = self.name+'_input')
        self.loss = loss
        self.optimizer = optimizer

    def compile_model(self):
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model.summary()

    def train_model(self, x_train, y_train):
        self.compile_model()
        self.history = self.model.fit(
                                    x_train,
                                    y_train,
                                    epochs=self.epochs,
                                    batch_size=self.batchSize,
                                    verbose=1,
                                    validation_split=0.33,
                                    shuffle=True)
#        self.model.save(self.model_dir+model.name+'_trained.h5')
#        self.trained_models[model_name] = model

#        if plot_history:
#            plt.clf()
#            plt.plot(history.history['loss'])
#            plt.plot(history.history['val_loss'])
#            plt.title('Model loss')
#            plt.ylabel('Loss')
#            plt.xlabel('Epoch')
#            plt.legend(['Train', 'Test'], loc='upper left')
#           plt.savefig("{0}/history_{1}".format(self.out_dir, model_name))

class SklearnBdtModel(MVAModel):
    def __init__(self, name, feature_set, binary):
        super().__init__(name, feature_set, binary)

    def train_model(self, x_train, y_train):
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
        decision_tree = decision_tree.fit(x_train, y_train)
        r = export_text(decision_tree, feature_names=self.feature_set)
        print(r)

def get_model(name, feature_set_name, feature_set, output_dim):
    
    input_dim = len(feature_set)
    # For multi-class models set number of units in output layer to out_dim  

    model_purdue_old = KerasModel('model_purdue_old_'+feature_set_name, feature_set, input_dim, 2048, 10, 'binary_crossentropy', 'adam', True)
    x = Dense(50, name = model_purdue_old.name+'_layer_1', activation='relu')(model_purdue_old.inputs)
    x = Dropout(0.2)(x)
    x = Dense(25, name = model_purdue_old.name+'_layer_2', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(25, name = model_purdue_old.name+'_layer_3', activation='relu')(x)
    x = Dropout(0.2)(x)
    model_purdue_old.outputs = Dense(1, name = model_purdue_old.name+'_output',  activation='sigmoid')(x)

    caltech_model = KerasModel('caltech_model_'+feature_set_name, feature_set, input_dim, 2048, 10, 'binary_crossentropy', 'adam', True)
    x = Dense(100, name = caltech_model.name+'_layer_1', activation='tanh')(caltech_model.inputs)
    x = Dropout(0.2)(x)
    x = Dense(100, name = caltech_model.name+'_layer_2', activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(100, name = caltech_model.name+'_layer_3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    caltech_model.outputs = Dense(1, name = caltech_model.name+'_output',  activation='sigmoid')(x)

    simple_dt = SklearnBdtModel('simple_dt_'+feature_set_name, feature_set, True)

    models = {
        "model_purdue_old": model_purdue_old,
        "caltech_model": caltech_model,
        "simple_dt": simple_dt
    }
    
    return models[name]
