import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

class MVAModel(object):
    def __init__(self, name, binary):
        self.name = name
        self.binary = binary
        self.feature_sets = {}

    def add_feature_set(self, feature_set_name, feature_set):
        self.feature_sets[feature_set_name] = feature_set

class KerasModel(MVAModel):
    def __init__(self, name, batchSize, epochs, loss, optimizer, binary = False):
        super().__init__(name, binary)
        self.batchSize = batchSize
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        inputs, outputs = get_architecture(self.name, feature_set_name, feature_set)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model.summary()
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
    def __init__(self, name, binary):
        super().__init__(name, binary)

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
        decision_tree = decision_tree.fit(x_train, y_train)
        r = export_text(decision_tree, feature_names=feature_set)
        print(r)

def get_model(name):
    if name in initialized_models.keys():
        return initialized_models[name]
    else:
        print("Model {0} has not been initialized!".format(name))
        sys.exit(1)

initialized_models = {
        "model_purdue_old":  KerasModel('model_purdue_old', 2048, 10, 'binary_crossentropy', 'adam', True),
        "caltech_model" : KerasModel('caltech_model', 2048, 10, 'binary_crossentropy', 'adam', True),
        "simple_dt" : SklearnBdtModel('simple_dt', True),
    }

def get_architecture(name, feature_set_name, feature_set):
    input_dim = len(feature_set)

    def model_purdue_old_architecture(label, input_dim):
        inputs = Input(shape=(len(feature_set),), name = label+'_input') 
        x = Dense(50, name = label+'_layer_1', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(25, name = label+'_layer_2', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(25, name = label+'_layer_3', activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
        return inputs, outputs

    def model_caltech_architecture(label, input_dim):
        inputs = Input(shape=(len(feature_set),), name = label+'_input')
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

    if name not in architectures.keys():
        print("Architecture not defined for {0}!".format(name))
        sys.exit(1)

    return architectures[name](name+"_"+feature_set_name, input_dim)
