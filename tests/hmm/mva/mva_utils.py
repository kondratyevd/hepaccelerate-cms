import keras
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import SGD
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

from architectures import architectures


def get_architecture(name, feature_set_name, feature_set):
    input_dim = len(feature_set)
    if name not in architectures.keys():
        print("Architecture not defined for {0}!".format(name))
        sys.exit(1)

    return architectures[name](name+"_"+feature_set_name, input_dim)

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
        self.model = {}
        self.history = {}

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        inputs, outputs = get_architecture(self.name, feature_set_name, feature_set)
        self.model[feature_set_name] = Model(inputs=inputs, outputs=outputs)
        self.model[feature_set_name].compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model[feature_set_name].summary()
        self.history[feature_set_name] = self.model[feature_set_name].fit(
                                    x_train,
                                    y_train,
                                    epochs=self.epochs,
                                    batch_size=self.batchSize,
                                    verbose=1,
                                    validation_split=0.33,
                                    shuffle=True)

    def predict(self, x_test, feature_set_name):
        return self.model[feature_set_name].predict(x_test).ravel()

#        self.model.save(self.model_dir+model.name+'_trained.h5')

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
        self.model = {}

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        model = DecisionTreeClassifier(random_state=0, max_depth=2)
        self.model[feature_set_name] = model.fit(x_train, y_train)
        r = export_text(self.model[feature_set_name], feature_names=feature_set)
        print(r)

    def predict(self, x_test, feature_set_name):
        return self.model[feature_set_name].predict(x_test).ravel()

