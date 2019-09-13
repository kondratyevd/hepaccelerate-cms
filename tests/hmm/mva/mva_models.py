from keras.models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
import tensorflow as tf
import numpy as np
import pandas as pd

class MVAModel(object):
    def __init__(self, name, binary):
        self.name = name
        self.binary = binary
        self.feature_sets = {}

    def add_feature_set(self, feature_set_name, feature_set):
        self.feature_sets[feature_set_name] = feature_set

class KerasModel(MVAModel):
    def __init__(self, name, arch, batch_size, epochs, loss, optimizer, binary = False):
        super().__init__(name, binary)
        self.architecture = arch
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.model = {}
        self.history = {}

    def train_model(self, x_train, y_train, feature_set_name):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        feature_set = x_train.columns
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        label = self.name+"_"+feature_set_name
        inputs, outputs = self.architecture(label=label, input_dim=len(feature_set))
        self.model[feature_set_name] = Model(inputs=inputs, outputs=outputs)
        self.model[feature_set_name].compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model[feature_set_name].summary()
        self.history[feature_set_name] = self.model[feature_set_name].fit(
                                    x_train,
                                    y_train,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    validation_split=0.33,
                                    shuffle=True)

    def predict(self, x_test, y_test, feature_set_name):
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
    def __init__(self, name, max_depth, binary):
        super().__init__(name, binary)
        self.model = {}
        self.max_depth = max_depth

    def train_model(self, x_train, y_train, feature_set_name):
        feature_set = x_train.columns
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        model = DecisionTreeClassifier(random_state=0, max_depth=self.max_depth)
        self.model[feature_set_name] = model.fit(x_train, y_train)

    def predict(self, x_test, y_test, feature_set_name):
        return self.model[feature_set_name].predict_proba(x_test)[:,1].ravel()


class TfBdtModel(MVAModel):
    def __init__(self, name,  n_trees, max_depth, max_steps, batch_size):
        super().__init__(name, binary=True)
        self.model = {}
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.batch_size = batch_size

    def make_input_fn(self, X, y, training=False):
        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
            if training:
                batch_size = self.batch_size
                dataset = dataset.shuffle(batch_size)
                dataset = dataset.repeat(None)
            else:
                batch_size = len(y)
                dataset = dataset.repeat(1)
            dataset = dataset.batch(batch_size)
            return dataset
        return input_fn


    def train_model(self, x_train, y_train, feature_set_name):
        feature_set = x_train.columns
        self.nsamples = len(y_train)
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        feature_columns = []
        for feature_name in feature_set:
            feature_columns.append(tf.feature_column.numeric_column(feature_name))
        self.model[feature_set_name] = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, 
                                                                           n_batches_per_layer=1, 
                                                                           n_trees=self.n_trees, 
                                                                           max_depth=self.max_depth,
                                                                           learning_rate=0.01,
                                                                           center_bias = True,
#                                                                           tree_complexity = 0.001,
#                                                                           pruning_mode='post'
                                                                           )

        self.model[feature_set_name].train(self.make_input_fn(x_train, y_train, training=True), max_steps=self.max_steps)


    def predict(self, x_test, y_test, feature_set_name):

        pred_dicts = list(self.model[feature_set_name].predict(self.make_input_fn(x_test, y_test, training=False), yield_single_examples=False ))
        probs = pd.DataFrame(pred_dicts[0]['probabilities'])

        return probs[1].ravel()
