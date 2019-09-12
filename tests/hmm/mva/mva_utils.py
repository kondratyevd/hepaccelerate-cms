from keras.models import Model
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
    def __init__(self, name, arch, batch_size, epochs, loss, optimizer, binary = False):
        super().__init__(name, binary)
        self.architecture = arch
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.model = {}
        self.history = {}

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        inputs, outputs = self.architecture(label=self.name+"_"+feature_set_name, input_dim=len(feature_set))
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
    def __init__(self, name, max_depth, binary):
        super().__init__(name, binary)
        self.model = {}
        self.max_depth = max_depth

    def train_model(self, x_train, y_train, feature_set_name, feature_set):
        print("Considering model {0} with feature set {1}".format(self.name, feature_set_name))
        if feature_set_name not in self.feature_sets.keys():
            self.add_feature_set(feature_set_name, feature_set)
        model = DecisionTreeClassifier(random_state=0, max_depth=self.max_depth)
        self.model[feature_set_name] = model.fit(x_train, y_train)
        r = export_text(self.model[feature_set_name], feature_names=feature_set)
        print(r)

    def predict(self, x_test, feature_set_name):
        return self.model[feature_set_name].predict_proba(x_test)[:,1].ravel()

