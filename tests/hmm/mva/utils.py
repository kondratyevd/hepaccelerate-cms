import os,sys
import numpy as np
import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from keras.utils import to_categorical


def read_npy(path):
    content = np.load(path)
    df = pd.DataFrame(data=content)
    return df

def get_dataset_from_path(path,dataset):
    df_all = pd.DataFrame()
    filepath = path+"/"+dataset+".npy"
    for f in glob.glob(filepath):
        df_ds = read_npy(f)
        df_ds = filter(df_ds)
        df_all = pd.concat((df_all, df_ds))
        print("Appending data from ", f)
    return df_all

def filter(df):
    df = df[(df['Higgs_mass']>110) & (df['Higgs_mass']<150) & (df['cat_index']==5) & (df['M_jj']>400)]
    return df

class MVASetup(object):
    def __init__(self, name):
        self.name = name
        self.category_labels = {}
        self.categories = []
        self.mva_models = []
        self.roc_curves = {}
        self.feature_sets = {}
        self.scalers = {}
        self.df = pd.DataFrame()
        self.df_dict = {}
        self.inputs = []
        self.out_dir = ""
        self.model_dir = ""
        os.system("mkdir -p "+self.out_dir)
        os.system("mkdir -p "+self.model_dir)

    def add_feature_set(self, name, option):
        self.feature_sets[name] = option

    def load_model(self, model):
        print("Adding model: {0}".format(model.name))
        self.mva_models.append(model)

    def load_as_category(self, path, ds, category, wgt, use_for_training):
        # categories should be enumerated from 0 to num.cat. - 1
        # 0, 1 for binary classification
        if category not in self.categories:
            cat_name = "({0})".format(self.category_labels[category]) if (category in self.category_labels.keys()) else ""
            print("Added new category: {0} {1}".format(category, cat_name))
            self.categories.append(category)
            self.df_dict[category] = pd.DataFrame()
        new_df = get_dataset_from_path(path, ds)
        new_df["category"] = category
        new_df["training"] = use_for_training
        new_df["wgt"] = wgt
        self.df_dict[category] = pd.concat((self.df_dict[category], new_df))

    def prepare_data(self, label, inputs):
        for i in inputs:
            if i not in self.df.columns:
                print("Feature set {0}: variable {1} not in columns!".format(label, i))
                sys.exit(1)

        x_mean = np.mean(self.x_train[inputs].values,axis=0)
        x_std = np.std(self.x_train[inputs].values,axis=0)
        training_data = (self.x_train[inputs]-x_mean)/x_std
        testing_data = (self.x_test[inputs]-x_mean)/x_std
        np.save("{0}/{1}_{2}_scalers".format(self.model_dir, self.name, label), [x_mean, x_std])

#        self.scalers[label] = StandardScaler().fit(self.x_train[inputs].values)
#        training_data = pd.DataFrame(columns=inputs, data=self.scalers[label].transform(self.x_train[inputs].values))
#        testing_data = pd.DataFrame(columns=inputs, data=self.scalers[label].transform(self.x_test[inputs].values))
        return training_data, testing_data

    def train_models(self):
        if not self.feature_sets:
            print("Error: no input feature sets found!")
            sys.exit(1)
        self.df = pd.concat(self.df_dict.values())

        self.df['resweight'] = 1/self.df['massErr_rel']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df.loc[:,self.df.columns!='category'], self.df["category"], train_size=0.6, test_size=0.4, shuffle=True)
        
        # Remove some samples from training 
        train = self.x_train
        train["category"] = self.y_train
        train = train[train["training"]]
        self.x_train = train.loc[:, train.columns!='category']
        self.y_train = train['category']


        for feature_set_name, feature_set in self.feature_sets.items():

            training_data, testing_data = self.prepare_data(feature_set_name, feature_set)

            for model in self.mva_models:

                if model.binary:
                    if len(self.categories) is not 2:
                        print("Can't perform binary classification with {0} categories!".format(len(self.categories)))
                        sys.exit(1)
                else:
                    self.y_train = to_categorical(self.y_train, len(self.categories))
                    self.y_test = to_categorical(self.y_test, len(self.categories))

                #model.train(training_data, self.y_train, feature_set_name)
                model.train(training_data, self.y_train, feature_set_name, self.model_dir, self.name,  self.x_train['resweight'])

                roc = roc_curve(self.y_test, model.predict(testing_data, self.y_test, feature_set_name), sample_weight=self.x_test['wgt']*self.x_test["genweight"])
                np.save("{0}/{1}_{2}_{3}_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
                self.roc_curves[model.name+"_"+feature_set_name] = roc
                
