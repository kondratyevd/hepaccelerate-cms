import os,sys
import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree.export import export_text
from keras.utils import to_categorical
from models import get_model

def read_npy(path):
    content = np.load(path)
    df = pd.DataFrame(data=content)
    return df

def get_dataset_from_path(path,dataset):
    df_all = pd.DataFrame()
    filepath = path+"/"+dataset+".npy"
    for f in glob.glob(filepath):
        df_ds = read_npy(f)
        df_all = pd.concat((df_all, df_ds))
        print("Appending data from ", f)
    return df_all

def filter(df):
    df = df[(df['Higgs_mass']>110) & (df['Higgs_mass']<150)]
    return df

class MVASetup(object):
    def __init__(self, out_dir):
        self.category_labels = {}
        self.categories = []
        self.mva_models = []
        self.trained_models = {}
        self.feature_sets = {}
        self.scalers = {}
        self.df = pd.DataFrame()
        self.df_dict = {}
        self.inputs = []
        self.out_dir = out_dir
        self.model_dir = "tests/hmm/dnn/trained_models/"
        os.system("mkdir -p "+self.out_dir)
        os.system("mkdir -p "+self.model_dir)

    def add_feature_set(self, name, option):
        self.feature_sets[name] = option

    def add_model(self, model_name):
        print("Adding model: {0}".format(model_name))
        self.mva_models.append(model_name)

    def load_as_category(self, path, ds, cat_index):
        # categories should be enumerated from 0 to num.cat. - 1
        # 0, 1 for binary classification
        if cat_index not in self.categories:
            cat_name = "({0})".format(self.category_labels[cat_index]) if (cat_index in self.category_labels.keys()) else ""
            print("Added new category: {0} {1}".format(cat_index, cat_name))
            self.categories.append(cat_index)
            self.df_dict[cat_index] = pd.DataFrame()
        new_df = get_dataset_from_path(path, ds)
        new_df["cat_index"] = cat_index
        self.df_dict[cat_index] = pd.concat((self.df_dict[cat_index], new_df))

    def prepare_data(self, label, inputs):
        for i in inputs:
            if i not in self.df.columns:
                print("Feature set {0}: variable {1} not in columns!".format(label, i))
                sys.exit(1)

        self.scalers[label] = StandardScaler().fit(self.x_train[inputs].values)
        training_data = self.scalers[label].transform(self.x_train[inputs].values)
        return training_data
#        self.x_test[inputs] = self.scalers[label].transform(self.x_test[inputs].values)

    def train_models(self):
        if not self.feature_sets:
            print("Error: no input feature sets found!")
            sys.exit(1)
        self.df = pd.concat(self.df_dict.values())
        self.df = filter(self.df) 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df.loc[:,self.df.columns!='b'], self.df["cat_index"], test_size=0.25, shuffle=True)

        for feature_set_name, feature_set in self.feature_sets.items():
            training_data = self.prepare_data(feature_set_name, feature_set)
#            input_dim = len(feature_set)
            for model_name in self.mva_models:
                print("Considering model {0}".format(model_name))
                output_dim = len(self.categories)
                model = get_model(model_name, feature_set_name, feature_set, output_dim)
                if model.binary:
                    if len(self.categories) is not 2:
                        print("Can't perform binary classification with {0} categories!".format(len(self.categories)))
                        sys.exit(1)
                else:
                    self.y_train = to_categorical(self.y_train, len(self.categories))
                    self.y_test = to_categorical(self.y_test, len(self.categories))
                model.train_model(training_data, self.y_train)

    def plot_rocs(self, out_name):
        roc_parameters = {} # [0]: fpr, [1]: tpr, [2]: threshold
        for model_name, model in self.trained_models.items():
            predictions = model.model.predict(self.x_test).ravel()
            roc_parameters[model_name] = roc_curve(self.y_test, predictions)
        
        plt.clf()
        plt.plot([0, 1], [0, 1], 'k--')
        for model_name, roc in roc_parameters.items():
            plt.plot(roc[0], roc[1], label=model_name)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Test ROC curves')
        plt.legend(loc='best')
        plt.savefig("{0}/{1}".format(self.out_dir, out_name))
        print("ROC curves are saved to {0}/{1}".format(self.out_dir, out_name))




mva_setup = MVASetup(out_dir = "tests/hmm/mva/performance/")
ds_path = "/depot/cms/hmm/out_dkondra/dnn_vars/2016"
sig_list = [
#            "ggh_*", 
#            "vbf_*", 
            "vbf_[0-9]",
            ]
bkg_list = [
            "dy_[0-9]",
#            "dy_[0-9][0-9]",
#            "ttjets_dl_*",
            ]
caltech_vars = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj',
      'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj',
      'leadingJet_pt', 'subleadingJet_pt',
      'leadingJet_eta', 'subleadingJet_eta', 'dRmin_mj', 'dRmax_mj',
      'dRmin_mmj', 'dRmax_mmj', 'Zep',  'leadingJet_qgl',
      'subleadingJet_qgl', 'cthetaCS', 'softJet5', 'Higgs_pt', 'Higgs_eta',
     'Higgs_mass']

mva_setup.add_feature_set("caltech_variables",caltech_vars)

mva_setup.category_labels = {0: "signal", 1: "background"}

for s in sig_list:
    mva_setup.load_as_category(ds_path,s,0)
for b in bkg_list:
    mva_setup.load_as_category(ds_path,b,1)
)

mva_setup.add_model("model_purdue_old")
mva_setup.add_model("caltech_model")
mva_setup.add_model("simple_dt")

mva_setup.train_models()

#mva_setup.plot_rocs("roc_test.png")
