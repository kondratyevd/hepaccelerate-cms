import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
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

class DNNSetup(object):
    def __init__(self):
        self.category_labels = {}
        self.categories = []
        self.df = pd.DataFrame()
        self.df_dict = {}
        self.inputs = []
        self.model_dir = "tests/hmm/dnn/trained_models/"
        os.system("mkdir -p "+self.model_dir)

    def load_as_category(self, path, ds, cat_index):
        if cat_index not in self.categories:
            cat_name = "({0})".format(self.category_labels[cat_index]) if (cat_index in self.category_labels.keys()) else ""
            print("Added new category: {0} {1}".format(cat_index, cat_name))
            self.categories.append(cat_index)
            self.df_dict[cat_index] = pd.DataFrame()
        new_df = get_dataset_from_path(path, ds)
        self.df_dict[cat_index] = pd.concat((self.df_dict[cat_index], new_df))


    def load_to_eval(self):
        pass

    def prepare_data(self, inputs):
        self.inputs = inputs
        self.df = pd.concat(self.df_dict.values())
        for i in self.inputs:
            if i not in self.df.columns:
                print("Variable {0} not in columns!".format(i))
                sys.exit(1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df.loc[:,self.inputs], self.df.iloc[:,-1], test_size=0.33, shuffle=True)
        
    def train_model(self, model_name, binary=True):
        if self.inputs:
            input_dim = len(self.inputs)
        else:
            print("Error: no inputs found!")
            sys.exit(1)

        if binary:
            if len(self.categories) is 2:
                output_dim = 1 
            else:
                print("Can't perform binary classification with {0} categories!".format(len(self.categories)))
                sys.exit(1)
        else:
            output_dim = len(self.categories)

        model = get_model(model_name, input_dim, output_dim)
        model.compile_model()

        history = model.model.fit(            
                                    self.x_train,
                                    self.y_train,
                                    epochs=model.epochs, 
                                    batch_size=model.batchSize, 
                                    verbose=1,
                                    validation_split=0.25,
                                    shuffle=True)
        
        model.model.save(self.model_dir+model.name+'_trained.h5')

dnn_setup = DNNSetup()
ds_path = "/depot/cms/hmm/out_nicolo_withJER/dnn_vars/2016"
sig_list = ["ggh_*", "vbf_0", "vbf_1"]
bkg_list = ["dy_[0-9]",
#            "dy_[0-9][0-9]",
            "ttjets_dl_*"]
input_vars = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj',
      'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj',
      'leadingJet_pt', 'subleadingJet_pt',
      'leadingJet_eta', 'subleadingJet_eta', 'dRmin_mj', 'dRmax_mj',
      'dRmin_mmj', 'dRmax_mmj', 'Zep',  'leadingJet_qgl',
      'subleadingJet_qgl', 'cthetaCS', 'softJet5', 'Higgs_pt', 'Higgs_eta',
      'Higgs_mass',]

dnn_setup.category_labels = {0: "signal", 1: "background"}

for s in sig_list:
    dnn_setup.load_as_category(ds_path,s,0)
for b in bkg_list:
    dnn_setup.load_as_category(ds_path,b,1)

dnn_setup.prepare_data(input_vars)
dnn_setup.train_model("model_50_D2_25_D2_25_D2")
