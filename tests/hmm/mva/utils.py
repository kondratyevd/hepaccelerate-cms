import os,sys
import numpy as np
import pandas as pd
import glob, re
import math
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from keras.utils import to_categorical


def get_df(path):
    content = np.load(path)
    df = pd.DataFrame(data=content)
    df = filter(df)
    return df

def get_dataset(path, dataset):
    df_all = pd.DataFrame()
    dataset_regex = f"{dataset}_[0-9]+.npy"
    filenames = [f"{path}/{f}" for f in os.listdir(path) if re.search(dataset_regex,f)]
    pool = mp.Pool(mp.cpu_count()-1)
    dfs = pool.map(get_df, filenames)
    df_all = pd.concat(dfs)
    pool.close()
    pool.join()
    return df_all

def filter(df):
    df = df[(df['Higgs_mass']>110) & (df['Higgs_mass']<150) & (df['cat_index']==5) & (df['M_jj']>400)]
    return df

def train_filter(df):
    df = df[(df['Higgs_mass']>115) & (df['Higgs_mass']<135)]
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
        self.converted_to_cat = False
        self.year = '2016'
        os.system("mkdir -p "+self.out_dir)
        os.system("mkdir -p "+self.model_dir)

    def add_feature_set(self, name, option):
        self.feature_sets[name] = option

    def load_model(self, model):
        print("Adding model: {0}".format(model.name))
        self.mva_models.append(model)

    def load_categories(self):
        for k,v in self.category_labels.items():
            self.categories.append(k)
            self.df_dict[k] = pd.DataFrame()

    def load_as_category(self, path, name, category, wgt, for_train, for_test):
        if (category not in self.categories):
            cat_name = "{0}".format(self.category_labels[category]) if (category in self.category_labels.keys()) else ""
            print("Added new category: {0} ({1})".format(category, cat_name))
            self.categories.append(category)
            self.df_dict[category] = pd.DataFrame()
        new_df = get_dataset(path, name)
        new_df["label"] = name
        new_df["category"] = category
        new_df["wgt"] = wgt
        new_df["for_train"] = for_train
        new_df["for_test"] = for_test
        
        integral = np.multiply(new_df["wgt"], new_df["genweight"]).sum()

        new_df["resweight"] = 1/new_df["massErr_rel"] if "signal" in self.category_labels.values() else 1

        print(f"Appended {name} from {path}")
        print(f"Train: {for_train}, Test: {for_test}")
        print(f"Entries: {new_df.shape[0]},  Event yield: {integral}")
        print("-"*30)
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

        return training_data, testing_data

    def train_models(self):
        if not self.feature_sets:
            print("Error: no input feature sets found!")
            sys.exit(1)
        self.df = pd.concat(self.df_dict.values())

        only_train = self.df[self.df["for_train"] & ~self.df["for_test"]]
        x_otrain = only_train.loc[:,only_train.columns!='category']
        y_otrain =  only_train["category"]
        
        only_test = self.df[~self.df["for_train"] & self.df["for_test"]]
        x_otest = only_test.loc[:,only_test.columns!='category']
        y_otest =  only_test["category"]
        
        both = self.df[self.df["for_train"] & self.df["for_test"]]
        x_both = both.loc[:, both.columns!='category']
        y_both = both["category"]
        
        train_frac = 0.6
        # if sample used for both training and testing, split it
        x_train, x_test, y_train, y_test = train_test_split(x_both, y_both, train_size=train_frac, test_size=(1-train_frac), shuffle=True)
        # if only for testing - assign a weight so roc curves don't mess up
        only_test["wgt"] = only_test["wgt"]*(1-train_frac)

        x_train["category"] = y_train
        x_otrain["category"] = y_otrain
        train = pd.concat([x_train, x_otrain])
        train = train.sample(frac=1) # shuffle

        x_test["category"] = y_test
        x_otest["category"] = y_otest
        test = pd.concat([x_test, x_otest])

        tr_filter = False
        if tr_filter:
            train = train_filter(train)

        self.x_train = train.loc[:, train.columns!='category']
        self.y_train = train['category']

        self.x_test = test.loc[:, test.columns!='category']
        self.y_test = test['category']
        
        for feature_set_name, feature_set in self.feature_sets.items():

            for model in self.mva_models:
                training_data, testing_data = self.prepare_data(feature_set_name, feature_set)
                
                if model.binary:
                    if len(self.categories) is not 2:
                        print("Can't perform binary classification with {0} categories!".format(len(self.categories)))
                        sys.exit(1)
                elif not self.converted_to_cat:
                    self.y_train = to_categorical(self.y_train, len(self.categories))
                    self.y_test = to_categorical(self.y_test, len(self.categories))
                    # need this to convert only once (for the case when several models are trained)
                    self.converted_to_cat = True
                    
                if "resweights" in model.name:
                    self.y_train = train[['category', 'resweight']]
                    self.y_test = test[['category', 'resweight']]
                else:
                    training_data = training_data.loc[:, training_data.columns!='category']
                    testing_data = testing_data.loc[:, testing_data.columns!='category']
                
                model.train(training_data, self.y_train, feature_set_name, self.model_dir, self.name)
                print(f"Test shape: {testing_data.shape}")
                prediction = model.predict(testing_data, self.y_test, feature_set_name)

                if model.binary:
                    if "resweights" in model.name:
                        roc = roc_curve(self.y_test.iloc[:,0], prediction, sample_weight=self.x_test['wgt']*self.x_test["genweight"])
                        testing_data["category"]=self.y_test.iloc[:,0]
                    else:
                        roc = roc_curve(self.y_test, prediction, sample_weight=self.x_test['wgt']*self.x_test["genweight"])
                        testing_data["category"]=self.y_test

                    self.print_yields(roc, prediction, 0.01)

                    self.plot_hist("dnn_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction)
                    np.save("{0}/{1}_{2}_{3}_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
                    self.roc_curves[model.name+"_"+feature_set_name] = roc

                else:
                    vbf_pred = prediction[0]
                    ggh_pred = prediction[1]
                    dy_pred = prediction[2]
                    ewk_pred = prediction[3]

#                    cuts = (ewk_pred<0.7)
                    cuts = None
                    if cuts:
                        self.x_test = self.x_test[cuts]
                        self.y_test = self.y_test[cuts]
                        vbf_pred = vbf_pred[cuts]
                        ggh_pred = ggh_pred[cuts]
                        dy_pred = dy_pred[cuts]
                        ewk_pred = ewk_pred[cuts]
                    pred = vbf_pred
#                    pred = np.sum([vbf_pred,ggh_pred], axis=0)
#                    pred = np.sum([vbf_pred, (-1)*ewk_pred], axis=0)
                    roc = roc_curve(np.logical_or(self.y_test[:,0], self.y_test[:,1]), pred, sample_weight=self.x_test['wgt']*self.x_test["genweight"])

                    self.print_yields(roc, pred, 0.01)
                    
#                    np.save("{0}/{1}_{2}_{3}_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
#                    np.save("{0}/{1}_{2}_{3}_vbf-ewk_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
                    np.save("{0}/{1}_{2}_{3}_ewk<07_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)

#                    testing_data["category"]=y_test                
#                    self.plot_hist("vbf_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[0])
#                    self.plot_hist("ggh_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[1])
#                    self.plot_hist("dy_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[2])
#                    self.plot_hist("ewk_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[3])
#                    self.plot_hist("ttbar_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[4])

    def print_yields(self, roc, prediction, bkg_eff_threshold):
        threshold =  roc[2][np.abs(roc[0]-bkg_eff_threshold).argmin()]
        yields = np.multiply(self.x_test['wgt'],self.x_test["genweight"])

        if self.year is '2016':
            vbfy = yields[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powheg")]
            vbf_pred = prediction[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powheg")]
        elif self.year is '2017':
            vbfy = yields[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powheg_herwig")]
            vbf_pred = prediction[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powheg_herwig")]
        elif self.year is '2018':
            vbfy = yields[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powhegPS")]
            vbf_pred = prediction[(self.x_test["label"]=="vbf") | (self.x_test["label"]=="vbf_powhegPS")]
        vbfy = vbfy[vbf_pred>threshold]
        vbf_yield = vbfy.sum()/0.4

        yields_by_process = {"vbf": vbf_yield}
            
        other_processes = ["ggh", "dy", "ewk", "tt"]

        for process in other_processes:
            y = yields[(self.x_test["label"].str.contains(process))]
            pred = prediction[self.x_test["label"].str.contains(process)]
            y = y[pred>threshold]
            y = y.sum()/0.4
            if y:
                yields_by_process[process] = y

        for key, value in yields_by_process.items():
            print(f"{key} yield at bkg. eff. {bkg_eff_threshold} = {value}")

        processes = {
            "sig": ["ggh", "vbf"],
            "bkg": ["dy", "ewk", "tt"]
            }
        sb_yields = {"sig": 0, "bkg": 0}
        for p, pp in processes.items():
            for proc in pp:
                if yields_by_process[proc]:
                    sb_yields[p] += yields_by_process[proc]
        
        sigma = sb_yields["sig"] / math.sqrt(sb_yields["bkg"])
        print(f"S/sqrt(B) at bkg. eff. {bkg_eff_threshold}  = {sigma}")


    def plot_hist(self, var_name, xlim=None, df=None, values=None):
        if df is None:
            self.df = pd.concat(self.df_dict.values())
            self.df = filter(self.df)
            df = self.df
        print("Plotting "+var_name)
        plt.clf()
        ax = plt.gca()

        if values is not None:
            df["values"] = values

        for cat_num, cat_name in self.category_labels.items():
            df_cat = df[df["category"]==cat_num]
            if "_score" in var_name:
                ax.hist(df_cat["values"], bins=40, histtype='step', label=cat_name,normed=True, range=xlim)
            else:
                ax.hist(df_cat[var_name].values, bins=40, histtype='step', label=cat_name,normed=True, range=xlim)
        plt.xlabel(var_name)
        plt.legend(loc='best')
        plt.savefig("tests/hmm/mva/plots_updated/{0}.png".format(var_name))
