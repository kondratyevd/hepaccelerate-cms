import os,sys
import numpy as np
import pandas as pd
import glob, re
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from keras.utils import to_categorical


def read_npy(path):
    content = np.load(path)
    df = pd.DataFrame(data=content)
    return df

def get_dataset_from_path(path,dataset):
    df_all = pd.DataFrame()
    filepath = path+"/"+dataset+".npy"
#    for f in glob.glob(filepath):
    for f in [f for f in os.listdir(path) if re.search("{0}.npy".format(dataset),f)]:
        df_ds = read_npy(path+"/"+f)
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
        self.converted_to_cat = False
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

    def load_as_category(self, path, ds, category, wgt, only_train, only_test, both):
        # categories should be enumerated from 0 to num.cat. - 1
        # 0, 1 for binary classification
        if (category not in self.categories):
            cat_name = "{0}".format(self.category_labels[category]) if (category in self.category_labels.keys()) else ""
            print("Added new category: {0} ({1})".format(category, cat_name))
            self.categories.append(category)
            self.df_dict[category] = pd.DataFrame()
        new_df = get_dataset_from_path(path, ds)
        new_df["category"] = category
        new_df["only_train"] = only_train
        new_df["only_test"] = only_test
        new_df["both"] = both
        new_df["wgt"] = wgt
        new_df = filter(new_df)
        integral = np.multiply(new_df["wgt"], new_df["genweight"]).sum()
        print(ds, integral)
        print(new_df.shape[0])
        self.df_dict[category] = pd.concat((self.df_dict[category], new_df))


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

        self.df['resweight'] = 1/self.df['massErr_rel']

        only_train = self.df[self.df["only_train"]]
        only_test = self.df[self.df["only_test"]]
        both = self.df[self.df["both"]]

        x_train, x_test, y_train, y_test = train_test_split(both.loc[:,both.columns!='category'], both["category"], train_size=0.6, test_size=0.4, shuffle=True)
        x_otrain = only_train.loc[:,only_train.columns!='category'] 
        y_otrain =  only_train["category"]
        x_otest = only_test.loc[:,only_test.columns!='category']
        y_otest =  only_test["category"]
        #self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df.loc[:,self.df.columns!='category'], self.df["category"], train_size=0.6, test_size=0.4, shuffle=True)

        # Remove some samples from training 
        train = x_train
        train["category"] = y_train
        otrain = x_otrain
        otrain["category"] = y_otrain
        train = pd.concat([train, otrain])
#        train = train[train["training"]]
        train = train.sample(frac=1)
        self.x_train = train.loc[:, train.columns!='category']
        self.y_train = train['category']

        test = x_test
        test["category"] = y_test
        otest = x_otest
        otest["category"] = y_otest
        test = pd.concat([test, otest])
        self.x_test = test.loc[:, test.columns!='category']
        self.y_test = test['category']


        for feature_set_name, feature_set in self.feature_sets.items():

            training_data, testing_data = self.prepare_data(feature_set_name, feature_set)

            for model in self.mva_models:

                if model.binary:
                    if len(self.categories) is not 2:
                        print("Can't perform binary classification with {0} categories!".format(len(self.categories)))
                        sys.exit(1)
                elif not self.converted_to_cat:
                    self.y_train = to_categorical(self.y_train, len(self.categories))
                    y_test = self.y_test
                    self.y_test = to_categorical(self.y_test, len(self.categories))
                    print(self.y_train)
                    self.converted_to_cat = True

                model.train(training_data, self.y_train, feature_set_name, self.model_dir, self.name,  self.x_train['resweight'])

                prediction = model.predict(testing_data, self.y_test, feature_set_name)
                
                simple=True

                if model.binary:
                    roc = roc_curve(self.y_test, prediction, sample_weight=self.x_test['wgt']*self.x_test["genweight"])
                    testing_data["category"]=self.y_test
                    self.plot_hist("dnn_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction)
                    np.save("{0}/{1}_{2}_{3}_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
                    self.roc_curves[model.name+"_"+feature_set_name] = roc

                elif simple:

                    vbf_pred = prediction[0]
                    ggh_pred = prediction[1]
                    dy_pred = prediction[2]
                    ewk_pred = prediction[3]

                    cuts = (ewk_pred<0.7)
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
#                    np.save("{0}/{1}_{2}_{3}_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
#                    np.save("{0}/{1}_{2}_{3}_vbf-ewk_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
                    np.save("{0}/{1}_{2}_{3}_ewk<07_roc".format(self.out_dir, self.name,  model.name, feature_set_name), roc)
#                    testing_data["category"]=y_test                
#                    self.plot_hist("vbf_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[0])
#                    self.plot_hist("ggh_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[1])
#                    self.plot_hist("dy_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[2])
#                    self.plot_hist("ewk_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[3])
#                    self.plot_hist("ttbar_score_{0}_{1}_{2}".format(self.name, model.name, feature_set_name), df=testing_data, values=prediction[4])

                else:
                    nsteps = 10
                    margin=0.001 #to make sure score=0 and score=1 are included
                    step = (1.+2*margin)/nsteps
                    stob_dict = {}
#                    s_ = self.x_test[np.logical_or(self.y_test[:,0], self.y_test[:,1])]
#                    s_total = np.multiply(s_["wgt"], s_["genweight"]).sum()
#                    b_ = self.x_test[np.logical_or.reduce((self.y_test[:,2], self.y_test[:,3], self.y_test[:,4]))]
#                    b_total = np.multiply(b_["wgt"], b_["genweight"]).sum()
                    self.x_test["category"]=y_test
                    new_df = pd.DataFrame(columns=self.x_test.columns)
                    bin_id=0
                    for i0 in range(nsteps): #vbf
                                print(i0,"/",nsteps)
#                        for i1 in range(nsteps): #ggh
#                            for i2 in range(nsteps): #dy
                                for i3 in range(nsteps): #ewk
                                    cut0 = np.logical_and(prediction[0]>-margin+i0*step, prediction[0]<-margin+(i0+1)*step)
 #                                   cut1 = np.logical_and(prediction[1]>-margin+i1*step, prediction[1]<-margin+(i1+1)*step)
#                                    cut2 = np.logical_and(prediction[2]>-margin+i2*step, prediction[2]<-margin+(i2+1)*step)
                                    cut3 = np.logical_and(prediction[3]>-margin+i3*step, prediction[3]<-margin+(i3+1)*step)
  #                                  cut = np.logical_and.reduce((cut0,cut1,cut2,cut3))
#                                    cut02 = np.logical_and(cut0,cut2)
                                    cut = np.logical_and(cut0, cut3)
                                    truth = self.y_test[cut]
                                    data = self.x_test[cut]
                                    s = data[np.logical_or(truth[:,0], truth[:,1])]
                                    s_int = np.multiply(s["wgt"], s["genweight"]).sum()
                                    b = data[np.logical_or.reduce((truth[:,2], truth[:,3], truth[:,4]))]
                                    b_int = np.multiply(b["wgt"], b["genweight"]).sum()
#                                    label = "{0}:{1}:{2}:{3}".format(i0,i1,i2,i3)
                                    label = "{0}:{1}".format(i0,i3)                                    
                                    #print("Bin {0}. S={1}, B={2}".format(label,s_int,b_int))
                                    if s_int and b_int>0 and s.shape[0]>100 and b.shape[0]>100:
#                                        stob_dict[bin_id] = [s_int, b_int, (s_int)/math.sqrt(b_int)]
                                        stob_dict[bin_id] = [s_int, b_int, (s_int)/(b_int)]
                                    else:
                                        stob_dict[bin_id] = [s_int, b_int, 0.]
                                    if data.shape[0]:
                                        data["dnn_bin"] = bin_id
                                        new_df = pd.concat([new_df, data], ignore_index=True)
                                    bin_id+=1
                                    
                    sorted_dict = sorted(stob_dict.items(), key=lambda kv: kv[1][2]) #[1] here means 'value', [2] means 3rd element (s/b)
#                    print(sorted_dict)
                    ibin = 0
                    new_df["crazy_score"] = 0
                    sig_below=0.
                    bkg_below=0.
                    sig_eff=[0.]
                    bkg_eff=[0.]
                    s_total = sum([v[0] for k,v in sorted_dict])
                    b_total = sum([v[1] for k,v in sorted_dict])
                    for key, value in sorted_dict:
                        new_df.loc[(new_df["dnn_bin"]==key),"crazy_score"] = ibin
                        sig_eff_i = sig_below/s_total if sig_below else 0
                        sig_eff.append(sig_eff_i)
                        sig_below+=value[0]
                        bkg_eff_i = bkg_below/b_total if bkg_below else 0
                        bkg_below+=value[1]
                        bkg_eff.append(bkg_eff_i)
#                        if value[0]:
#                            print(sig_eff_i, bkg_eff_i)
                        ibin+=1
                    sig_eff.append(1.)
                    bkg_eff.append(1.)
                    roc = (sig_eff, bkg_eff)
                    np.save("{0}/crazy_test_roc".format(self.out_dir, self.name), roc)
                    self.plot_hist("new_score", df=new_df, values=new_df["crazy_score"])
#                    np.save("{0}/crazy_test_sqrt_roc".format(self.out_dir, self.name), roc)
#                    self.plot_hist("new_score_sqrt", df=new_df, values=new_df["crazy_score"])
