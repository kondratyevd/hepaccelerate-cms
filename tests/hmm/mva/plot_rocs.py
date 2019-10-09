import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def plot_rocs(rocs, out_path, title=""):
    plt.clf()
#    plt.plot([0, 1], [0, 1], 'k--')
    for name, path in rocs.items():
        try: 
            roc = np.load(path)
            plt.plot(roc[1], roc[0], label=name) # [0]: fpr, [1]: tpr, [2]: threshold
        except:
            pass

    plt.title(title)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Bkg efficiency')
    plt.legend(loc='best', fontsize=12)
    plt.ylim(0.0001, 1)
    plt.yscale("log")
    plt.xlim(0, 1)
#    plt.ylim(0, 1)
    plt.savefig(out_path.replace(".png", "_log.png"))

    plt.xlim(0, 0.2)
    plt.ylim(0.0001, 0.01)
    plt.savefig(out_path)

#    plt.yscale("linear")
    plt.xlim(0.8, 0.95)
    plt.ylim(0.1, 0.3)
    plt.savefig(out_path.replace(".png", "_zoom.png"))

    print("ROC curves are saved to {0}".format(out_path))

path = "tests/hmm/mva/performance/"
plot_path = "tests/hmm/mva/plots_updated/"
path_multi = "tests/hmm/mva/performance_multi/"

var_options = ["V0", "V1", "V2", "V3", "V4"]
run_options = ["longrun1", "longrun2","longrun3","longrun4"]
#mva_options = ["caltech_model"]#, "tf_bdt", "tf_bdt_resweight"]


def plot_rocs_by_run(lbl):
    rocs_by_run = {
        "ggH+VBF vs. DY+EWK" : path+"longrun1_caltech_model_{0}_roc.npy".format(lbl),
        "ggH+VBF vs. DY+EWK+ttbar" : path+"longrun2_caltech_model_{0}_roc.npy".format(lbl),
        "VBF vs. DY+EWK" : path+"longrun3_caltech_model_{0}_roc.npy".format(lbl),
        "VBF vs. DY+EWK+ttbar" : path+"longrun4_caltech_model_{0}_roc.npy".format(lbl),
    }
    plot_rocs(rocs_by_run, plot_path+"roc_{0}.png".format(lbl), "Variable set "+lbl)

def plot_rocs_by_var(lbl, title):
    rocs_by_var = { 
        "V0 (baseline)" : path+"{0}_caltech_model_V0_roc.npy".format(lbl),
        "V1 (dR->dEta,dPhi)" : path+"{0}_caltech_model_V1_roc.npy".format(lbl),
        "V2 (V0+massErr)" : path+"{0}_caltech_model_V2_roc.npy".format(lbl),
        "V3 (V0+DeepB)" : path+"{0}_caltech_model_V3_roc.npy".format(lbl),   
        "V4 (V0+singleMu)" : path+"{0}_caltech_model_V4_roc.npy".format(lbl),
    }
    plot_rocs(rocs_by_var, plot_path+"roc_{0}.png".format(lbl), title)


#plot_rocs_by_var("longrun1", "Training samples: ggH+VBF vs. DY+EWK")
#plot_rocs_by_var("longrun2", "Training samples: ggH+VBF vs. DY+EWK+ttbar")
#plot_rocs_by_var("longrun3", "Training samples: VBF vs. DY+EWK")
#plot_rocs_by_var("longrun4", "Training samples: VBF vs. DY+EWK+ttbar")

#for var in var_options:
#    plot_rocs_by_run(var)



rocs = {
#    "multi": path+"multirun1_caltech_multi_V0_roc.npy",
#    "0.2": path+"multirun1_caltech_multi_V0_roc1.npy",
#    "0.4": path+"multirun1_caltech_multi_V0_roc2.npy",
#    "0.6": path+"multirun1_caltech_multi_V0_roc3.npy",
#    "0.8": path+"multirun1_caltech_multi_V0_roc4.npy",
#    "1": path+"multirun1_caltech_multi_V0_roc5.npy",

#    "V0 (baseline)" : path+"longrun3_caltech_model_V0_roc.npy",
#    "V5 (V0+singleMu+massErr+DeepB; dR->dEta,dPhi)" : path+"longrun3_caltech_model_V5_roc.npy",
#    "V6 (V0+singleMu+massErr; remove dR)" : path+"longrun3_caltech_model_V6_roc.npy",
#    "V7 (V6 w/o dimuon kinematics)" : path+"longrun3_caltech_model_V7_roc.npy",

#    "V0, VBF+ggH vs. DY+EWK" : path+"longrun1_caltech_model_V0_roc.npy",
#    "V6, VBF vs. DY+EWK" : path+"longrun3_caltech_model_V6_roc.npy",

#    "V0, binary" : path+"longrun3_caltech_model_V0_roc.npy",
    "V6, binary" : path+"longrun3_caltech_model_V6_roc.npy",
#    "V0, multi" : path_multi+"multirun3_caltech_multi_V0_roc.npy",
#    "V5, multi" : path_multi+"multirun3_caltech_multi_V5_roc.npy",
    "V6, multi" : path_multi+"multirun3_caltech_multi_V6_roc.npy",
#    "V7. multi" : path_multi+"multirun3_caltech_multi_V7_roc.npy",

#    "V0, multi VBF-EWK" : path_multi+"multirun3_caltech_multi_V0_vbf-ewk_roc.npy",
#    "V6, multi VBF-EWK" : path_multi+"multirun3_caltech_multi_V6_vbf-ewk_roc.npy",

#    "V0, multi VBF; EWK<0.9" : path_multi+"multirun3_caltech_multi_V0_ewk<09_roc.npy",
    "V6, multi; EWK<0.9" : path_multi+"multirun3_caltech_multi_V6_ewk<09_roc.npy",

#    "V0, multi VBF; EWK<0.8" : path_multi+"multirun3_caltech_multi_V0_ewk<08_roc.npy",
    "V6, multi; EWK<0.8" : path_multi+"multirun3_caltech_multi_V6_ewk<08_roc.npy",

#    "V0, multi VBF; EWK<0.7" : path_multi+"multirun3_caltech_multi_V0_ewk<07_roc.npy",
    "V6, multi; EWK<0.7" : path_multi+"multirun3_caltech_multi_V6_ewk<07_roc.npy",

#    "V0, multi VBF; EWK<0.6" : path_multi+"multirun3_caltech_multi_V0_ewk<06_roc.npy",
    "V6, multi; EWK<0.6" : path_multi+"multirun3_caltech_multi_V6_ewk<06_roc.npy",

#    "baseline+singleMu+massErr+DeepB" : path+"longrun2_reduced_caltech_model_V5_roc.npy",
#    "binaryV0" : path+"longrun2_caltech_model_V0_roc.npy",
#    "all variables" : path+"longrun2_caltech_model_V5_roc.npy",
#    "VBF score": path_multi+"1_0_0_0_roc.npy",
#    "ggH score": path_multi+"0_1_0_0_roc.npy",
#    "VBF+ggH score": path_multi+"1_1_0_0_roc.npy",
#    "VBF score, EWK<0.6": path_multi+"ewk<06_roc.npy",
#    "VBF+ggH score, V0": path_multi+"multirun2_caltech_multi_V0_roc.npy",
#    "VBF+ggH-DY score, V0": path_multi+"multirun2_caltech_multi_V0_cuts_roc.npy",
#    "VBF+ggH-DY-EWK score, V0": path_multi+"multirun2_caltech_multi_V0_vg-d-e_roc.npy",
#    "VBF+ggH-DY-EWK score, EWK<0.9, V0": path_multi+"multirun2_caltech_multi_V0_vg-d-e_9_roc.npy",
#    "VBF score, EWK<0.9, V0": path_multi+"multirun2_caltech_multi_V0_v_9_roc.npy", 
#    "new test: S/B sorting": path_multi+"crazy_test_roc.npy",
#    "new test: S/sqrt(B) sorting": path_multi+"crazy_test_sqrt_roc.npy",
#   "VBF+ggH score, V3": path_multi+"multirun2_caltech_multi_V3_roc.npy",
 #   "VBF+ggH score, V3, EWK<0.6": path_multi+"multirun2_caltech_multi_V3_cuts_roc.npy",
#    "test": path+"testrun_caltech_model_V0_roc.npy",
#    "test1": path+"testrun1_caltech_model_V0_roc.npy"
#    "baseline" : path+"longrun1_caltech_model_V0_roc.npy",
#    "+ singleMu" : path+"longrun1_caltech_model_V4_roc.npy"
}

#plot_rocs(rocs, path+"roc_{0}_gghandttbar.png".format(lbl))
#plot_rocs(rocs, plot_path+"roc_baseline.png")
#plot_rocs(rocs, plot_path+"roc_V567.png", "Training samples: VBF vs. DY+EWK")
plot_rocs(rocs, plot_path+"roc_multi.png","Training samples: VBF vs. DY+EWK")
#plot_rocs(rocs, plot_path+"roc_progress.png")
