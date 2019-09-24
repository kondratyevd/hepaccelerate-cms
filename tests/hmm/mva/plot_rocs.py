import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#lbl = "V3"

#lbl = ("longrun1", "ggH+VBF vs. DY+EWK")
#lbl = ("longrun2", "ggH+VBF vs. DY+EWK+ttbar")
#lbl = ("longrun3", "VBF vs. DY+EWK")
lbl = ("longrun4", "VBF vs. DY+EWK+ttbar")

def plot_rocs(rocs, out_path):
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    for name, path in rocs.items():
        roc = np.load(path)
        plt.plot(roc[0], roc[1], label=name) # [0]: fpr, [1]: tpr, [2]: threshold
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
 #   plt.title('Variable set '+lbl)
    plt.title(lbl[1])
    plt.legend(loc='best', fontsize=12)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)

    plt.savefig(out_path)
    plt.xlim(0.005, 1)
    plt.xscale("log")
    plt.ylim(0.5, 1)
    plt.yscale("log")
    plt.savefig(out_path.replace(".png", "_log.png"))
    print("ROC curves are saved to {0}".format(out_path))

path = "tests/hmm/mva/performance/"

#run_options = ["run1", "run2", "run3", "run4", "run5", "run6"]
var_options = ["V0", "V1", "V2", "V3"]
run_options = [lbl[0]]
#run_options = ["longrun1", "longrun2","longrun3","longrun4"]
#var_options = [lbl]
mva_options = ["caltech_model"]#, "tf_bdt", "tf_bdt_resweight"]

rocs = {}

for run in run_options:
    for var in var_options:
        for mva in mva_options:
            label = "{0}_{1}_{2}".format(run,mva,var)
            roc_file = label+"_roc.npy"
            roc_path = path+roc_file
            if os.path.exists(roc_path):
                rocs[label] = roc_path
                print("Found ROC for {0}, adding".format(label))
    
#    plot_rocs(rocs, path+"roc_{0}.png".format(run))
#    rocs = {}

#plot_rocs(rocs, path+"roc_long_V3.png")

best_rocs = {
#    "ggH+VBF vs. DY+EWK" : path+"longrun1_caltech_model_{0}_roc.npy".format(lbl),
#    "ggH+VBF vs. DY+EWK+ttbar" : path+"longrun2_caltech_model_{0}_roc.npy".format(lbl),
#    "VBF vs. DY+EWK" : path+"longrun3_caltech_model_{0}_roc.npy".format(lbl),
#    "VBF vs. DY+EWK+ttbar" : path+"longrun4_caltech_model_{0}_roc.npy".format(lbl),
 
    "V0" : path+"{0}_caltech_model_V0_roc.npy".format(lbl[0]),
    "V1" : path+"{0}_caltech_model_V1_roc.npy".format(lbl[0]),
    "V2" : path+"{0}_caltech_model_V2_roc.npy".format(lbl[0]),
    "V3" : path+"{0}_caltech_model_V3_roc.npy".format(lbl[0]),   

}

plot_rocs(best_rocs, path+"roc_{0}.png".format(lbl[0]))
