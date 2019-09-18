import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_rocs(rocs, out_path):
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    for name, path in rocs.items():
        roc = np.load(path)
        plt.plot(roc[0], roc[1], label=name) # [0]: fpr, [1]: tpr, [2]: threshold
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Test ROC curves')
    plt.legend(loc='best')
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

run_options = ["run1", "run2", "run3", "run4", "run5", "run6"]
var_options = ["V0", "V1", "V2", "V3"]
mva_options = ["caltech_model", "tf_bdt", "tf_bdt_resweight"]

rocs = {
    
    "ggH+VBF vs. DY+EWK": path+"run1_caltech_model_V0_roc.npy",
    "VBF vs. DY+EWK+ttbar": path+"run2_caltech_model_V0_roc.npy",

}

plot_rocs(rocs, path+"roc_test.png")
