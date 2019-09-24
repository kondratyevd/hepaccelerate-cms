from utils import MVASetup
from mva_models import KerasModel, SklearnBdtModel, TfBdtModel
from architectures import architectures
import json

### Configuration ###

ds_path = "/depot/cms/hmm/out_dkondra/dnn_vars/2016"
norm_path = "/depot/cms/hmm/out_nicolo_withJER_2/baseline/plots/2016/normalization.json"

with open(norm_path) as norm_json:
    norm = json.load(norm_json)

class InputSample(object):
    def __init__(self, label, path, category, wgt=1):
        self.label = label
        self.path = path
        self.category = category
        self.wgt = wgt

vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powheg1"]+norm["genweights"]["vbf_powheg2"]

input_list = [
    InputSample("ggh", "ggh_amcPS_*", 1),
    InputSample("vbf", "vbf_[0-9]", 0,  wgt=norm["genweights"]["vbf"]/vbf_genwgt_sum),
#    InputSample("vbf_powheg1", "vbf_powheg1_*", 0,  wgt=norm["genweights"]["vbf_powheg1"]/vbf_genwgt_sum),
#    InputSample("vbf_powheg2", "vbf_powheg2_*", 0,  wgt=norm["genweights"]["vbf_powheg2"]/vbf_genwgt_sum),
    InputSample("dy_m105_160_amc", "dy_m105_160_amc_*", 2),
#    InputSample("dy_m105_160_vbf_amc", "dy_m105_160_vbf_amc_*", 2),
    InputSample("ewk_lljj_mll105_160", "ewk_lljj_mll105_160_*", 3),
    InputSample("ttjets_sl", "ttjets_sl_*", 4),
    InputSample("ttjets_dl", "ttjets_dl_*", 4)
]



def run(vars_to_plot):

    mva_setup = MVASetup("")
    mva_setup.category_labels = {0: "VBF", 1: "ggH", 2: "DY", 3: "EWK", 4: "ttbar"}

    for i in input_list:
        wgt = i.wgt*norm["weight_xs"][i.label]
        mva_setup.load_as_category(ds_path, i.path, i.category, wgt, True)

    for v in vars_to_plot:
        if v is "massErr":
            mva_setup.plot_hist(v, (0,7))
        else:
            mva_setup.plot_hist(v)
    del mva_setup

vars_to_plot = [
    "dEtaMin_mj", 
    "dEtaMax_mj", 
    "dPhiMin_mj", 
    "dPhiMax_mj", 
    "dEtaMin_mmj", 
    "dEtaMax_mmj", "dPhiMin_mmj", "dPhiMax_mmj", "leadingJet_btag", "subleadingJet_btag", "massErr"]

run(vars_to_plot)
