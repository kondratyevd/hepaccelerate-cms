from utils import MVASetup
from mva_models import KerasModel, SklearnBdtModel, TfBdtModel
from architectures import architectures
import json

### Configuration ###

ds_path = "/depot/cms/hmm/out_dkondra/dnn_vars/2016"
#norm_path = "/depot/cms/hmm/out_nicolo_withJER_2/baseline/plots/2016/normalization.json"
norm_path = "/depot/cms/hmm/out_dkondra/baseline/plots/2016/normalization.json"

with open(norm_path) as norm_json:
    norm = json.load(norm_json)

class InputSample(object):
    def __init__(self, label, path, isSignal, wgt=1):
        self.label = label
        self.path = path
        self.isSignal = isSignal
        self.wgt = wgt

#vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powheg1"]+norm["genweights"]["vbf_powheg2"]
vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powheg"]

input_list = [
    InputSample("ggh_amcPS", "ggh_amcPS_[0-9]", True),
    InputSample("vbf", "vbf_[0-9]+", True,  wgt=norm["genweights"]["vbf"]/vbf_genwgt_sum),
    InputSample("vbf_powheg", "vbf_powheg_[0-9]+", True,  wgt=norm["genweights"]["vbf_powheg"]/vbf_genwgt_sum),
    InputSample("vbf_amcPS", "vbf_amcPS_[0-9]+", True),
#    InputSample("vbf_powheg1", "vbf_powheg1_*", True,  wgt=norm["genweights"]["vbf_powheg1"]/vbf_genwgt_sum),
#    InputSample("vbf_powheg2", "vbf_powheg2_*", True,  wgt=norm["genweights"]["vbf_powheg2"]/vbf_genwgt_sum),
#    InputSample("dy", "dy_[0-9]+", False),
#    InputSample("dy_0j", "dy_0j_[0-9]+", False),
#    InputSample("dy_1j", "dy_1j_[0-9]+", False),
#    InputSample("dy_2j", "dy_2j_[0-9]+", False),
    InputSample("dy_m105_160_amc", "dy_m105_160_amc_[0-9]+", False),
    InputSample("dy_m105_160_vbf_amc", "dy_m105_160_vbf_amc_[0-9]+", False),
    InputSample("ewk_lljj_mll105_160", "ewk_lljj_mll105_160_[0-9]+", False),
    InputSample("ttjets_sl", "ttjets_sl_[0-9]+", False),
    InputSample("ttjets_dl", "ttjets_dl_[0-9]+", False)
]

training_samples = {
    "testrun": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"],
    "testrun1": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"],
    "longrun1": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"],
    "longrun2": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun2_reduced": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun3": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"], 
    "longrun4": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "newrun5": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ttjets_sl", "ttjets_dl"],
    "newrun6": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc"],
}

testing_samples = {
    "testrun": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "testrun1": [ "vbf_amcPS",  "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun2_reduced": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun1": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160","ttjets_sl", "ttjets_dl"],
    "longrun2": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun3": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "longrun4": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
}

caltech_vars = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
                'leadingJet_eta', 'subleadingJet_eta', 'dRmin_mj', 'dRmax_mj', 'dRmin_mmj', 'dRmax_mmj', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS', 
                'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass']
nodRs = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
                'leadingJet_eta', 'subleadingJet_eta', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
                        'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass']
v5 = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
                'leadingJet_eta', 'subleadingJet_eta', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
                'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass', "massErr_rel", "leadingJet_btag", "subleadingJet_btag", "leading_muon_pt", "leading_muon_eta", 
                "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi", "dEtaMin_mj", "dEtaMax_mj", "dPhiMin_mj", "dPhiMax_mj", 
                "dEtaMin_mmj", "dEtaMax_mmj", "dPhiMin_mmj", "dPhiMax_mmj"]

v6 = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
                'leadingJet_eta', 'subleadingJet_eta', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
                'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass', "massErr_rel", "leadingJet_btag", "subleadingJet_btag", "leading_muon_pt", "leading_muon_eta",
            "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"]

v7 = ['M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
        'leadingJet_eta', 'subleadingJet_eta', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
                'softJet5', 'Higgs_mass', "massErr_rel", "leadingJet_btag", "subleadingJet_btag", "leading_muon_pt", "leading_muon_eta",
            "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"]

initialized_models = [
#        KerasModel(name='model_purdue_old', arch=architectures['model_purdue_old'], batch_size=2048, epochs=20, loss='binary_crossentropy', optimizer='adam',binary=True),
        KerasModel(name='caltech_model', arch=architectures['caltech_model'], batch_size=2048, epochs=200, loss='binary_crossentropy', optimizer='adam',binary=True),
#        KerasModel(name='caltech_model', arch=architectures['caltech_model'], batch_size=2048, epochs=20, loss='binary_crossentropy', optimizer='adam',binary=True),
#        SklearnBdtModel(name='simple_dt', max_depth=10, binary=True),
#        TfBdtModel(name='tf_bdt', n_trees=800, max_depth=7, max_steps=500, batch_size=1024, tree_complexity=0.01, pruning='pre', lr=0.01, bpl=1),
#        TfBdtModel(name='tf_bdt_resweight', n_trees=800, max_depth=7, max_steps=500, batch_size=1024, tree_complexity=0.01, pruning='pre', lr=0.01, bpl=1, weighted=True),
    ]


def run(run_label):
    ### Load configuration and run training ###

    mva_setup = MVASetup(run_label)
    mva_setup.out_dir = "tests/hmm/mva/performance/"
    mva_setup.model_dir = "tests/hmm/mva/trained_models/"
    mva_setup.category_labels = {0.: "background", 1.: "signal"}

    for i in input_list:
        cat = 1 if i.isSignal else 0
        wgt = i.wgt*norm["weight_xs"][i.label]
        use_for_training = (i.label in training_samples[run_label])
        use_for_testing = (i.label in testing_samples[run_label])
        only_train = use_for_training and not use_for_testing
        only_test = use_for_testing and not use_for_training
        both = use_for_training and use_for_testing
#        if use_for_training and (not use_for_testing):
#            wgt = wgt*0.6
        if only_test:
            wgt = wgt*0.4
        if use_for_training or use_for_testing:
            print("Adding {0}.  train: {1},  test: {2}".format(i.label, use_for_training, use_for_testing))
            mva_setup.load_as_category(ds_path, i.path, cat, wgt, only_train, only_test, both)

#    mva_setup.add_feature_set("V0",caltech_vars)
#    mva_setup.add_feature_set("V1",nodRs+["dEtaMin_mj", "dEtaMax_mj", "dPhiMin_mj", "dPhiMax_mj", "dEtaMin_mmj", "dEtaMax_mmj", "dPhiMin_mmj", "dPhiMax_mmj"])
#    mva_setup.add_feature_set("V2",caltech_vars+["massErr_rel"])
#    mva_setup.add_feature_set("V3",caltech_vars+["leadingJet_btag", "subleadingJet_btag"])
#    mva_setup.add_feature_set("V4",caltech_vars+["leading_muon_pt", "leading_muon_eta", "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"])
    mva_setup.add_feature_set("V5", v5)
    mva_setup.add_feature_set("V6", v6)
    mva_setup.add_feature_set("V7", v7)
    for m in initialized_models:
        mva_setup.load_model(m)

    mva_setup.train_models()
    del mva_setup

#run("longrun1")
#run("longrun2")
#run("testrun1")
#run("longrun2_reduced")
run("longrun3")
#run("longrun4")
#run("newrun5")
#run("newrun6")
