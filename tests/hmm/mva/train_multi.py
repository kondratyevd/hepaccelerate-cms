from utils import MVASetup
from mva_models import KerasModel, SklearnBdtModel, TfBdtModel
from architectures import architectures
import json

### Configuration ###

year = '2016'
#in_path = "/depot/cms/hmm/out_dkondra/"
in_path = "/depot/cms/hmm/out_dkondra_nocalib/"
ds_path = in_path+"/dnn_vars/{0}".format(year)
norm_path = in_path+"/baseline/plots/{0}/normalization.json".format(year)

with open(norm_path) as norm_json:
    norm = json.load(norm_json)

class InputSample(object):
    def __init__(self, name, cat, wgt=1):
        self.name = name
        self.cat = cat
        self.wgt = wgt


input_list = [
    InputSample("ggh_amcPS", 1),
    InputSample("vbf_amcPS", 0),
    InputSample("dy_m105_160_amc", 2),
    InputSample("dy_m105_160_vbf_amc", 2),
    InputSample("ewk_lljj_mll105_160", 3),
    InputSample("ttjets_sl", 4),
    InputSample("ttjets_dl", 4)
]

if year is '2016':
    vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powheg"]
    input_list += [InputSample("vbf_powheg", True,  wgt=norm["genweights"]["vbf_powheg"]/vbf_genwgt_sum)]
    testing_samples = ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160","ttjets_sl", "ttjets_dl"]
elif year is '2017':
    vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powheg_herwig"]
    input_list += [InputSample("vbf_powheg_herwig", True,  wgt=norm["genweights"]["vbf_powheg_herwig"]/vbf_genwgt_sum)]
    testing_samples = ["ggh_amcPS", "vbf", "vbf_powheg_herwig", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160","ttjets_sl", "ttjets_dl"]
elif year is '2018':
    vbf_genwgt_sum = norm["genweights"]["vbf"]+norm["genweights"]["vbf_powhegPS"]
    input_list += [InputSample("vbf_powhegPS", True,  wgt=norm["genweights"]["vbf_powhegPS"]/vbf_genwgt_sum)]
    testing_samples = ["ggh_amcPS", "vbf", "vbf_powhegPS", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160","ttjets_sl", "ttjets_dl"]

input_list+=[InputSample("vbf", True,  wgt=norm["genweights"]["vbf"]/vbf_genwgt_sum)]

training_samples = {
    "multirun1": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"],
    "multirun2": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "multirun3": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"],
#    "longrun3": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160"], 
#    "longrun4": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
#    "newrun5": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ttjets_sl", "ttjets_dl"],
#    "newrun6": ["vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc"],
}

testing_samples = {
    "multirun1": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "multirun2": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
    "multirun3": ["ggh_amcPS", "vbf", "vbf_powheg", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160", "ttjets_sl", "ttjets_dl"],
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
                'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass', "massErr_rel", "leading_muon_pt", "leading_muon_eta",
            "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"]

v7 = ['M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
        'leadingJet_eta', 'subleadingJet_eta', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
                'softJet5', 'Higgs_mass', "massErr_rel", "leading_muon_pt", "leading_muon_eta",
            "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"]

initialized_models = [
#        KerasModel(name='caltech_multi', arch=architectures['caltech_multi'], batch_size=2048, epochs=200, loss='categorical_crossentropy', optimizer='adam',binary=False),
        KerasModel(name='caltech_multi', arch=architectures['caltech_multi'], batch_size=2048, epochs=2, loss='categorical_crossentropy', optimizer='adam',binary=False),
    ]


def run(run_label):
    ### Load configuration and run training ###

    mva_setup = MVASetup(run_label)
    mva_setup.year = year
    mva_setup.out_dir = "tests/hmm/mva/performance_multi/"
    mva_setup.model_dir = "tests/hmm/mva/trained_models/"
    mva_setup.category_labels = {0: "VBF", 1: "ggH", 2: "DY", 3: "EWK", 4: "ttbar"}
    mva_setup.load_categories()

    for i in input_list:
    #    cat = 1 if i.isSignal else 0
        wgt = i.wgt*norm["weight_xs"][i.name]
        for_train = (i.name in training_samples[run_label])
        for_test = (i.name in testing_samples[run_label])
        if for_train or for_test:
            mva_setup.load_as_category(ds_path, i.name, i.cat, wgt, for_train, for_test)

    mva_setup.add_feature_set("V0",caltech_vars)
#    mva_setup.add_feature_set("V6", v6)
#    mva_setup.add_feature_set("V5", v5)
#    mva_setup.add_feature_set("V7", v7)
    #mva_setup.add_feature_set("V1",nodRs+["dEtaMin_mj", "dEtaMax_mj", "dPhiMin_mj", "dPhiMax_mj", "dEtaMin_mmj", "dEtaMax_mmj", "dPhiMin_mmj", "dPhiMax_mmj"])
    #mva_setup.add_feature_set("V2",caltech_vars+["massErr_rel"])
    #mva_setup.add_feature_set("V3",caltech_vars+["leadingJet_btag", "subleadingJet_btag"])
    #mva_setup.add_feature_set("V4",caltech_vars+["leading_muon_pt", "leading_muon_eta", "leading_muon_phi", "subleading_muon_pt", "subleading_muon_eta", "subleading_muon_phi"])

    for m in initialized_models:
        mva_setup.load_model(m)

    mva_setup.train_models()
    del mva_setup

run("multirun3")
