from utils import MVASetup
from mva_models import KerasModel, SklearnBdtModel, TfBdtModel
from architectures import architectures

### Configuration ###

ds_path = "/depot/cms/hmm/out_dkondra/dnn_vars/2016"
sig_list = [
            "ggh_*", 
            "vbf_*", 
            ]
bkg_list = [
            "dy_m105_160_*",
            "ewk_lljj_mll105_160_*",
            ]
caltech_vars = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
                'leadingJet_eta', 'subleadingJet_eta', 'dRmin_mj', 'dRmax_mj', 'dRmin_mmj', 'dRmax_mmj', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS', 
                'softJet5', 'Higgs_pt', 'Higgs_eta', 'Higgs_mass']
#caltech_vars_no_mass = ['dEtamm', 'dPhimm', 'dRmm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'leadingJet_pt', 'subleadingJet_pt',
#                'leadingJet_eta', 'subleadingJet_eta', 'dRmin_mj', 'dRmax_mj', 'dRmin_mmj', 'dRmax_mmj', 'Zep',  'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS',
#                'softJet5', 'Higgs_pt', 'Higgs_eta']

initialized_models = [
        KerasModel(name='model_purdue_old', arch=architectures['model_purdue_old'], batch_size=2048, epochs=20, loss='binary_crossentropy', optimizer='adam',binary=True),
        KerasModel(name='caltech_model', arch=architectures['caltech_model'], batch_size=2048, epochs=20, loss='binary_crossentropy', optimizer='adam',binary=True),
#        SklearnBdtModel(name='simple_dt', max_depth=10, binary=True),
        TfBdtModel(name='tf_bdt', n_trees=800, max_depth=10, max_steps=500, batch_size=128)
    ]


### Load configuration and run training ###

mva_setup = MVASetup()
mva_setup.out_dir = "tests/hmm/mva/performance/"
mva_setup.model_dir = "tests/hmm/dnn/trained_models/"
mva_setup.category_labels = {0.: "signal", 1.: "background"}

for s in sig_list:
    mva_setup.load_as_category(ds_path,s,0.)
for b in bkg_list:
    mva_setup.load_as_category(ds_path,b,1.)

mva_setup.add_feature_set("V1",caltech_vars)
#mva_setup.add_feature_set("V2",caltech_vars_no_mass)

for m in initialized_models:
    mva_setup.load_model(m)

mva_setup.train_models()

mva_setup.plot_rocs("roc_test_2.png")
