import numpy as np
import os
from hepaccelerate.utils import Histogram, Results

from collections import OrderedDict
import uproot


import copy
import multiprocessing

from pars import catnames, varnames, analysis_names, shape_systematics
from scipy.stats import wasserstein_distance

import argparse
import pickle
import glob

import cloudpickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis plotting')
    parser.add_argument('--input', action='store', type=str, help='Input directory from the previous step')
    parser.add_argument('--keep-processes', action='append', help='Keep only certain processes, defaults to all', default=None)
    parser.add_argument('--histnames', action='append', help='Process only these histograms, defaults to all', default=None)
    args = parser.parse_args()
    return args

def assign_plot_title_label(histname):
    spl = histname.split("__")
    varname_nice = "UNKNOWN"
    catname_nice = "UNKNOWN"
    if len(spl) == 3:
        catname = spl[1]
        varname = spl[2]
        catname_nice = catnames[catname]
        if varname in varnames:
            varname_nice = varnames[varname]
        else:
            varname_nice = varname
            print("WARNING: please define {0} in pars.py".format(varname))
            
    return varname_nice, catname_nice
             
def plot_hist_ratio(hists_mc, hist_data,
        total_err_stat=None,
        total_err_stat_syst=None,
        figure=None):
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not figure:
        figure = plt.figure(figsize=(5,5), dpi=100)

    ax1 = plt.axes([0.0, 0.23, 1.0, 0.8])
       
    hmc_tot = np.zeros_like(hist_data.contents)
    hmc_tot2 = np.zeros_like(hist_data.contents)
    for h in hists_mc:
        plot_hist_step(ax1, h.edges, hmc_tot + h.contents,
            np.sqrt(hmc_tot2 + h.contents_w2),
            kwargs_step={"label": getattr(h, "label", None)}
        )
        hmc_tot += h.contents
        hmc_tot2 += h.contents_w2
#    plot_hist_step(h["edges"], hmc_tot, np.sqrt(hmc_tot2), kwargs_step={"color": "gray", "label": None})
    ax1.errorbar(
        midpoints(hist_data.edges), hist_data.contents,
        np.sqrt(hist_data.contents_w2), marker=".", lw=0,
        elinewidth=1.0, color="black", ms=3, label=getattr(hist_data, "label", None))
    
    if not (total_err_stat_syst is None):
        histstep(ax1, hist_data.edges, hmc_tot + total_err_stat_syst, color="blue", linewidth=1, linestyle="--", label="stat+syst")
        histstep(ax1, hist_data.edges, hmc_tot - total_err_stat_syst, color="blue", linewidth=1, linestyle="--")
    
    if not (total_err_stat is None):
        histstep(ax1, hist_data.edges, hmc_tot + total_err_stat, color="gray", linewidth=1, linestyle="--", label="stat")
        histstep(ax1, hist_data.edges, hmc_tot - total_err_stat, color="gray", linewidth=1, linestyle="--")
        
    ax1.set_yscale("log")
    ax1.set_ylim(1e-2, 100*np.max(hist_data.contents))
    
    #ax1.get_yticklabels()[-1].remove()
    
    ax2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=ax1)

    ratio = hist_data.contents / hmc_tot
    ratio_err = np.sqrt(hist_data.contents_w2) /hmc_tot
    ratio[np.isnan(ratio)] = 0

    plt.errorbar(midpoints(hist_data.edges), ratio, ratio_err, marker=".", lw=0, elinewidth=1, ms=3, color="black")

    if not (total_err_stat_syst is None):
        ratio_up = (hmc_tot + total_err_stat_syst) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat_syst) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, hist_data.edges, ratio_up, color="blue", linewidth=1, linestyle="--")
        histstep(ax2, hist_data.edges, ratio_down, color="blue", linewidth=1, linestyle="--")

    if not (total_err_stat is None):
        ratio_up = (hmc_tot + total_err_stat) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, hist_data.edges, ratio_up, color="gray", linewidth=1, linestyle="--")
        histstep(ax2, hist_data.edges, ratio_down, color="gray", linewidth=1, linestyle="--")

                
    plt.ylim(0.5, 1.5)
    plt.axhline(1.0, color="black")
    
    return ax1, ax2

def plot_variations(args):
    res, hd, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir, datataking_year = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _outdir = outdir + "/shape_systematics/{0}/".format(var)
    try:
        os.makedirs(_outdir)
    except Exception as e:
        pass
    for mc_samp in mc_samples:
        for unc in shape_systematics:
            fig = plt.figure(figsize=(5,5), dpi=100)
            ax = plt.axes()
            hnom = res[mc_samp]["nominal"]* weight_xs[mc_samp]
            plot_hist_step(ax, hnom.edges, hnom.contents,
                np.sqrt(hnom.contents_w2),
                kwargs_step={"label": "nominal"},
            )
            for sdir in ["__up", "__down"]:
                if (unc + sdir) in res[mc_samp]:
                    hvar = res[mc_samp][unc + sdir]* weight_xs[mc_samp]
                    plot_hist_step(ax, hvar.edges, hvar.contents,
                        np.sqrt(hvar.contents_w2),
                        kwargs_step={"label": sdir.replace("__", "")},
                    )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=4, loc=1, ncol=2)
            plt.savefig(_outdir + "/{0}_{1}.pdf".format(mc_samp, unc), bbox_inches="tight")
            plt.close(fig)
            del fig

def make_pdf_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, hd, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir, datataking_year = args

    hist_template = copy.deepcopy(hd)
    hist_template.contents[:] = 0
    hist_template.contents_w2[:] = 0
   
    
    hmc = []
    
    for mc_samp in mc_samples:
        h = res[mc_samp][weight]
        h = h * weight_xs[mc_samp]
        h.label = "{0} ({1:.1E})".format(mc_samp, np.sum(h.contents))
                
        hmc += [h]
    
    htot_nominal = sum(hmc, hist_template)
    htot_variated = {}
    hdelta_quadrature = np.zeros_like(hist_template.contents)
    
    for sdir in ["__up", "__down"]:
        for unc in shape_systematics:
            if (unc + sdir) in res[mc_samp]:
                htot_variated[unc + sdir] = sum([
                    res[mc_samp][unc + sdir]* weight_xs[mc_samp] for mc_samp in mc_samples
                ], hist_template)
                hdelta_quadrature += (htot_nominal.contents - htot_variated[unc+sdir].contents)**2
            
    hdelta_quadrature_stat = np.sqrt(htot_nominal.contents_w2)
    hdelta_quadrature_stat_syst = np.sqrt(hdelta_quadrature_stat**2 + hdelta_quadrature)
    hd.label = "data ({0:.1E})".format(np.sum(hd.contents))

    if var == "hist_inv_mass_d":
        mask_inv_mass(hd)
    #    hd.contents[0] = 0
    #    hd.contents_w2[0] = 0

    figure = plt.figure(figsize=(5,5), dpi=100)
    a1, a2 = plot_hist_ratio(hmc, hd,
        total_err_stat=hdelta_quadrature_stat, total_err_stat_syst=hdelta_quadrature_stat_syst, figure=figure)
    a2.grid(which="both", linewidth=0.5)
    # Ratio axis ticks
    ts = a2.set_yticks([0.5, 1.0, 1.5], minor=False)
    ts = a2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], minor=True)

    #a2.set_yticks(np.linspace(0.5,1.5, ))
    if var.startswith("hist_numjet"):
        a1.set_xticks(hd["edges"])

    a1.text(0.015,0.99, r"CMS internal, $L = {0:.2f}\ pb^{{-1}}$ ({1})".format(
        int_lumi, datataking_year) + 
        "\nd/m={0:.2f}".format(np.sum(hd.contents)/np.sum(htot_nominal.contents)) + 
        ", wd={0:.2E}".format(wasserstein_distance(htot_nominal.contents/np.sum(htot_nominal.contents), hd.contents/np.sum(hd.contents))),
        horizontalalignment='left',
        verticalalignment='top',
        transform=a1.transAxes,
        fontsize=6
    )
    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=4, loc=1, ncol=2)
    
    varname, catname = assign_plot_title_label(var)
    
    a1.set_title(catname + " ({0})".format(analysis_names[analysis]))
    a2.set_xlabel(varname)
    
    binwidth = np.diff(hd.edges)[0]
    a1.set_ylabel("events / bin [{0:.1f}]".format(binwidth))
    try:
        os.makedirs(outdir + "/png")
    except Exception as e:
        pass
    try:
        os.makedirs(outdir + "/pdf")
    except Exception as e:
        pass
    plt.savefig(outdir + "/pdf/{0}_{1}_{2}.pdf".format(analysis, var, weight), bbox_inches="tight")
    plt.savefig(outdir + "/png/{0}_{1}_{2}.png".format(analysis, var, weight), bbox_inches="tight", dpi=100)
    plt.close(figure)
    del figure
 
    return

def histstep(ax, edges, contents, **kwargs):
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []
    for istep in range(len(edges)-1):
        xmins += [edges[istep]]
        xmaxs += [edges[istep+1]]
        ymins += [contents[istep]]
        if istep + 1 < len(contents):
            ymaxs += [contents[istep+1]]

    if not "color" in kwargs:
        kwargs["color"] = next(ax._get_lines.prop_cycler)['color']

    ymaxs += [ymaxs[-1]]
    l0 = ax.hlines(ymins, xmins, xmaxs, **kwargs)
    l1 = ax.vlines(xmaxs, ymins, ymaxs, color=l0.get_color(), linestyles=l0.get_linestyle())
    return l0

def midpoints(arr):
    return arr[:-1] + np.diff(arr)/2.0

def plot_hist_step(ax, edges, contents, errors, kwargs_step={}, kwargs_errorbar={}):
    line = histstep(ax, edges, contents, **kwargs_step)
    ax.errorbar(midpoints(edges), contents, errors, lw=0, elinewidth=1, color=line.get_color()[0], **kwargs_errorbar)

def load_hist(hist_dict):
    return Histogram.from_dict({
        "edges": np.array(hist_dict["edges"]),
        "contents": np.array(hist_dict["contents"]),
        "contents_w2": np.array(hist_dict["contents_w2"]),
    })

def mask_inv_mass(hist):
    bin_idx1 = np.searchsorted(hist["edges"], 120) - 1
    bin_idx2 = np.searchsorted(hist["edges"], 130) + 1
    hist["contents"][bin_idx1:bin_idx2] = 0.0
    hist["contents_w2"][bin_idx1:bin_idx2] = 0.0

def create_variated_histos(
    hdict,
    baseline="nominal",
    variations=shape_systematics):
 
    if not baseline in hdict.keys():
        raise KeyError("baseline histogram missing")
    
    #hbase = copy.deepcopy(hdict[baseline])
    hbase = hdict[baseline]
    ret = Results(OrderedDict())
    ret["nominal"] = hbase
    for variation in variations:
        for vdir in ["up", "down"]:
            #print("create_variated_histos", variation, vdir)
            sname = "{0}__{1}".format(variation, vdir)
            if sname.endswith("__up"):
                sname2 = sname.replace("__up", "Up")
            elif sname.endswith("__down"):
                sname2 = sname.replace("__down", "Down")

            if sname not in hdict:
                #print("systematic", sname, "not found, taking baseline") 
                hret = hbase
            else:
                hret = hdict[sname]
            ret[sname2] = hret
    return ret

def create_datacard(dict_procs, parameter_name, all_processes, histname, baseline, variations, weight_xs):
    
    ret = Results(OrderedDict())
    event_counts = {}

    hists_mc = []
 
    #print("create_datacard processes=", all_processes)
    for proc in all_processes:
        #print("create_datacard", proc)
        rr = dict_procs[proc]
        _variations = variations

        #don't produce variated histograms for data
        if proc == "data":
            _variations = []

        variated_histos = create_variated_histos(rr, baseline, _variations)

        for syst_name, histo in variated_histos.items():
            if proc != "data":
                histo = histo * weight_xs[proc]

            if syst_name == "nominal":

                event_counts[proc] = np.sum(histo.contents)
                #print(proc, syst_name, np.sum(histo.contents))
                if proc != "data":
                    hists_mc += [histo]
            #create histogram name for combine datacard
            hist_name = "{0}__{2}".format(proc, histname, syst_name)
            if hist_name == "data__nominal":
                hist_name = "data_obs"
            hist_name = hist_name.replace("__nominal", "")
            
            ret[hist_name] = copy.deepcopy(histo)

    assert(len(hists_mc) > 0)
    hist_mc_tot = copy.deepcopy(hists_mc[0])
    for h in hists_mc[:1]:
        hist_mc_tot += h
    ret["data_fake"] = hist_mc_tot
 
    return ret, event_counts

def save_datacard(dc, outfile):
    fi = uproot.recreate(outfile)
    for histo_name in dc.keys():
        fi[histo_name] = to_th1(dc[histo_name], histo_name)
    fi.close()

def create_datacard_combine_wrap(args):
    return create_datacard_combine(*args)

def create_datacard_combine(
    dict_procs, parameter_name,
    all_processes,
    signal_processes,
    histname, baseline,
    weight_xs,
    variations,
    common_scale_uncertainties,
    scale_uncertainties,
    txtfile_name
    ):
     
    dc, event_counts = create_datacard(
        dict_procs, parameter_name, all_processes,
        histname, baseline, variations, weight_xs)
    rootfile_name = txtfile_name.replace(".txt", ".root")
    
    save_datacard(dc, rootfile_name)
 
    all_processes.pop(all_processes.index("data"))

    shape_uncertainties = {v: 1.0 for v in variations}
    cat = Category(
        name=histname,
        processes=list(all_processes),
        signal_processes=signal_processes,
        common_shape_uncertainties=shape_uncertainties,
        common_scale_uncertainties=common_scale_uncertainties,
        scale_uncertainties=scale_uncertainties,
     )
    
    categories = [cat]

    filenames = {}
    for cat in categories:
        filenames[cat.full_name] = rootfile_name

    PrintDatacard(categories, event_counts, filenames, txtfile_name)

from uproot_methods.classes.TH1 import from_numpy

def to_th1(hdict, name):
    content = np.array(hdict.contents)
    content_w2 = np.array(hdict.contents_w2)
    edges = np.array(hdict.edges)
    
    #remove inf/nan just in case
    content[np.isinf(content)] = 0
    content_w2[np.isinf(content_w2)] = 0

    content[np.isnan(content)] = 0
    content_w2[np.isnan(content_w2)] = 0
    
    #update the error bars
    centers = (edges[:-1] + edges[1:]) / 2.0
    th1 = from_numpy((content, edges))
    th1._fName = name
    th1._fSumw2 = np.array(hdict.contents_w2)
    th1._fTsumw2 = np.array(hdict.contents_w2).sum()
    th1._fTsumwx2 = np.array(hdict.contents_w2 * centers).sum()

    return th1

class Category:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.full_name = self.name
        self.rebin = kwargs.get("rebin", 1)
        self.do_limit = kwargs.get("do_limit", True)


        self.cuts = kwargs.get("cuts", [])

        self.processes = kwargs.get("processes", [])
        self.data_processes = kwargs.get("data_processes", [])
        self.signal_processes = kwargs.get("signal_processes", [])
        
        #[process][systematic] -> scale factor in datacard
        self.shape_uncertainties = {}
        self.scale_uncertainties = {}

        #[syst] -> scale factor, common for all processes
        common_shape_uncertainties = kwargs.get("common_shape_uncertainties", {})
        common_scale_uncertainties = kwargs.get("common_scale_uncertainties", {})
        for proc in self.processes:
            self.shape_uncertainties[proc] = {}
            self.scale_uncertainties[proc] = {}
            for systname, systval in common_shape_uncertainties.items():
                self.shape_uncertainties[proc][systname] = systval
            for systname, systval in common_scale_uncertainties.items():
                self.scale_uncertainties[proc][systname] = systval

        #Load the process-dependent shape uncertainties
        self.proc_shape_uncertainties = kwargs.get("shape_uncertainties", {})
        for proc, v in self.proc_shape_uncertainties.items():
            self.shape_uncertainties[proc].update(v)
        
        self.proc_scale_uncertainties = kwargs.get("scale_uncertainties", {})
        for proc, v in self.proc_scale_uncertainties.items():
            if not (proc in self.scale_uncertainties):
                self.scale_uncertainties[proc] = {}
            self.scale_uncertainties[proc].update(v)

def PrintDatacard(categories, event_counts, filenames, ofname):
    dcof = open(ofname, "w")
    
    number_of_bins = len(categories)
    number_of_backgrounds = 0
    
    backgrounds = []    
    signals = []    
    for cat in categories:
        for proc in cat.processes:
            if (proc in cat.signal_processes):
                signals += [proc]
            else:
                backgrounds += [proc]
    
    backgrounds = set(backgrounds)
    signals = set(signals)
    number_of_backgrounds = len(backgrounds)
    number_of_signals = len(signals)
    analysis_categories = list(set([c.full_name for c in categories]))

    dcof.write("imax {0}\n".format(number_of_bins))
    dcof.write("jmax {0}\n".format(number_of_backgrounds + number_of_signals - 1))
    dcof.write("kmax *\n")
    dcof.write("---------------\n")

    for cat in categories:
#old format
#        dcof.write("shapes * {0} {1} $PROCESS__$CHANNEL $PROCESS__$CHANNEL__$SYSTEMATIC\n".format(
        dcof.write("shapes * {0} {1} $PROCESS $PROCESS__$SYSTEMATIC\n".format(
            cat.full_name,
            os.path.basename(filenames[cat.full_name])
        ))

    dcof.write("---------------\n")

    dcof.write("bin\t" +  "\t".join(analysis_categories) + "\n")
    dcof.write("observation\t" + "\t".join("-1" for _ in analysis_categories) + "\n")
    dcof.write("---------------\n")

    bins        = []
    processes_0 = []
    processes_1 = []
    rates       = []

    for cat in categories:
        for i_sample, sample in enumerate(cat.processes):
            bins.append(cat.full_name)
            processes_0.append(sample)
            if sample in cat.signal_processes:
                i_sample = -i_sample
            processes_1.append(str(i_sample))
            rates.append("{0}".format(event_counts[sample]))
    
    #Write process lines (names and IDs)
    dcof.write("bin\t"+"\t".join(bins)+"\n")
    dcof.write("process\t"+"\t".join(processes_0)+"\n")
    dcof.write("process\t"+"\t".join(processes_1)+"\n")
    dcof.write("rate\t"+"\t".join(rates)+"\n")
    dcof.write("---------------\n")

    # Gather all shape uncerainties
    all_shape_uncerts = []
    all_scale_uncerts = []
    for cat in categories:
        for proc in cat.processes:
            all_shape_uncerts.extend(cat.shape_uncertainties[proc].keys())
            all_scale_uncerts.extend(cat.scale_uncertainties[proc].keys())
    # Uniquify
    all_shape_uncerts = sorted(list(set(all_shape_uncerts)))
    all_scale_uncerts = sorted(list(set(all_scale_uncerts)))

    #print out shape uncertainties
    for syst in all_shape_uncerts:
        dcof.write(syst + "\t shape \t")
        for cat in categories:
            for proc in cat.processes:
                if (proc in cat.shape_uncertainties.keys() and
                    syst in cat.shape_uncertainties[proc].keys()):
                    dcof.write(str(cat.shape_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")


    #print out scale uncertainties
    for syst in all_scale_uncerts:
        dcof.write(syst + "\t lnN \t")
        for cat in categories:
            for proc in cat.processes:
                if (proc in cat.scale_uncertainties.keys() and
                    syst in cat.scale_uncertainties[proc].keys()):
                    dcof.write(str(cat.scale_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")

    #create nuisance groups for easy manipulation and freezing
    nuisance_groups = {}
    for nuisance_group, nuisances in nuisance_groups.items():
        good_nuisances = []
        for nui in nuisances:
            good_nuisances += [nui]
        dcof.write("{0} group = {1}\n".format(nuisance_group, " ".join(good_nuisances)))
    
    #dcof.write("* autoMCStats 20\n")
    #
    #shapename = os.path.basename(datacard.output_datacardname)
    #shapename_base = shapename.split(".")[0]
    dcof.write("\n")
    dcof.write("# Execute with:\n")
    dcof.write("# combine -n {0} -M FitDiagnostics -t -1 {1} \n".format(cat.full_name, os.path.basename(ofname)))

if __name__ == "__main__":

    cmdline_args = parse_args()

    pool = multiprocessing.Pool(24)

    from pars import cross_sections, categories
    from pars import signal_samples, shape_systematics, common_scale_uncertainties, scale_uncertainties

    #create a list of all the processes that need to be loaded from the result files
    mc_samples_load = set()
    for catname, category_dict in categories.items():
        for process in category_dict["datacard_processes"]:
            mc_samples_load.add(process)
    mc_samples_load = list(mc_samples_load)

    eras = []
    data_results_glob = cmdline_args.input + "/results/data_*.pkl"
    print("looking for {0}".format(data_results_glob))
    data_results = glob.glob(data_results_glob)
    if len(data_results) == 0:
        raise Exception("Did not find any data_*.pkl files in {0}, please check that this is a valid results directory and that the merge step has been completed".format(data_results_glob))

    for dr in data_results:
        dr_filename = os.path.basename(dr)
        dr_filename_noext = dr_filename.split(".")[0]
        name, era = dr_filename_noext.split("_")
        eras += [era]
    print("Will make datacards and control plots for eras {0}".format(eras))

    for era in eras:
        rea = {}
        genweights = {}
        weight_xs = {}
        datacard_args = []
        plot_args = []
        
        analysis = "results"
        input_folder = cmdline_args.input
        dd = "{0}/{1}".format(input_folder, analysis)
        res = {} 
        res["data"] = pickle.load(open(dd + "/data_{0}.pkl".format(era), "rb"))
        for mc_samp in mc_samples_load:
            res_file_name = dd + "/{0}_{1}.pkl".format(mc_samp, era)
            try:
                res[mc_samp] = pickle.load(open(res_file_name, "rb"))
            except Exception as e:
                print("Could not find results file {0}, skipping process {1}".format(res_file_name, mc_samp))

        analyses = [k for k in res["data"].keys() if not k in ["cache_metadata", "num_events"]]

        for analysis in analyses:
            print("processing analysis {0}".format(analysis))
            outdir = "{0}/{1}/plots/{2}".format(input_folder, analysis, era)
            outdir_datacards = "{0}/{1}/datacards/{2}".format(input_folder, analysis, era)
            try:
                os.makedirs(outdir)
            except FileExistsError as e:
                pass
            try:
                os.makedirs(outdir_datacards)
            except FileExistsError as e:
                pass

            #in inverse picobarns
            int_lumi = res["data"]["baseline"]["int_lumi"]
            for mc_samp in res.keys():
                if mc_samp != "data":
                    genweights[mc_samp] = res[mc_samp]["genEventSumw"]
                    weight_xs[mc_samp] = cross_sections[mc_samp] * int_lumi / genweights[mc_samp]
           
            with open(outdir + "/normalization.json", "w") as fi:
                fi.write(json.dumps({
                    "weight_xs": weight_xs,
                    "genweights": genweights,
                    "int_lumi": int_lumi,
                    }, indent=2)
                )


            histnames = []
            if cmdline_args.histnames is None:
                histnames = [h for h in res["data"]["baseline"].keys() if h.startswith("hist__")]
                print("Will create datacards and plots for all histograms")
                print("Use commandline option --histnames hist__dimuon__leading_muon_pt --histnames hist__dimuon__subleading_muon_pt ... to change that")
            else:
                histnames = cmdline_args.histnames
            print("Processing histnames", histnames)
            
            for var in histnames:
                if var in ["hist_puweight", "hist__dijet_inv_mass_gen", "hist__dnn_presel__dnn_pred"]:
                    print("Skipping {0}".format(var))
                    continue

                if ("h_peak" in var):
                    mc_samples = categories["h_peak"]["datacard_processes"]
                elif ("h_sideband" in var):
                    mc_samples = categories["h_sideband"]["datacard_processes"]
                elif ("z_peak" in var):
                    mc_samples = categories["z_peak"]["datacard_processes"]
                else:
                    mc_samples = categories["dimuon"]["datacard_processes"]


                #If we specified to only use certain processes in the datacard, keep only those
                if cmdline_args.keep_processes is None:
                    pass
                else:
                    mc_samples_new = []
                    for proc in mc_samples:
                        print(proc)
                        if proc in cmdline_args.keep_processes:
                            mc_samples_new += [proc]
                    mc_samples = mc_samples_new
                if len(mc_samples) == 0:
                    raise Exception(
                        "Could not match any MC process to histogram {0}, ".format(var) + 
                        "please check the definition in pars.py -> categories as "
                        "well as --keep-processes commandline option."
                        )


                histos = {s: res[s][analysis][var] for s in mc_samples + ["data"]}
                print(era, analysis, var)
                datacard_args += [
                    (histos,
                    analysis,
                    ["data"] + mc_samples,
                    signal_samples,
                    var,
                    "nominal",
                    weight_xs,
                    shape_systematics,
                    common_scale_uncertainties,
                    scale_uncertainties,
                    outdir_datacards + "/{0}.txt".format(var)
                )]

                hdata = res["data"][analysis][var]["nominal"]
                plot_args += [(
                    histos, hdata, mc_samples, analysis,
                    var, "nominal", weight_xs, int_lumi, outdir, era)]
        rets = list(pool.map(plot_variations, plot_args))
        rets = list(pool.map(create_datacard_combine_wrap, datacard_args))
        rets = list(pool.map(make_pdf_plot, plot_args))

        #for args, retval in zip(datacard_args, rets):
        #    res, hd, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir, datataking_year = args
        #    htot_nominal, hd, htot_variated, hdelta_quadrature = retval
        #    wd = wasserstein_distance(htot_nominal.contents/np.sum(htot_nominal.contents), hd.contents/np.sum(hd.contents))
        #    print("DataToMC", analysis, var, wd)
