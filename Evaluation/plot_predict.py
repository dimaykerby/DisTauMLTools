import os
import math
import json
import pandas as pd
from collections import defaultdict
from dataclasses import fields

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig, ListConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import eval_tools

@hydra.main(config_path='configs/eval_dis', config_name='disp_taus')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")

    # setting paths
    # path_to_weights_taus = to_absolute_path(cfg.path_to_weights_taus) if cfg.path_to_weights_taus is not None else None
    # path_to_weights_vs_type = to_absolute_path(cfg.path_to_weights_vs_type) if cfg.path_to_weights_vs_type is not None else None
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    output_json_path = f'{path_to_artifacts}/performance.json'

    # init Discriminator() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.Discriminator))
    init_params = {k:v for k,v in cfg.discriminator.items() if k in field_names}
    discriminator = eval_tools.Discriminator(**init_params)
    
    # init PlotSetup() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.PlotSetup))
    init_params = {k:v for k,v in cfg.plot_setup.items() if k in field_names}
    plot_setup = eval_tools.PlotSetup(**init_params)

    # construct branches to be read from input files
    input_branches = OmegaConf.to_object(cfg.input_branches)
    id_branches = []
    if ((_b:=discriminator.pred_column) is not None) and (cfg.path_to_pred is None):
        id_branches.append(_b)
    if (_b:=discriminator.wp_column) is not None:
        id_branches.append(_b)

    # loop over input samples
    df_list = []
    print()
    for sample_alias, tau_types in cfg.input_samples.items():
        input_files, pred_files, target_files = eval_tools.prepare_filelists(sample_alias, cfg.path_to_input, cfg.path_to_pred, cfg.path_to_target, path_to_artifacts)

        # loop over all input files per sample with associated predictions/targets (if present) and combine together into df
        print(f'[INFO] Creating dataframe for sample: {sample_alias}')
        for input_file, pred_file, target_file in zip(input_files, pred_files, target_files):
            df = eval_tools.create_df(input_file, input_branches, id_branches, pred_file, target_file, None, # weights functionality is WIP
                                            cfg.discriminator.pred_column_prefix,
                                            cfg.discriminator.target_column_prefix)
            gen_selection = ' or '.join([f'(gen_{tau_type}==1)' for tau_type in tau_types]) # gen_* are constructed in `add_targets()`
            df = df.query(gen_selection)
            df_list.append(df)
    df_all = pd.concat(df_list)

    # df_all.to_csv('/afs/desy.de/user/m/mykytaua/nfscms/softDeepTau/RecoML/DisTauTag/TauMLTools/Evaluation/DeepTauId_predictions_old.csv', encoding='utf-8')
    # exit()

    # apply selection
    if cfg.cuts is not None: df_all = df_all.query(cfg.cuts)

    # # inverse scaling
    # df_all['tau_pt'] = df_all.tau_pt*(1000 - 20) + 20
    
    # dump curves' data into json file
    json_exists = os.path.exists(output_json_path)
    json_open_mode = 'r+' if json_exists else 'w'
    with open(output_json_path, json_open_mode) as json_file:
        if json_exists: # read performance data to append additional info 
            performance_data = json.load(json_file)
        else: # create dictionary to fill with data
            performance_data = {'name': discriminator.name, 'period': cfg.period, 'metrics': defaultdict(list), 
                                'roc_curve': defaultdict(list), 'roc_wp': defaultdict(list)}

        # loop over pt bins
        print(f'\n{discriminator.name}')
        for L_index, (L_min, L_max) in enumerate(cfg.L_bins):
            for eta_index, (eta_min, eta_max) in enumerate(cfg.eta_bins):
                for pt_index, (pt_min, pt_max) in enumerate(cfg.pt_bins):
                    
                    output_name=f'predict_{cfg.dataset_alias}_pt_{pt_min}_{pt_max}_eta_{eta_min}_{eta_max}_L_{L_min}_{L_max}.png'

                    # L_bins are in cylindrical coordinates
                    # L_cut = f'((Lxy>{rho_min} and Lxy<{rho_max} and abs(Lz)<{z_min}) or (abs(Lz)>{z_min} and abs(Lz)<{z_max} and Lxy<{rho_max}))'
                    # # apply pt/eta/dm bin selection
                    # df_cut = df_all.query(f'jet_pt >= {pt_min} and jet_pt < {pt_max} and abs(jet_eta) >= {eta_min} and abs(jet_eta) < {eta_max} and ({L_cut} or (gen_tau != 1))') # L cut only for signal
                    L_cut = f'( Lrel >= {L_min} and Lrel <= {L_max} )'
                    df_cut = df_all.query(f'jet_pt >= {pt_min} and jet_pt < {pt_max} and abs(jet_eta) >= {eta_min} and abs(jet_eta) < {eta_max} and ({L_cut} or (gen_tau != 1))') # L cut only for signal
                    
                    if df_cut.shape[0] == 0:
                        print("Warning: bin with pt ({}, {}) and eta ({}, {}) is empty.".format(pt_min, pt_max, eta_min, eta_max))
                        continue
                    print(f'\n-----> pt bin: [{pt_min}, {pt_max}], eta bin: [{eta_min}, {eta_max}], L [{L_min}, {L_max}]')
                    print('[INFO] counts:\n', df_cut[['gen_tau', f'gen_{cfg.vs_type}']].value_counts())

                    # create roc curve and working points
                    # roc, wp_roc = discriminator.create_roc_curve(df_cut)
                    # if roc is not None:

                    pred_hists = {}
                    count = {}

                    pred_hists["tau"] = df_cut.query('gen_tau==1')[discriminator.pred_column]
                    print(pred_hists["tau"].count())
                    # count = pred_hists["tau"][pred_hists["tau"] ]
                    plt.hist(pred_hists["tau"].to_numpy(), 200, density=True, histtype='stepfilled',color='red',alpha=0.75, label='tau',range=(0.0,1.0))

                    pred_hists["jet"] = df_cut.query('gen_jet==1')[discriminator.pred_column]
                    print(pred_hists["jet"].count())
                    plt.hist(pred_hists["jet"].to_numpy(), 200, density=True, histtype='stepfilled',color='blue',alpha=0.75, label='jet',range=(0.0,1.0))

                    plt.xlabel('ids')
                    # plt.xlim(0.0, 1.0)
                    plt.yscale("log")
                    plt.grid(True)
                    
                    plt.savefig(output_name)
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()

                    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id):
                        mlflow.log_artifact(output_name, 'plots_pred')
   
    
if __name__ == '__main__':
    main()
