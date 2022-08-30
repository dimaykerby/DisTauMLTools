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
    if 'wp_thresholds' in init_params:
        if not (isinstance(init_params['wp_thresholds'], DictConfig) or isinstance(init_params['wp_thresholds'], dict)):
            if isinstance(init_params['wp_thresholds'], str): # assume that it's the filename to read WPs from
                with open(f"{path_to_artifacts}/{init_params['wp_thresholds']}", 'r') as f:
                    wp_thresholds = json.load(f)
                init_params['wp_thresholds'] = wp_thresholds[cfg['vs_type']] # pass laoded dict with thresholds to Discriminator() class
            else:
                raise RuntimeError(f"Expect `wp_thresholds` argument to be either dict-like or str, but got the type: {type(init_params['wp_thresholds'])}")
    else:
        wp_thresholds = None
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
    if cfg['cuts'] is not None:
        df_all = df_all.query(cfg.cuts)
    # if cfg['WPs_to_require'] is not None:
    #     for wp_vs_type, wp_name in cfg['WPs_to_require'].items():
    #         if wp_thresholds is not None:
    #             wp_thr = wp_thresholds[wp_vs_type][wp_name]
    #         else:
    #             if cfg['discriminator']['wp_thresholds_map'] is not None:
    #                 wp_thr = cfg['discriminator']['wp_thresholds_map'][wp_vs_type][wp_name]
    #             else:
    #                 raise RuntimeError('WP thresholds either via wp_thresholds_map or via input json file are not provided.')
    #         wp_cut = f"{cfg['discriminator']['pred_column_prefix']}{wp_vs_type} > {wp_thr}"
    #         df_all = df_all.query(wp_cut)
        

    # # inverse scaling
    # df_all['tau_pt'] = df_all.tau_pt*(1000 - 20) + 20
    
    # dump curves' data into json file
    json_exists = os.path.exists(output_json_path)
    json_open_mode = 'r+' if json_exists else 'w'
    with open(output_json_path, json_open_mode) as json_file:
        if json_exists: # read performance data to append additional info 
            performance_data = json.load(json_file)
        else: # create dictionary to fill with data
            performance_data = {'name': discriminator.name, 'period': cfg.period, 'metrics': defaultdict(list)}

        # loop over pt bins
        print(f'\n{discriminator.name}')
        for L_index, (L_min, L_max) in enumerate(cfg.L_bins):
            for eta_index, (eta_min, eta_max) in enumerate(cfg.eta_bins):
                for pt_index, (pt_min, pt_max) in enumerate(cfg.pt_bins):

                    # L_bins are in cylindrical coordinates
                    # L_cut = f'((Lxy>{rho_min} and Lxy<{rho_max} and abs(Lz)<{z_min}) or (abs(Lz)>{z_min} and abs(Lz)<{z_max} and Lxy<{rho_max}))'
                    # # apply pt/eta/dm bin selection
                    L_cut = f'( Lrel >= {L_min} and Lrel <= {L_max} )'
                    df_cut = df_all.query(f'jet_pt >= {pt_min} and jet_pt < {pt_max} and abs(jet_eta) >= {eta_min} and abs(jet_eta) < {eta_max} and ({L_cut} or (gen_tau != 1))') # L cut only for signal

                    if df_cut.shape[0] == 0:
                        print("Warning: bin with pt ({}, {}) and eta ({}, {}) is empty.".format(pt_min, pt_max, eta_min, eta_max))
                        continue
                    print(f'\n-----> pt bin: [{pt_min}, {pt_max}], eta bin: [{eta_min}, {eta_max}], L [{L_min}, {L_max}]')
                    print('[INFO] counts:\n', df_cut[['gen_tau', f'gen_{cfg.vs_type}']].value_counts())
                    # print(df_cut.query("(gen_tau==1) and (tau_byDeepTau2017v2p1VSjetraw>0)"))
                    # print(df_cut.query("(gen_jet==1) and (tau_byDeepTau2017v2p1VSjetraw>0)"))
                    # create roc curve and working points
                    roc, _ = discriminator.create_roc_curve(df_cut)
                    if roc is not None:
                        # prune the curve
                        lim = getattr(plot_setup,  'xlim')
                        x_range = lim[1] - lim[0] if lim is not None else 1
                        roc = roc.Prune(tpr_decimals=max(0, round(math.log10(1000 / x_range))))
                        if roc.auc_score is not None:
                            print(f'[INFO] ROC curve done, AUC = {roc.auc_score}')

                    # loop over [ROC curve, ROC curve WP] for a given discriminator and store its info into dict
                    for curve_type, curve in zip(['roc_curve'], [roc]):
                        if curve is None: continue
                        if json_exists and curve_type in performance_data['metrics'] \
                                        and (existing_curve := eval_tools.select_curve(performance_data['metrics'][curve_type], 
                                                                                        pt_min=pt_min, pt_max=pt_max, eta_min=eta_min, eta_max=eta_max, 
                                                                                        L_min=L_min, L_max=L_max,
                                                                                        vs_type=cfg.vs_type,
                                                                                        dataset_alias=cfg.dataset_alias)) is not None:
                            print(f'[INFO] Found already existing curve (type: {curve_type}) in json file for a specified set of parameters: will overwrite it.')
                            performance_data['metrics'][curve_type].remove(existing_curve)

                        curve_data = {
                            'pt_min': pt_min, 'pt_max': pt_max, 
                            'eta_min': eta_min, 'eta_max': eta_max,
                            'L_min': L_min, 'L_max': L_max,
                            'vs_type': cfg.vs_type,
                            'dataset_alias': cfg.dataset_alias,
                            'auc_score': curve.auc_score,
                            'false_positive_rate': eval_tools.FloatList(curve.pr[0, :].tolist()),
                            'true_positive_rate': eval_tools.FloatList(curve.pr[1, :].tolist()),
                        }
                        if curve.thresholds is not None:
                            curve_data['thresholds'] = eval_tools.FloatList(curve.thresholds.tolist())
                        if curve.pr_err is not None:
                            curve_data['false_positive_rate_up'] = eval_tools.FloatList(curve.pr_err[0, 0, :].tolist())
                            curve_data['false_positive_rate_down'] = eval_tools.FloatList(curve.pr_err[0, 1, :].tolist())
                            curve_data['true_positive_rate_up'] = eval_tools.FloatList(curve.pr_err[1, 0, :].tolist())
                            curve_data['true_positive_rate_down'] = eval_tools.FloatList(curve.pr_err[1, 1, :].tolist())

                        # plot setup for the curve
                        curve_data['plot_setup'] = {
                            'color': curve.color,
                            'dots_only': curve.dots_only,
                            'dashed': curve.dashed,
                            'marker_size': curve.marker_size
                        }
                        # curve_data['plot_setup']['ratio_title'] = 'MVA/DeepTau' if cfg.vs_type != 'mu' else 'cut based/DeepTau'
                        curve_data['plot_setup']['ratio_title'] = "ratio"

                        # plot setup for the curve
                        for lim_name in [ 'x', 'y', 'ratio_y' ]:
                            lim = getattr(plot_setup, lim_name + 'lim')
                            if lim is not None:
                                lim = OmegaConf.to_object(lim[eta_index][pt_index]) if isinstance(lim[0], (list, ListConfig)) else lim
                                curve_data['plot_setup'][lim_name + '_min'] = lim[0]
                                curve_data['plot_setup'][lim_name + '_max'] = lim[1]
                        for param_name in [ 'ylabel', 'yscale', 'ratio_yscale', 'legend_loc', 'ratio_ylabel_pad']:
                            val = getattr(plot_setup, param_name)
                            if val is not None:
                                curve_data['plot_setup'][param_name] = val

                        # plot setup for the curve
                        if cfg.plot_setup.public_plots:
                            if pt_max == 1000:
                                pt_text = r'$p_T > {}$ GeV'.format(pt_min)
                            elif pt_min == 20:
                                pt_text = r'$p_T < {}$ GeV'.format(pt_max)
                            else:
                                pt_text = r'$p_T\in ({}, {})$ GeV'.format(pt_min, pt_max)
                            curve_data['plot_setup']['pt_text'] = pt_text
                        else:
                            if cfg.plot_setup.inequality_in_title and (pt_min == 20 or pt_max == 1000) \
                                    and not (pt_min == 20 and pt_max == 1000):
                                if pt_min == 20:
                                    title_str = 'tau vs {}. pt < {} GeV'.format(cfg.vs_type, pt_max)
                                else:
                                    title_str = 'tau vs {}. pt > {} GeV'.format(cfg.vs_type, pt_min)
                            else:
                                title_str = 'tau vs {}. pt range ({}, {}) GeV'.format(cfg.vs_type, pt_min,
                                                                                        pt_max)
                            curve_data['plot_setup']['pt_text'] = title_str
                        curve_data['plot_setup']['eta_text'] = r'${} < |\eta| < {}$'.format(eta_min, eta_max)
                        # if len(dm_bin)==1:
                        #     curve_data['plot_setup']['dm_text'] = r'DM$ = {}$'.format(dm_bin[0])
                        # else:
                        #     curve_data['plot_setup']['dm_text'] = r'DM$ \in {}$'.format(dm_bin)

                        # append data for a given curve_type and pt bin
                        curve_data['plot_setup']['Lrel'] = r'${} < Lrel < {} cm$'.format(L_min, L_max)
                        performance_data['metrics'][curve_type].append(curve_data)

        json_file.seek(0) 
        json_file.write(json.dumps(performance_data, indent=4, cls=eval_tools.CustomJsonEncoder))
        json_file.truncate()
    print()
    
if __name__ == '__main__':
    main()
