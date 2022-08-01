import os
import sys
import json
import git
import yaml
import glob
from tqdm import tqdm
from shutil import rmtree

import uproot
import numpy as np
import pandas as pd

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "../Training/python")

# epsilon = 0.0001
epsilon = 0.0000001

class FeatureDecoder:
    """A Class to compere the difference of an input tensor"""

    def __init__(self, cfg):
        path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
        train_cfg_path = f'{path_to_artifacts}/input_cfg/training_cfg.yaml'
        with open(train_cfg_path, "r") as stream:
            train_cfg = yaml.safe_load(stream)
        all_features = {}
        for key in train_cfg["Features_all"].keys():
            all_features[key] = []
            for var in train_cfg["Features_all"][key]:
                assert(len(list(var.keys()))==1)
                var_str = list(var.keys())[0]
                if not(var_str in train_cfg["Features_disable"][key]):
                    all_features[key].append(var_str)

        self.feature_map = {}
        for tensor in cfg.tensor_map.keys():
            self.feature_map[tensor] = []
            for block in cfg.tensor_map[tensor]:
                self.feature_map[tensor].extend(all_features[block])

        for elem in self.feature_map.keys():
            print("Tensor added to the map:",elem, len(self.feature_map[elem]))
        
    def get(self, tensor_name, index):
        return self.feature_map[tensor_name][index]


def compare_ids(cfg, sort=False, print_n=30, plot_deltas=True):
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')

    prediction_path = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred.h5'

    dfs = []
    dfs_names = [
            "predictions",
            "targets",
            "propagated_vars"
            ]
    for name in dfs_names:
        dfs.append(pd.read_hdf(prediction_path, key=name))
    
    df_dis = pd.concat(dfs, axis=1)
    df_dis["node_jet_cmssw"] = -1
    df_dis["node_tau_cmssw"] = -1


    with tqdm(total=df_dis.shape[0]) as pbar:
        for jet_i in range(df_dis.shape[0]):
            run = int(df_dis["run"].iloc[jet_i])
            lumi = int(df_dis["lumi"].iloc[jet_i])
            evt = int(df_dis["evt"].iloc[jet_i])
            jet_index = int(df_dis["jet_index"].iloc[jet_i])

            # print(run, lumi, evt, jet_index)

            # pyhton_file = f'tensor_{run}_{lumi}_{evt}_jet_{jet_index}.npy'
            # data_python = np.load(f'{path_to_artifacts}/predictions/{cfg.sample_alias}/eventTuple_pred_input/{cmssw_file}',allow_pickle=True)[()]

            if run==lumi==evt==jet_index==-1:
                pbar.update(1)
                continue

            cmssw_file = f'distag_{run}_{lumi}_{evt}_jet_{jet_index}.json'
            with open(f'{cfg.cmssw_input.input_dir}/{cmssw_file}') as file:
                data_cmssw = json.load(file)

            df_dis.at[jet_i,"node_jet_cmssw"], df_dis.at[jet_i,"node_tau_cmssw"] = data_cmssw["Output"][0], data_cmssw["Output"][1]
            pbar.update(1)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 500):
    #     print(df_dis.head(10))

    df_dis[df_dis < 0] = None

    df_dis[f'delta'] = df_dis.node_tau_pred - df_dis.node_tau_cmssw
    df_dis[f'abs_delta'] = df_dis[f'delta'].abs()

    print_n=10

    df_dis_noNan = df_dis.sort_values(['abs_delta'], ascending=False)
    if print_n: df_dis_noNan = df_dis_noNan[:print_n]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 500):  # more options can be specified also
        print(df_dis_noNan)

    if plot_deltas:
        img_path = 'deltaIDs'
        if not os.path.exists(img_path): os.makedirs(img_path)
        plt.hist(df_dis[f'delta'], density=True, bins=100)  # density=False would make counts
        plt.ylabel('arb. units')
        plt.xlabel(f'delta')
        plt.savefig(f'{img_path}/delta.png')
        plt.cla()
        plt.clf()
        plt.hist(df_dis[f'delta'], density=True, bins=100)  # density=False would make counts
        plt.ylabel('arb. units')
        plt.xlabel(f'delta')
        plt.yscale('log')
        plt.savefig(f'{img_path}/delta_log.png')
        plt.cla()
        plt.clf()
        plt.hist(df_dis[f'delta'], density=True, bins=100,range=[-0.000002, 0.000002])  # density=False would make counts
        plt.ylabel('arb. units')
        plt.xlabel(f'delta')
        plt.savefig(f'{img_path}/delta_max10E-06.png')
        plt.cla()
        plt.clf()
        with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
            mlflow.log_artifact(img_path, f'predictions/{cfg.sample_alias}')

def compare_input(cfg, print_grid=False):
    # assert(cfg.compare_input)
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    file_cmssw_path = to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.cmssw_input.input_dir}')
    files_cmssw_python = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred_input'

    file_idx = {
            "run" : cfg.cmssw_input.run,
            "lumi" : cfg.cmssw_input.lumi,
            "evt" : cfg.cmssw_input.evt,
            "jet_index" : cfg.cmssw_input.jet_index
        }

    cmssw_file = f'distag_{file_idx["run"]}_{file_idx["lumi"]}_{file_idx["evt"]}_jet_{file_idx["jet_index"]}.json'
    with open(f'{cfg.cmssw_input.input_dir}/{cmssw_file}') as file:
        json_input = json.load(file)
        data_cmssw = {}
        for tensor_name in json_input.keys():
            data_cmssw[tensor_name] = np.array(json_input[tensor_name])

    # data_python = np.load(f'{files_cmssw_python}/tensor_{event}_{index}.npy',allow_pickle=True)[()]
    pyhton_file = f'tensor_{file_idx["run"]}_{file_idx["lumi"]}_{file_idx["evt"]}_jet_{file_idx["jet_index"]}.npy'
    data_python = np.load(f'{path_to_artifacts}/predictions/{cfg.sample_alias}/eventTuple_pred_input/{pyhton_file}',allow_pickle=True)[()]
    for key in data_python.keys():
        data_python[key] = np.expand_dims(data_python[key], axis=0)

    # print(data_cmssw["PfCand"][:10])
    # print(data_python["PfCand"][:10])

    map_f = FeatureDecoder(cfg)
    print(map_f)

    for key in data_python.keys():

        print("Shape consistency check:",data_cmssw[key].shape, data_python[key].shape)
        assert(data_cmssw[key].shape == data_python[key].shape)

    print("Check number of pfCands consistency:")
    cmssw_n = np.sum((data_cmssw["PfCand"][0][:,0] == 1))
    python_n = np.sum((data_python["PfCand"][0][:,0] == 1))
    print("counts", cmssw_n, python_n)
    assert(cmssw_n==python_n)

    for key in data_python.keys():

        number = data_cmssw[key][0].shape[0]
        print("Check check feature consistency: ",key)

        for i in range(number):

            if(data_cmssw["PfCand"][0][i][0]!=1): continue

            delta = np.abs(data_cmssw[key][0][i] - data_python[key][0][i])
            # print(np.c_[data_cmssw[key][0][i], data_python[key][0][i]])
            # delta = np.sum(delta, axis=0)

            f_idx = np.where(delta > epsilon)
            print(f"Inconsistent features: pfcand=",i)
            for f in np.unique(f_idx):
                print("-> ",map_f.get(key,f), " (",data_cmssw[key][0][i][f], data_python[key][0][i][f],") ", " delta =", delta[f])


@hydra.main(config_path='configs', config_name='input_compare')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    
    if 'compare_ids' in cfg.mode: compare_ids(cfg)
    if 'compare_input' in cfg.mode: compare_input(cfg)

if __name__ == '__main__':
    main()