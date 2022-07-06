
# DATASET='ShuffleMergeFlat'
DATASET='TT_GluGlu'

# Evaluation:
# python evaluate_performance.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=2 run_id=997f18c9924c409eb80c1b7b757c127a discriminator=DeepTau_v2p1 'path_to_input="/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/TauMLTools/FlatMerge-output-test/*.root"' path_to_pred=null 'path_to_target="${path_to_mlflow}/${experiment_id}/fb57a260e56945cb8f8b1e11cb2bfbae/artifacts/predictions/{sample_alias}/*_pred.h5"' vs_type=jet dataset_alias=${DATASET} discriminator.wp_from=pred_column

# python evaluate_performance.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=2 run_id=fb57a260e56945cb8f8b1e11cb2bfbae discriminator=PartNet_v1 dataset_alias=${DATASET}

# python evaluate_performance.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=2 run_id=528a5e4391c143fdad06451e8bf98a0e discriminator=SNN_v1 dataset_alias=${DATASET}

# All plots:

python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[0.0, 0.2]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[0.2, 1.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[1.0, 5.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[5.0, 10.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[10.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[50.0, 100.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[100.0, 200.0]'

python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[0.0, 0.2]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[0.2, 1.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[1.0, 5.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[5.0, 10.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[10.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[50.0, 100.0]'
# python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=3 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[100.0, 200.0]'

