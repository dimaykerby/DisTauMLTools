
# DATASET='ShuffleMergeFlat'
DATASET='TT_STAU'

allID=(
    84943ef984c24f5281206f8861cd64a7
    2429b37680c9449a9520dd9995b18352
    84c706f05c7c4a15bbcb90dc241026d6
    aa234384bf5040929163e2da76c71bae
    )

discriminators=(
    "SNN_v1_lite"
    "SNN_v1"
    "PartNet_v1"
    "PartNet_v1_lite"
    )

# for one_id in ${allID[@]}; do
#     echo "Apply for:"$one_id
#     python apply_training.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 run_id=84943ef984c24f5281206f8861cd64a7 path_to_input_dir=/pnfs/desy.de/cms/tier2/store/user/myshched/ntuples-tau-pog/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/crab_TTToSemiLeptonic/220603_162129/0002 input_filename=eventTuple_2000 sample_alias=TTToSemiLeptonic
#     python apply_training.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 run_id=84943ef984c24f5281206f8861cd64a7 path_to_input_dir=/pnfs/desy.de/cms/tier2/store/user/myshched/ntuples-tau-pog/SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau1000mm_v4/crab_STAU_longlived_M100/220510_120705/0000/ input_filename=eventTuple_6 sample_alias=STAU_M100_1000mm
# done

for ((i=1;i<=${#allID[@]};i++))
do
    echo "Evaluate:" ${allID[$i]} ${discriminators[$i]};

    python evaluate_performance.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 run_id=${allID[$i]} discriminator="${discriminators[$i]}" dataset_alias=${DATASET}
done


# Evaluation DeepTau:
# python evaluate_performance.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=2 run_id=997f18c9924c409eb80c1b7b757c127a discriminator=DeepTau_v2p1 'path_to_input="/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/TauMLTools/FlatMerge-output-test/*.root"' path_to_pred=null 'path_to_target="${path_to_mlflow}/${experiment_id}/fb57a260e56945cb8f8b1e11cb2bfbae/artifacts/predictions/{sample_alias}/*_pred.h5"' vs_type=jet dataset_alias=${DATASET} discriminator.wp_from=pred_column


# All plots:

python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[0.0, 0.2]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[0.2, 1.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[1.0, 5.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[5.0, 10.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[10.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[50.0, 100.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[20,100]' 'eta_bin=[0, 2.4]' 'L_bin=[100.0, 200.0]'

python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[0.0, 0.2]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[0.2, 1.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[1.0, 5.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[5.0, 10.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[10.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[50.0, 100.0]'
python plot_roc.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=13 dataset_alias=${DATASET} 'pt_bin=[100,1000]' 'eta_bin=[0, 2.4]' 'L_bin=[100.0, 200.0]'

# For cumulitive histograms:
# python plot_predict.py path_to_mlflow=../Training/python/DisTauTag/mlruns experiment_id=9 run_id=3f64c59aef7b476087ecfefc80209927 discriminator=SNN_v1 dataset_alias=TT_STAU
