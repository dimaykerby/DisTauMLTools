# !/bin/bash

PATH_TO_MLFLOW=$1

if ! [ -d "$PATH_TO_MLFLOW" ]; then
    echo "ERROR: directory with mlflow output '$PATH_TO_MLFLOW' not found."
    exit 1
fi

if ! [[ $PATH_TO_MLFLOW == *"mlruns/" || $PATH_TO_MLFLOW == *"mlruns" ]]; then
  echo "ERROR: '$PATH_TO_MLFLOW' is not mlruns output directory."
  exit 1
fi

DIRS=`readlink -f ${PATH_TO_MLFLOW}/*/*/artifacts`

CMD=""
for PATH_GLOB in ${DIRS[@]}; do

    ls ${PATH_GLOB}/model_summary.txt || continue
    NAME=`cat ${PATH_GLOB}/model_summary.txt | awk '$1=="Model:"{print $2}'`
    echo $NAME " - added"
    HASH=`awk -F "/" '{print $(NF-1)}' <<< $PATH_GLOB`
    HASH=${HASH:0:5}
    echo ${HASH}

    CMD+="${NAME}_${HASH}_train:${PATH_GLOB}/tensorboard_logs/train,"
    CMD+="${NAME}_${HASH}_val:${PATH_GLOB}/tensorboard_logs/validation,"
    CMD+="${NAME}_${HASH}_steps:${PATH_GLOB}/tensorboard_logs/steps,"

done

CMD=`echo $CMD | head -c -2`
CMD="tensorboard --port=9090 --logdir_spec $CMD"
echo "$CMD"
eval $CMD
