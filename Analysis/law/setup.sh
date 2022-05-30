#!/usr/bin/env bash


action() {
    export X509_USER_PROXY=$HOME/public/x509_voms

    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    export PYTHONPATH="$this_dir:$PYTHONPATH"
    export LAW_HOME="$this_dir/.law"
    export LAW_CONFIG_FILE="$this_dir/law.cfg"

    export ANALYSIS_PATH="$this_dir"
    export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"

    source "/afs/desy.de/user/r/riegerma/public/law_sw/setup.sh" ""
    source "$( law completion )" ""
}
action
