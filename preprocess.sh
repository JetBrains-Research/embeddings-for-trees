#!/bin/bash

PS3="Choose data preprocessing step: "
options=("Download" "Build AST" "Collect vocabulary" "Convert to DGL format" "Upload to cloud")
select opt in "${options[@]}"
do
    case ${opt} in
        "Download")
            # download data
            PYTHONPATH='.' python data_workers/preprocess.py "$1" --download
            break
            ;;
        "Build AST")
            # build ast
            PYTHONPATH='.' python data_workers/preprocess.py "$1" --build_ast
            break
            ;;
        "Collect vocabulary")
            # collect vocabulary)
            PYTHONPATH='.' python data_workers/preprocess.py "$1" --collect_vocabulary \
            --split_vocabulary --n_tokens 190000 --n_labels 27000 --wrap_labels
            break
            ;;
        "Convert to DGL format")
            # convert to dgl format
            PYTHONPATH='.' python data_workers/preprocess.py "$1" \
                --convert --n_jobs -1 --batch_size 10000 --split_vocabulary \
                --tokens_to_leaves --max_token_len 5 --max_label_len 6 --wrap_labels
            break
            ;;
        "Upload to cloud")
            PYTHONPATH='.' python data_workers/preprocess.py "$1" --upload --store aws --tar_suffix "$2"
            break
            ;;
         *) echo "invalid option $REPLY";;
    esac
done
