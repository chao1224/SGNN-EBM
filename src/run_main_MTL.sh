#!/usr/bin/env bash

source $HOME/.bashrc
source activate structured_mtl

echo $@
date

echo "start"
python main_MTL.py $@
echo "end"
date
