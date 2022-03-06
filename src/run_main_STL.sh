#!/usr/bin/env bash

source $HOME/.bashrc
source activate structured_mtl

echo $@
date

echo "start"
python main_STL.py $@
echo "end"
date
