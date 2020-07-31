#!/bin/bash
PWDDIR=`pwd`

# 作成日2020/07/27 xfeatで特徴量取捨選択

SRC=../src

conda activate py37

python ${SRC}/run_xfeat_lgb.py -c config/run_xfeat_lgb.py.casual.yml
python ${SRC}/run_xfeat_lgb.py -c config/run_xfeat_lgb.py.registered.yml
