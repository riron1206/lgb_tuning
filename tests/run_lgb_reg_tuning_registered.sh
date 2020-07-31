#!/bin/bash
PWDDIR=`pwd`

SRC=../src
VAR=v1

conda activate py37

# registeredとcasualでそれぞれモデル作る
# パラメータチューニング -cでパラメータのyamlファイル切り替える
python ${SRC}/run_lgb_reg_tuning.py -c config/run_lgb_reg_tuning.py.registered.yml

# 実行完了後、できたsubmitファイルをで足し合わせる
python ${SRC}/util.py -o model/${VAR}_tuning \
                      -c1 model/${VAR}_tuning/casual/best/lgb-submission.csv \
                      -c2 model/${VAR}_tuning/registered/best/lgb-submission.csv \
                      -o_c_n blend_best_lgb-submission
python ${SRC}/util.py -o model/${VAR}_tuning \
                      -c1 model/${VAR}_tuning/casual/best_retrain/lgb-submission.csv \
                      -c2 model/${VAR}_tuning/registered/best_retrain/lgb-submission.csv \
                      -o_c_n blend_best_retrain_lgb-submission
python ${SRC}/util.py -o model/${VAR}_tuning \
                      -c1 model/${VAR}_tuning/casual/best_retrain/lgb-all-submission.csv \
                      -c2 model/${VAR}_tuning/registered/best_retrain/lgb-all-submission.csv \
                      -o_c_n blend_best_retrain_lgb-all-submission
