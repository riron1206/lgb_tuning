#!/bin/bash
PWDDIR=`pwd`

SRC=../src
VAR=v1

conda activate py37

# registeredとcasualでそれぞれモデル作る
# パラメータチューニング -cでパラメータのyamlファイル切り替える
python ${SRC}/run_lgb_reg.py -c config/run_lgb_reg.py.casual.yml
python ${SRC}/run_lgb_reg.py -c config/run_lgb_reg.py.registered.yml

# 実行完了後、できたsubmitファイルをで足し合わせる
python ${SRC}/util.py -o model/${VAR} \
                      -c1 model/${VAR}/casual/lgb-submission.csv \
                      -c2 model/${VAR}/registered/lgb-submission.csv
python ${SRC}/util.py -o model/${VAR} \
                      -c1 model/${VAR}/casual/lgb-all-submission.csv\
                      -c2 model/${VAR}/registered/lgb-all-submission.csv \
                      -o_c_n blend_submission-all
