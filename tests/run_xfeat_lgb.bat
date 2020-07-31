@rem 作成日2020/07/27 xfeatで特徴量取捨選択

set SRC=../src

call activate py37

call python %SRC%/run_xfeat_lgb.py -c config/run_xfeat_lgb.py.casual.yml
call python %SRC%/run_xfeat_lgb.py -c config/run_xfeat_lgb.py.registered.yml

pause
