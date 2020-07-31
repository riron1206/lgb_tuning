@rem 作成日2020/07/27 run_lgb_reg.pyでregisteredとcasualでそれぞれモデル作りsubmitファイル作成までのパイプライン

set SRC=../src
set VAR=v1

call activate py37

@rem registeredとcasualでそれぞれモデル作る
@rem パラメータチューニング -cでパラメータのyamlファイル切り替える
call python %SRC%/run_lgb_reg.py -c config/run_lgb_reg.py.casual.yml
call python %SRC%/run_lgb_reg.py -c config/run_lgb_reg.py.registered.yml

@rem 実行完了後、できたsubmitファイルをで足し合わせる
call python %SRC%/util.py -o model/%VAR% -c1 model/%VAR%/casual/lgb-submission.csv -c2 model/%VAR%/registered/lgb-submission.csv
call python %SRC%/util.py -o model/%VAR% -c1 model/%VAR%/casual/lgb-all-submission.csv -c2 model/%VAR%/registered/lgb-all-submission.csv -o_c_n blend_submission-all

pause
