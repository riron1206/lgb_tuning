@rem 作成日2020/07/27 run_lgb_reg_tuning.pyでregisteredとcasualでそれぞれモデル作りsubmitファイル作成までのパイプライン

set SRC=../src
set VAR=v1

call activate py37

@rem registeredとcasualでそれぞれモデル作る
@rem パラメータチューニング -cでパラメータのyamlファイル切り替える
call python %SRC%/run_lgb_reg_tuning.py -c config/run_lgb_reg_tuning.py.registered.yml

@rem 実行完了後、できたsubmitファイルを足し合わせる
call python %SRC%/util.py -o model/%VAR%_tuning -c1 model/%VAR%_tuning/casual/best/lgb-submission.csv -c2 model/%VAR%_tuning/registered/best/lgb-submission.csv -o_c_n blend_best_lgb-submission
call python %SRC%/util.py -o model/%VAR%_tuning -c1 model/%VAR%_tuning/casual/best_retrain/lgb-submission.csv -c2 model/%VAR%_tuning/registered/best_retrain/lgb-submission.csv -o_c_n blend_best_retrain_lgb-submission
call python %SRC%/util.py -o model/%VAR%_tuning -c1 model/%VAR%_tuning/casual/best_retrain/lgb-all-submission.csv -c2 model/%VAR%_tuning/registered/best_retrain/lgb-all-submission.csv -o_c_n blend_best_retrain_lgb-all-submission

pause
