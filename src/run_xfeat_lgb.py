"""
xfeat + lightGBMで特徴量エンジニアリング自動化
参考: https://github.com/pfnet-research/xfeat/blob/master/examples/feature_selection_with_gbdt_and_optuna.py
Usage:
    # feature_engineering 実行せず既存のカラムでbestなものだけ残す場合
    $ python run_xfeat_lgb.py -c config/run_xfeat_lgb.py.casual.yml

    # feature_engineering も実行する場合（列数1400ぐらいになりわけわからなくなり、精度も上がらず。n_trial増やせばよくなるのかなあ）
    $ python run_xfeat_lgb.py -c config/run_xfeat_lgb.py.casual.yml --is_feature_engineering
"""
import os
import sys
import argparse
import yaml
import copy
from functools import partial
import pathlib

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna

from util import Xfeat

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../github/xfeat")
from xfeat import (
    SelectCategorical,
    LabelEncoder,
    Pipeline,
    ConcatCombination,
    SelectNumerical,
    ArithmeticCombinations,
    GBDTFeatureExplorer,
    GBDTFeatureSelector,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run_lgb_cv(df, input_cols, target_col, lgb_params, cv_params):
    """lgbでcross-validation"""
    train_set = lgb.Dataset(df[input_cols], label=df[target_col])
    scores = lgb.cv(lgb_params, train_set, **cv_params)
    score = scores[f"{lgb_params['metric']}-mean"][-1]
    return score


def evaluate_dataframe(df, y, cv_params, lgb_params=None, target_col="target"):
    # データ前処理
    X_train, X_test, y_train, y_test = train_test_split(
        df.values, y, test_size=0.5, random_state=1
    )

    if lgb_params is None:
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "verbosity": -1,
        }

    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(lgb_params, train_set, **cv_params)
    rmsle_score = scores[f"{lgb_params['metric']}-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(
        lgb_params, train_set, num_boost_round=cv_params["num_boost_round"]
    )
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")

    return rmsle_score, test_rmsle_score


def objective(df, selector, params, trial):
    """
    Objective function for optuna
    特徴量取捨選択のパラメータ探索する
    """
    if selector is not None:
        # 特徴量取捨選択
        selector.set_trial(trial)
        selector.fit(df)
        input_cols = selector.get_selected_cols()
    else:
        input_cols = df.columns.tolist().remove(params["target_col"])

    # lgbのパラメータ
    lgb_params = {
        "objective": params["objective"],
        "metric": params["metric"],
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    # lgb.cvのパラメータ
    cv_params = {
        "num_boost_round": params["num_boost_round"],
        "seed": params["seed"],
        "nfold": params["nfold"],
        "stratified": False,  # 回帰なので層別サンプリングはしない
        "shuffle": True,
    }
    # Evaluate with selected columns
    return run_lgb_cv(df, input_cols, params["target_col"], lgb_params, cv_params)


def run_feature_selection(df, y, params, n_trials=20):
    """特徴量選択してoptunaでパラメチューニング実行"""
    # データ前処理
    df[params["target_col"]] = y
    df_train, _ = train_test_split(df, test_size=0.5, random_state=1)

    # 特徴量の列名（取捨選択前）
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)
    input_cols.remove(params["target_col"])

    # 特徴量選択用モデル取得
    selector = Xfeat.get_feature_explorer(
        input_cols,
        params["target_col"],
        objective=params["objective"],
        metric=params["metric"],
    )

    # パラメータチューニング
    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(objective, df_train, selector, params), n_trials=n_trials,
    )
    print(f"\nstudy.best_params:\n{study.best_params}")
    print(f"\nstudy.best_trial:\n{study.best_trial}")

    # bestな選択をした特徴量を返す
    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")
    return df[selected_cols], study.best_params


def get_data():
    """引数からデータロード"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=str,
        default="../tests/config/run_xfeat_lgb.py.casual.yml",
    )
    ap.add_argument(
        "-is_f_e",
        "--is_feature_engineering",
        action="store_const",
        const=True,
        default=False,
        help="feature_engineering 実行するか",
    )
    args = vars(ap.parse_args())
    with open(args["config"]) as f:
        config = yaml.load(f)
    args.update(config)
    if "num_feature" in config and "categorical_feature" in config:
        args["features"] = [*config["num_feature"], *config["categorical_feature"]]
    else:
        args["features"] = None

    if "n_trials" not in args:
        args["n_trials"] = 100

    df_train = pd.read_csv(os.path.join(args["input_dir"], "train.csv"))
    if args["features"] is None:
        cols = df_train.columns.to_list()
        for del_col in [
            "datetime",
            "casual_log",
            "registered_log",
            "count_log",
            "casual",
            "registered",
            "count",
        ]:
            if del_col in cols:
                cols.remove(del_col)
        args["features"] = cols
    X_train = df_train[args["features"]]
    y_train = df_train[args["target"]]

    df_test = pd.read_csv(os.path.join(args["input_dir"], "test.csv"))
    X_test = df_test[args["features"]]

    os.makedirs(args["output_dir"], exist_ok=True)

    return X_train, y_train, X_test, args


def main():
    X_train, y_train, X_test, args = get_data()

    cv_params = {
        "num_boost_round": 100,
        "stratified": False,  # 回帰なので層別サンプリングはしない
        "seed": 71,
        "nfold": 5,
        "shuffle": True,
    }

    print("--- 加工なし ---")
    print(X_train.shape)
    train_score, test_score = evaluate_dataframe(X_train, y_train, cv_params)
    pd.DataFrame({"train_score": [train_score], "test_score": [test_score]}).T.to_csv(
        os.path.join(args["output_dir"], "base_xfeat_score.tsv"), sep="\t"
    )

    if args["is_feature_engineering"]:
        print("--- feature_engineering 実行 ---")
        X_train = Xfeat.feature_engineering(X_train)
        X_test = Xfeat.feature_engineering(X_test)
        # ラベルつけて保存
        _X_train = X_train.copy()
        _X_train[args["target"]] = y_train
        _X_train.to_csv(
            os.path.join(args["output_dir"], "train_xfeat_feature_engineering.csv"),
            index=False,
        )
        X_test.to_csv(
            os.path.join(args["output_dir"], "test_xfeat_feature_engineering.csv"),
            index=False,
        )
        print("train: ", X_train.shape)
        print("test: ", X_test.shape)
        train_score, test_score = evaluate_dataframe(X_train, y_train, cv_params)
        pd.DataFrame(
            {"train_score": [train_score], "test_score": [test_score]}
        ).T.to_csv(
            os.path.join(args["output_dir"], "feature_engineering_score.tsv"), sep="\t"
        )

    print("--- GBDTFeatureSelector実行 ---")
    params = copy.deepcopy(cv_params)
    params["objective"] = "regression"
    params["metric"] = "rmse"
    params["target_col"] = args["target"]
    X_select, best_params = run_feature_selection(
        X_train, y_train, params, n_trials=args["n_trials"]
    )
    # ラベルつけて保存
    _X_select = X_select.copy()
    _X_select[args["target"]] = y_train
    _X_select.to_csv(
        os.path.join(args["output_dir"], "train_xfeat_feature_selection.csv"),
        index=False,
    )
    print("train: ", X_select.shape)
    train_score, test_score = evaluate_dataframe(X_select, y_train, cv_params)
    pd.DataFrame({"train_score": [train_score], "test_score": [test_score]}).T.to_csv(
        os.path.join(args["output_dir"], "GBDTFeatureSelector_score.tsv"), sep="\t"
    )

    # test setも見つけた列だけにする
    X_test = X_test[X_select.columns.to_list()]
    X_test.to_csv(
        os.path.join(args["output_dir"], "test_xfeat_feature_selection.csv"),
        index=False,
    )
    print("test: ", X_test.shape)


if __name__ == "__main__":
    main()
