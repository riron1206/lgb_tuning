# -*- coding: utf-8 -*-
"""
データ前処理
Usage:
    $ conda activate py37
    $ python preprocess.py -o ../data/preprocessed --is_boxcox --is_weather_del_3sigma
    $ python preprocess.py -o ../data/preprocessed_box_outlier --is_box_outlier  # 特徴量の外れ値削除
    $ python preprocess.py -o ../data/time_series --is_train_val_ts_split  # 時系列の分け方でtrain/validation setのcsv作成
    $ python preprocess.py -o ../data/add_feature --is_add_logs --is_add_target_shift --shift_row 24 --bin_cols windspeed  # 列追加
    $ python preprocess.py -o ../data/add_feature_v2 --is_add_time_count_class --is_add_split_datetime --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # 列追加v2
    $ python preprocess.py -o ../data/add_feature_v3 --is_add_time_count_class --is_add_split_datetime --is_add_registered_casual_mean_count --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --is_fillna_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # 列追加v3
    $ python preprocess.py -o ../data/tmp --is_add_time_count_class --is_add_split_datetime --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity  # 列追加v3の時にadd_time_count_class()いじったので、列追加v2との差分確認用
    $ python preprocess.py -o ../ch04-model-interface/input --is_labelencode_count_class --is_add_time_count_class --is_add_split_datetime --is_add_registered_casual_mean_count --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # lgb用v1
    $ python preprocess.py -o ../data/add_feature_v4 --is_log1p_count_cols --is_labelencode_count_class --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_time_count_class --is_add_split_datetime --is_add_registered_casual_mean_count --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # 列追加v4

    # casual用
    $ python preprocess.py -o ../data/add_feature_v4_casual -s_t casual --is_log1p_count_cols --is_labelencode_count_class --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_time_count_class --is_add_split_datetime --is_add_registered_casual_mean_count --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # 列追加v4
    # registered用
    $ python preprocess.py -o ../data/add_feature_v4_registered -s_t registered --is_log1p_count_cols --is_labelencode_count_class --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_time_count_class --is_add_split_datetime --is_add_registered_casual_mean_count --is_add_target_mean --is_add_elapsed_day --is_add_target_shift --shift_row 24 --bin_cols windspeed --clip_cols humidity # 列追加v4

    # count、casual、registered 兼用
    $ python preprocess.py -o ../data/add_feature_v5 --is_add_any_cols --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_elapsed_day # 列追加v5
    $ python preprocess.py -o ../data/add_feature_v6 --is_add_any_cols_v2 --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_elapsed_day --is_add_target_mean # 列追加v6
    $ python preprocess.py -o ../data/add_feature_v7 --is_add_any_cols_v2 --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_elapsed_day --is_add_target_mean --is_add_registered_casual_mean_count # 列追加v7

    # InClass用
    $ python preprocess.py -i ../data/orig_InClass/bikesharing-for-education_col_edit -o ../data/add_feature_v7_InClass --is_add_any_cols_v2 --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_elapsed_day --is_add_target_mean --is_add_registered_casual_mean_count # 列追加v7

    # target_encoding
    $ python preprocess.py -i ../data/orig_InClass/bikesharing-for-education_col_edit -o ../data/add_feature_v8_InClass -te_split_type TimeSeriesSplit --is_add_any_cols_v2 --is_add_am_pm_weather_mean --is_add_discomfort_index --is_add_elapsed_day --clip_cols humidity --is_add_next_holiday --is_add_season_mid_count --is_add_xfeat_mul_feature

"""
import argparse
import datetime
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(r"C:\Users\yokoi.shingo\GitHub\xfeat")
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

sns.set()
warnings.filterwarnings("ignore")


def add_year_season(df_train, df_test):
    """
    季節の変化としてyear + season / 12 した列入れる
    """
    df_train["year_seaso"] = df_train["year"] + df_train["season"] / 12
    df_test["year_seaso"] = df_test["year"] + df_train["season"] / 12
    return df_train, df_test


def add_discomfort_index(df_train, df_test):
    """
    不快指数の列追加
    不快指数＝0.81×気温+0.01×湿度x(0.99×温度－14.3)+46.3
    https://kenchiku-setsubi.biz/fukaishisu/
    """

    def _discomfort(row):
        return (
            0.81 * row["temp"]
            + 0.01 * row["humidity"] * (0.99 * row["temp"] - 14.3)
            + 46.3
        )

    df_train["discomfort"] = df_train.apply(_discomfort, axis=1)
    df_test["discomfort"] = df_test.apply(_discomfort, axis=1)
    return df_train, df_test


def add_shift_cols(
    df_train,
    df_test,
    cols=["temp", "atemp", "humidity", "windspeed"],
    n_row=24,
    target_col="count",
    is_fillna=False,
):
    """train+test setして差分列追加"""
    df = (
        pd.concat([df_train, df_test], axis=0, ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )

    for c in cols:
        diff = (df[c] - df[c].shift(n_row)) / df[c].shift(n_row)
        diff = diff.apply(lambda x: np.nan if np.isinf(x) else x)  # inf対策
        df[c + "_diff_shift"] = diff
        if is_fillna:
            df[c + "_diff_shift"] = df[c + "_diff_shift"].fillna(0.0)  # 欠損は0にしておく

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_logs(res, cols=["temp", "atemp", "humidity", "windspeed"]):
    """
    Log transform feature list
    https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm/data
    """
    m = res.shape[1]
    for c in cols:
        res = res.assign(newcol=pd.Series(np.log(1.01 + res[c])).values)
        res.columns.values[m] = c + "_log"
        m += 1
    return res


def add_bin_col(df_train, df_test, bin_col="windspeed", n_bin=8, target_col="count"):
    """ビンニングで数値列をカテゴリ型に変換した列追加"""
    df = (
        pd.concat([df_train, df_test], axis=0, ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )

    df[bin_col + "_category"] = pd.cut(df[bin_col], n_bin, labels=False)  # ビニング
    df[bin_col + "_category"] = df[bin_col + "_category"].astype("category")  # カテゴリ型に変換

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_target_mean(df_train, df_test, segment_target="count"):
    """月の目的変数の平均列と標準偏差列を追加"""
    df_train_m_concat = None
    df_test_m_concat = None
    for y in [2011, 2012]:
        for m in range(1, 13):
            train_m_idx = (
                df_train["datetime"]
                .apply(lambda x: x if x.month == m and x.year == y else None)
                .dropna()
                .index
            )
            df_train_m = df_train.loc[train_m_idx]

            test_m_idx = (
                df_test["datetime"]
                .apply(lambda x: x if x.month == m and x.year == y else None)
                .dropna()
                .index
            )
            df_test_m = df_test.loc[test_m_idx]

            m_mean = df_train_m[segment_target].mean()
            m_std = df_train_m[segment_target].std()

            df_train_m[segment_target + "_month_mean"] = m_mean
            df_train_m[segment_target + "_month_std"] = m_std
            df_train_m_concat = (
                df_train_m
                if df_train_m_concat is None
                else pd.concat([df_train_m_concat, df_train_m])
            )

            df_test_m[segment_target + "_month_mean"] = m_mean
            df_test_m[segment_target + "_month_std"] = m_std
            df_test_m_concat = (
                df_test_m
                if df_test_m_concat is None
                else pd.concat([df_test_m_concat, df_test_m])
            )

    return df_train_m_concat, df_test_m_concat


def add_split_datetime(df, col="datetime"):
    """
    datetime列をばらして追加
    後でonehotしやすいように文字型に変換する
    """
    df["year"] = df[col].dt.year.astype(str)
    df["month"] = df[col].dt.month.astype(str)
    df["day"] = df[col].dt.day.astype(str)
    df["dayofweek"] = df[col].dt.dayofweek.astype(str)
    df["hour"] = df[col].dt.hour.astype(str)
    df["minute"] = df[col].dt.minute.astype(str)
    df["second"] = df[col].dt.second.astype(str)
    return df


def add_target_shift(df_train, df_test, n_row=24 * 12, target_col="count"):
    """(n_row/24)日前の目的変数列を追加"""
    df = (
        pd.concat([df_train, df_test], ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )

    shift_targets = []
    for i, x in enumerate(df["datetime"]):
        shift_target = df[df["datetime"] == x - datetime.timedelta(days=(n_row // 24))][
            target_col
        ]

        if len(shift_target) == 0:
            shift_target = None
        else:
            shift_target = shift_target.values[0]

        # print(i, shift_target, x)
        shift_targets.append(shift_target)

    df[target_col + "_shift"] = shift_targets

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_elapsed_day(df_train, df_test, target_col="count"):
    """経過日数の列追加（2012年のほうが全体としてcount増えてるので）"""
    df = (
        pd.concat([df_train, df_test], ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )
    df["elapsed_day"] = df["datetime"].map(lambda x: (x - df.loc[0]["datetime"]).days)

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_am_pm_weather_mean(df_train, df_test):
    """午前午後列と午前午後の平均天気列を追加する"""
    cols = ["year", "month", "day", "am_pm"]

    df_train["year"] = df_train["datetime"].dt.year
    df_train["month"] = df_train["datetime"].dt.month
    df_train["day"] = df_train["datetime"].dt.day
    df_train["am_pm"] = df_train["datetime"].dt.hour.map(lambda x: 0 if x < 13 else 1)
    df_train["weather"] = df_train["weather"].astype("int")

    df_test["year"] = df_test["datetime"].dt.year
    df_test["month"] = df_test["datetime"].dt.month
    df_test["day"] = df_test["datetime"].dt.day
    df_test["am_pm"] = df_test["datetime"].dt.hour.map(lambda x: 0 if x < 13 else 1)
    df_test["weather"] = df_test["weather"].astype("int")

    df_train_am_pm_weather = (
        df_train.groupby(cols)
        .mean()["weather"]
        .reset_index()
        .rename(columns={"weather": "weather_am_pm"})
    )
    df_test_am_pm_weather = (
        df_test.groupby(cols)
        .mean()["weather"]
        .reset_index()
        .rename(columns={"weather": "weather_am_pm"})
    )

    df_train = pd.merge(df_train, df_train_am_pm_weather, on=cols, how="left")
    df_test = pd.merge(df_test, df_test_am_pm_weather, on=cols, how="left")

    df_train["weather"] = df_train["weather"].astype("category")
    df_test["weather"] = df_test["weather"].astype("category")

    return df_train, df_test


def add_time_count_class(df_train, df_test, target_col="count"):
    """count多い時間などをクラス分けした列を追加する"""

    def _set_dayofweek_class(x: datetime):
        """registered, casualが多い曜日を区別する"""
        return (
            "registered_major" if x.dayofweek in range(0, 5) else "casual_major"
        )  # dayofweek=5,6はcasualが多い

    def _set_month_class(x: datetime):
        """count多い月をクラス分けする"""
        return "active" if x.month in range(4, 12) else "inactive"  # 4-11月頃が利用者多い

    def _set_hour_class(x: datetime):
        """count多い時間をクラス分けする"""
        _class = None
        _class = (
            "registered_rush"
            if (x.hour in [8, 17, 18]) and (x.dayofweek in [0, 1, 2, 3, 4])
            else _class
        )  # この時間帯が特に利用者多い
        _class = (
            "registered_semi_rush"
            if (x.hour in [7, 9, 16, 19, 20])
            and (x.dayofweek in [0, 1, 2, 3, 4])
            and _class is None
            else _class
        )  # この時間帯はやや多い
        _class = (
            "casual_peak"
            if (x.hour in [12, 13, 14, 15, 16, 17])
            and (x.dayofweek in [5, 6])
            and _class is None
            else _class
        )  # この時間はやや多い
        _class = (
            "daytime" if x.hour in range(7, 23) and _class is None else _class
        )  # この時間はふつう
        if _class is None:  # 夜中は利用者少ない
            _class = "night"
        return _class

    df = (
        pd.concat([df_train, df_test], ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )
    df["month_class"] = df["datetime"].map(_set_month_class)
    df["dayofweek_class"] = df["datetime"].map(_set_dayofweek_class)
    df["hour_class"] = df["datetime"].map(_set_hour_class)

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_time_count_class_registered(df_train, df_test, target_col="count"):
    """registered多い時間などをクラス分けした列を追加する"""

    def _set_dayofweek_class(x: datetime):
        """registeredが多い曜日を区別する"""
        return 1 if x.dayofweek in range(0, 5) else 0  # dayofweek=5,6はcasualが多い

    def _set_month_class(x: datetime):
        """count多い月をクラス分けする"""
        return "active" if x.month in range(4, 12) else "inactive"  # 4-11月頃が利用者多い

    def _set_hour_class(x: datetime):
        """count多い時間をクラス分けする"""
        _class = None
        _class = (
            "registered_rush"
            if (x.hour in [8, 17, 18]) and (x.dayofweek in [0, 1, 2, 3, 4])
            else _class
        )  # この時間帯が特に利用者多い
        _class = (
            "registered_semi_rush"
            if (x.hour in [7, 9, 16, 19, 20])
            and (x.dayofweek in [0, 1, 2, 3, 4])
            and _class is None
            else _class
        )  # この時間帯はやや多い
        _class = (
            "daytime" if x.hour in range(7, 23) and _class is None else _class
        )  # この時間はふつう
        if _class is None:  # 夜中は利用者少ない
            _class = "night"
        return _class

    df = (
        pd.concat([df_train, df_test], ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )
    df["month_class"] = df["datetime"].map(_set_month_class)
    df["dayofweek_class"] = df["datetime"].map(_set_dayofweek_class)
    df["hour_class"] = df["datetime"].map(_set_hour_class)

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_time_count_class_casual(df_train, df_test, target_col="count"):
    """casual多い時間などをクラス分けした列を追加する"""

    def _set_dayofweek_class(x: datetime):
        """registered, casualが多い曜日を区別する"""
        return 0 if x.dayofweek in range(0, 5) else 1  # dayofweek=5,6はcasualが多い

    def _set_month_class(x: datetime):
        """count多い月をクラス分けする"""
        return "active" if x.month in range(4, 12) else "inactive"  # 4-11月頃が利用者多い

    def _set_hour_class(x: datetime):
        """count多い時間をクラス分けする"""
        _class = None
        _class = (
            "casual_peak"
            if (x.hour in [12, 13, 14, 15, 16, 17])
            and (x.dayofweek in [5, 6])
            and _class is None
            else _class
        )  # この時間はやや多い
        _class = (
            "daytime" if x.hour in range(7, 23) and _class is None else _class
        )  # この時間はふつう
        if _class is None:  # 夜中は利用者少ない
            _class = "night"
        return _class

    df = (
        pd.concat([df_train, df_test], ignore_index=True)
        .sort_values(by=["datetime"])
        .reset_index(drop=True)
    )
    df["month_class"] = df["datetime"].map(_set_month_class)
    df["dayofweek_class"] = df["datetime"].map(_set_dayofweek_class)
    df["hour_class"] = df["datetime"].map(_set_hour_class)

    df_train = df[df[target_col].notnull()].reset_index(drop=True)

    df_test = df[df[target_col].isnull()].reset_index(drop=True)
    for c in ["casual", "registered", "count"]:
        df_test = (
            df_test.drop(c, axis=1) if c in df_test.columns else df_test
        )  # test setで余分な列削除

    return df_train, df_test


def add_registered_casual_mean_count(df_train, df_test):
    """
    各時刻でのregistered, casualの平均値列を追加する
    ※リークさけるために自身のレコードの値は平均すべきでないができてない。。。
    """
    ts_cols = ["year", "month", "dayofweek", "hour"]

    df_train["year"] = df_train["datetime"].dt.year
    df_train["month"] = df_train["datetime"].dt.month
    df_train["dayofweek"] = df_train["datetime"].dt.dayofweek
    df_train["hour"] = df_train["datetime"].dt.hour

    df_test["year"] = df_test["datetime"].dt.year
    df_test["month"] = df_test["datetime"].dt.month
    df_test["dayofweek"] = df_test["datetime"].dt.dayofweek
    df_test["hour"] = df_test["datetime"].dt.hour

    df_registered_group = (
        df_train.groupby(ts_cols)
        .mean()["registered"]
        .reset_index()
        .rename(columns={"registered": "count_registered_ymdh_mean"})
    )
    df_casual_group = (
        df_train.groupby(ts_cols)
        .mean()["casual"]
        .reset_index()
        .rename(columns={"casual": "count_casual_ymdh_mean"})
    )
    df_group = pd.merge(df_registered_group, df_casual_group, on=ts_cols)
    # display(df_group.head(15))

    df_train = pd.merge(
        df_train, df_group, on=ts_cols, how="left"
    )  # .drop(ts_cols, axis=1)
    df_test = pd.merge(
        df_test, df_group, on=ts_cols, how="left"
    )  # .drop(ts_cols, axis=1)

    return df_train, df_test


def scaler_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """train+test setして数値列標準化"""
    from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）

    # train+validation+test set（全データ縦積み）
    dataset = pd.concat(
        objs=[df_train, df_test], axis=0, ignore_index=True
    ).reset_index(drop=True)
    cols = dataset.columns

    # 数値列を標準化
    df_scaler = pd.DataFrame(
        StandardScaler().fit_transform(
            dataset[["temp", "atemp", "humidity", "windspeed"]]
        )
    )
    df_scaler.columns = ["temp", "atemp", "humidity", "windspeed"]

    # 数値列消して、標準化したのをくっつける
    dataset = dataset.drop(["temp", "atemp", "humidity", "windspeed"], axis=1)
    dataset = pd.concat(objs=[dataset, df_scaler], axis=1)
    dataset = dataset[cols]

    # train//test setに戻す
    df_train = dataset[dataset["count"].notnull()].reset_index(drop=True)
    df_train["count"] = df_train["count"].astype("int")
    df_test = dataset[dataset["count"].isnull()].reset_index(drop=True)
    df_test = df_test.drop(["casual", "registered", "count"], axis=1)

    return df_train, df_test


def boxcox_num_col(
    df: pd.DataFrame,
    num_cols=["temp", "atemp", "humidity", "windspeed"],
    lam=0.7,
    skewness_threshold=0.5,
) -> pd.DataFrame:
    """
    skewnessでかい数値列をBoxCox変換
    https://qiita.com/Julio_Vantage/items/7160fcfce871dcbf57e6
    """
    from scipy.special import boxcox1p

    for col in num_cols:
        skewness = df[col].skew()
        if skewness > skewness_threshold:
            print(f"INFO: {col} skewness = {round(skewness, 3)} boxcox exe")
            df[col] = boxcox1p(df[col], lam)
            print("boxcox exe after:", round(df[col].skew(), 3))
        df[col] = df[col].fillna(0)  # 元の値0の場合欠損になるので
    return df


def invboxcox(y, lam=0.7):
    """
    box-cox変換であるscipy.stats.boxcox()の逆変換
    https://ja.coder.work/so/python/275802
    Usage:
        from scipy import stats
        x = np.array([100, 10, 15, 89, 5464, 1, 2, 1])
        lam = -1.0
        y = stats.boxcox(x, lam)
        print(y)
        print(invboxcox(y, lam))
    """
    import numpy as np

    if lam == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lam * y + 1) / lam)


def clipping_cols(
    df,
    num_cols=["temp", "atemp", "humidity", "windspeed"],
    min_clip=0.002,
    max_clip=0.99,
):
    """
    0.2％点以下の値は1％点に、99％点以上の値は99％点にclippingする
    https://github.com/ghmagazine/kagglebook/blob/master/ch03/ch03-01-numerical.py
    """
    p01 = df[num_cols].quantile(min_clip)
    p99 = df[num_cols].quantile(max_clip)
    print(f"### {min_clip * 100}% clip ###\n", p01)
    print(f"### {max_clip * 100}% clip ###\n", p99)
    # 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
    df[num_cols] = df[num_cols].clip(p01, p99, axis=1)
    return df


def delete_outlier_3sigma_df_cols(
    df: pd.DataFrame, cols=["temp", "atemp", "humidity", "windspeed"]
) -> pd.DataFrame:
    """
    データフレームの指定列について、外れ値(3σ以上のデータ)削除
    Usage:
        df = delete_outlier_3sigma_df_cols(df, ['value', 'value2'])
    """
    for col in cols:
        if df[col].dtype.name not in ["object", "category", "bool"]:
            # 数値型の列なら実行
            df = df[
                (abs(df[col] - np.mean(df[col])) / np.std(df[col]) <= 3)
            ].reset_index(drop=True)
    return df


def detect_box_outliers(df: pd.DataFrame, features: list, n=0) -> list:
    """
    箱ひげ図を使った外れ値検出(Tukey法)
    特徴量のデータフレームdfを受け取り、Tukey法に従ってn個以上の外れ値を含むデータフレームのインデックスのリストを返す
    https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook
    Args:
        df: 外れ値持つ特徴量のデータフレーム
        features: 外れ値を検出したいdfの列名リスト
        n: n個の列が共に外れ値となる場合を加味するか
           ->n=1 でlen(features)=2 の場合はfeatures で指定した2列が同時に外れ値の行を外れ値とする
           ->n=2 でlen(features)=3 の場合はfeatures で指定した3列の内いずれかの2列が同時に外れ値の行を外れ値とする

    Returns:
        外れ値と見なされたdfのindexのリスト

    Usage:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = sns.load_dataset('iris')
        #display(df.head())
        print(df.shape)

        # speciesでグループ化した単位でboxplotの外れ値を削除
        for col in ['sepal_length', 'sepal_length', 'sepal_length', 'petal_width']:
            outliers_list = df.groupby('species').apply(lambda x: detect_box_outliers(x, [col]))
            for outlier_indices in outliers_list:
                df = df.drop(outlier_indices, axis=0)

        print(df.shape)
        # boxplot確認
        for col in ['sepal_length', 'sepal_length', 'sepal_length', 'petal_width']:
            sns.boxplot(x='species', y=col, data=df)
            plt.show()
    """
    import numpy as np
    from collections import Counter  # リストの各要素の出現個数をカウントする

    outlier_indices = []

    # 各特徴量(列)ごとに処理
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # 四分位間距離:Interquartile range (IQR)
        IQR = Q3 - Q1

        # 四分位間距離の1.5倍の長さを外れ値の基準幅とする
        outlier_step = 1.5 * IQR

        # 1st quartile - outlier_step より小さい値 もしくは 3rd quartile + outlier_step より大きい値 を外れ値候補として、indexを保持
        outlier_list_col = df[
            (df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)
        ].index

        # 見つかった外れ値候補のindexをリストに追加
        outlier_indices.extend(outlier_list_col)

    # 各特徴量で同じindexで外れ値とみなされた要素があればカウント
    outlier_indices = Counter(outlier_indices)

    multiple_outliers = outlier_indices
    if len(features) > 1:
        # 複数の特徴量で同時に外れ値になってる場合も対応するために、同じindexでn個以上カウントあれば、そのindexを外れ値として確定する
        # featuresが1つだけの場合や、n=0なら関係ない
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def dummy_encode(in_df_train, in_df_test):
    """ ダミー化してonehot化 """
    df_train = in_df_train
    df_test = in_df_test
    categorical_feats = [
        f
        for f in df_train.columns
        if df_train[f].dtype.name in ["object", "category", "bool"]
    ]
    print(categorical_feats)
    for f_ in categorical_feats:
        prefix = f_
        df_train = pd.concat(
            [df_train, pd.get_dummies(df_train[f_], prefix=prefix)], axis=1
        ).drop(f_, axis=1)
        df_test = pd.concat(
            [df_test, pd.get_dummies(df_test[f_], prefix=prefix)], axis=1
        ).drop(f_, axis=1)

    return df_train, df_test


def label2onehot(labels: np.ndarray):
    """
    sklearnでnp.ndarrayのラベルをonehot化
    Args:
        labels:onehot化するラベル名の配列.np.array(['high' 'high' 'low' 'low'])のようなの
    Returns:
        enc:ラベル名を0-nの連番にした配列。np.array([[0] [0] [1] [1]])のようなの
        onehot:ラベル名をonehotにした配列。np.array([[1. 0.] [1. 0.] [0. 1.] [0. 1.]])のようなの
    Usage:
        labels = df['aaa'].values
        enc, onehot = label2onehot(labels)
    """
    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder

    enc = preprocessing.LabelEncoder().fit_transform(labels).reshape(-1, 1)
    onehot = OneHotEncoder().fit_transform(enc).toarray()
    return enc, onehot


def train_val_ts_split(df_train: pd.DataFrame, day_threshold=16):
    """時系列でtrain/validation set分ける"""
    day_train = (
        df_train["datetime"]
        .apply(lambda x: x if x.day in range(1, day_threshold) else None)
        .dropna()
    )  # train setは1から
    day_val = (
        df_train["datetime"]
        .apply(lambda x: x if x.day in range(day_threshold, 20) else None)
        .dropna()
    )  # train setは19日までなので
    return (
        df_train.loc[day_train.index].reset_index(drop=True),
        df_train.loc[day_val.index].reset_index(drop=True),
    )


def log1p_count_cols(df_train, df_test):
    """countに関する列をnp.log1p=1加えてから対数にする"""
    df_train["count"] = df_train["count"].apply(lambda x: np.log1p(x))
    df_train["casual"] = df_train["casual"].apply(lambda x: np.log1p(x))
    df_train["registered"] = df_train["registered"].apply(lambda x: np.log1p(x))
    for c in [
        s
        for s in df_train.columns.to_list()
        if ("count_" in s) or ("_month_mean" in s) or ("_month_std" in s)
    ]:
        print("np.log1p count col:", c)
        df_train[c] = df_train[c].apply(lambda x: np.log1p(x))
        df_test[c] = df_test[c].apply(lambda x: np.log1p(x))
    return df_train, df_test


def add_any_cols(train, test):
    """色々特徴量作成"""
    # 目的変数対数化
    for col in ["casual", "registered", "count"]:
        train["%s_log" % col] = np.log(train[col] + 1)

    for df in [train, test]:
        date = pd.DatetimeIndex(df["datetime"])
        # 年、月、時間、曜日の列追加
        df["year"], df["month"], df["hour"], df["dayofweek"] = (
            date.year,
            date.month,
            date.hour,
            date.dayofweek,
        )

        # intの列と足し合わせるから型変換
        # print(df.info())
        df["season"] = df["season"].astype(int)
        df["workingday"] = df["workingday"].astype(int)

        # 年と季節の中央値計算するための列用意
        df["year_season"] = df["year"] + df["season"] / 10

        # 平日のcasualの時間帯（利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_casual"] = df[["hour", "workingday"]].apply(
            lambda x: int(10 <= x["hour"] <= 19), axis=1
        )

        # 平日のregisteredの時間帯（通勤通学ラッシュの利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_registered"] = df[["hour", "workingday"]].apply(
            lambda x: int(
                (x["workingday"] == 1 and (x["hour"] == 8 or 17 <= x["hour"] <= 18))
                or (x["workingday"] == 0 and 10 <= x["hour"] <= 19)
            ),
            axis=1,
        )

    # 年と季節ごとの利用者数の中央値列追加
    by_season = train.groupby("year_season")[["count"]].median()
    by_season.columns = ["count_season"]
    train = train.join(by_season, on="year_season")
    test = test.join(by_season, on="year_season")

    return train, test


def add_season_mid_count(train, test):
    """四季ごとのcount数の中央値列や各カテゴリごとでのcount数の中央値列追加"""
    by_season = train.groupby("season")[["count"]].median()
    by_season.columns = ["count_season_mid"]
    train = train.join(by_season, on="season")
    test = test.join(by_season, on="season")

    cat_cols = ["season", "holiday", "workingday", "weather"]
    by_ = train.groupby(cat_cols)[["count"]].median()
    by_.columns = ["count_cat_cols_mid"]
    train = train.join(by_, on=cat_cols)
    test = test.join(by_, on=cat_cols)

    return train, test


def add_any_cols_v2(train, test):
    """色々特徴量作成_v2"""
    # 目的変数対数化
    for col in ["casual", "registered", "count"]:
        train["%s_log" % col] = np.log(train[col] + 1)

    for df in [train, test]:
        date = pd.DatetimeIndex(df["datetime"])
        # 年、月、時間、曜日の列追加
        df["year"], df["month"], df["hour"], df["dayofweek"] = (
            date.year,
            date.month,
            date.hour,
            date.dayofweek,
        )

        # intの列と足し合わせるから型変換
        # print(df.info())
        df["season"] = df["season"].astype(int)
        df["workingday"] = df["workingday"].astype(int)
        df["holiday"] = df["holiday"].astype(int)

        # 年と季節の中央値計算するための列用意
        df["year_season"] = df["year"] + df["season"] / 10

        # casualの時間帯（利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_casual"] = df[["hour", "workingday"]].apply(
            lambda x: int(10 <= x["hour"] <= 19), axis=1
        )

        # casualのworkingday,holidayの時間帯（利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_holiday_casual_v1"] = df[
            ["hour", "workingday", "holiday"]
        ].apply(
            lambda x: int(
                (x["workingday"] == 0)
                and (x["holiday"] == 1)
                and (10 <= x["hour"] <= 19)
            ),
            axis=1,
        )

        # casualのworkingday,holidayの時間帯（利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_holiday_casual_v2"] = df[
            ["hour", "workingday", "holiday"]
        ].apply(
            lambda x: int(
                (x["workingday"] == 0)
                and (x["holiday"] == 0)
                and (10 <= x["hour"] <= 19)
            ),
            axis=1,
        )

        # 平日のregisteredの時間帯（通勤通学ラッシュの利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_registered_rush"] = df[["hour", "workingday"]].apply(
            lambda x: int(
                (x["workingday"] == 1) and (x["hour"] == 8 or 17 <= x["hour"] <= 18)
            ),
            axis=1,
        )

        # 平日のregisteredの時間帯（通勤通学セミラッシュの利用者数大体同じ時間帯）に0/1フラグつける
        df["hour_workingday_registered_semi"] = df[["hour", "workingday"]].apply(
            lambda x: int(
                (x["workingday"] == 1) and (x["hour"] == 7 or x["hour"] == 19)
            ),
            axis=1,
        )

        # registeredの時間帯に0/1フラグつける
        df["hour_workingday_registered"] = df[["hour", "workingday"]].apply(
            lambda x: int((x["workingday"] == 0) and (10 <= x["hour"] <= 19)), axis=1,
        )

    # 年と季節ごとの利用者数の中央値列追加
    by_season = train.groupby("year_season")[["count"]].median()
    by_season.columns = ["count_season"]
    train = train.join(by_season, on="year_season")
    test = test.join(by_season, on="year_season")

    return train, test


def target_encoding(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    cat_cols: list,
    target: str,
    n_splits=4,
    random_state=71,
):
    """
    target encoding: 目的変数を用いてカテゴリ変数を数値に変換する（目的変数の平均値列を追加する）
    参考:https://github.com/ghmagazine/kagglebook/blob/master/ch03/ch03-02-categorical.py
    Usage:
        import pandas as pd
        pd.set_option('display.max_columns', None)
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = sns.load_dataset('titanic')
        #display(df)

        num_cols = ["age", "fare"]
        cat_cols = list(set(df.columns.to_list()) - set(num_cols))
        enc_cols = list(set(cat_cols) - set(["survived", "pclass", "sibsp", "parch"]))
        df[enc_cols] = df[enc_cols].apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
        #display(df)

        target = "survived"
        cat_cols.remove(target)
        (X_train, X_test, y_train, y_test) = train_test_split(df.drop(target, axis=1), df[target], test_size=0.3, random_state=71)
        #display(X_train)
        train_x, test_x = target_encoding(X_train, y_train, X_test, cat_cols, target)
        display(train_x)
        display(test_x)
    """
    from sklearn.model_selection import KFold

    # 変数をループしてtarget encoding
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train_x[c], target: train_y})
        target_mean = data_tmp.groupby(c)[target].mean()
        # テストデータのカテゴリを置換
        test_x[c + "_te"] = test_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for idx_1, idx_2 in kf.split(train_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)[target].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        train_x[c + "_te"] = tmp

    return train_x, test_x


def cv_target_encoding(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    cat_cols: list,
    target: str,
    split_type="KFold",
    n_splits=4,
    random_state=71,
):
    """
    クロスバリデーションのfoldごとのtarget encoding
    参考:https://github.com/ghmagazine/kagglebook/blob/master/ch03/ch03-02-categorical.py
    Usage:
        import pandas as pd
        pd.set_option('display.max_columns', None)
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = sns.load_dataset('titanic')
        #display(df)

        num_cols = ["age", "fare"]
        cat_cols = list(set(df.columns.to_list()) - set(num_cols))
        enc_cols = list(set(cat_cols) - set(["survived", "pclass", "sibsp", "parch"]))
        df[enc_cols] = df[enc_cols].apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
        #display(df)

        target = "survived"
        cat_cols.remove(target)
        (X_train, X_test, y_train, y_test) = train_test_split(df.drop(target, axis=1), df[target], test_size=0.3, random_state=71)
        #display(X_train)
        tr_dfs, va_dfs = preprocess.cv_target_encoding(X_train, y_train, X_test, cat_cols, target, split_type="StratifiedKFold")
        for tr_df, va_df in zip(tr_dfs, va_dfs):
            display(tr_df)
            display(va_df)
    """
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

    # クロスバリデーションのfoldごとにtarget encodingをやり直す
    if split_type == "StratifiedKFold":
        # StratifiedKFold:ラベルの比率が揃うようにtrainデータとtestデータを分ける
        sms = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    elif split_type == "TimeSeriesSplit":
        # 時系列クロスバリデーション
        # データは必ず時間の昇順に並べておく必要がある！！！
        # TimeSeriesSplit はtrain を先頭から順番にデータ切っていき、後ろのレコードをvalidation するだけなので！！！
        sms = TimeSeriesSplit(n_splits=n_splits)
    else:
        # K-分割交差検証
        sms = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    tr_dfs, va_dfs = [], []
    for i, (tr_idx, va_idx) in enumerate(sms.split(train_x, train_y)):

        # 学習データからバリデーションデータを分ける
        tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # target encoding
        tr_x, va_x = target_encoding(tr_x, tr_y, va_x, cat_cols, target)
        tr_dfs.append(pd.concat([tr_x, tr_y], axis=1))
        va_dfs.append(pd.concat([va_x, va_y], axis=1))

        # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく
        # display(tr_x)
        # display(tr_y)

    return tr_dfs, va_dfs


def add_next_holiday(df):
    """
    次の日holidayならフラグつける
    次の日休みなら飲みに行くから利用者減る気がするので
    """
    df["date"] = df["datetime"].dt.date

    # workingday=1のデータ取得
    work_idxs = df[df["workingday"] == 1].index.to_list()
    work_df = df.loc[work_idxs][["workingday", "date"]]
    work_dates = np.unique(work_df["date"].values)

    # workingday=1の翌日
    work_next_d = [d + datetime.timedelta(days=1) for d in work_dates]

    # workingday=1の翌日の内、holiday=1の日にち取得
    next_holiday_d = []
    for d in work_next_d:
        _df = df[df["date"] == d]
        next_holiday_idx = _df[_df["holiday"] == 1].index.to_list()
        if next_holiday_idx != []:
            next_holiday_d.append(d - datetime.timedelta(days=1))

    # workingday=1の翌日の内、holiday=1の日にちのレコードにフラグたてる
    df["next_holiday"] = 0
    for d in next_holiday_d:
        for i in df[df["date"] == d].index.to_list():
            df.loc[i, "next_holiday"] = 1

    return df.drop(["date"], axis=1)


def add_xfeat_mul_feature(df, cols):
    """
    xfeatで特徴量掛け算した列追加
    参考: https://megane-man666.hatenablog.com/entry/xfeat
    """
    encoder = Pipeline(
        [
            ArithmeticCombinations(
                input_cols=cols,
                drop_origin=False,
                operator="*",
                r=2,
                output_suffix="_mul",
            ),
        ]
    )
    return encoder.fit_transform(df)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default="../data/preprocessed")
    ap.add_argument("-s_t", "--segment_target", type=str, default="count")
    ap.add_argument(
        "-i", "--input_dir", type=str, default="../data/orig/bike-sharing-demand"
    )
    ap.add_argument(
        "--is_boxcox", action="store_const", const=True, default=False, help="BoxCox変換"
    )
    ap.add_argument(
        "--is_weather_del_3sigma",
        action="store_const",
        const=True,
        default=False,
        help="weather 列の外れ値(3σ以上のデータ)削除",
    )
    ap.add_argument(
        "--is_box_outlier",
        action="store_const",
        const=True,
        default=False,
        help="箱ひげ図を使ってtrain_setの数値列の外れ値を削除",
    )
    ap.add_argument(
        "--is_num_scaler",
        action="store_const",
        const=True,
        default=False,
        help="数値列を標準化",
    )
    ap.add_argument(
        "-d_t",
        "--day_threshold",
        type=int,
        default=None,
        help="時系列でtrain/validation set分ける。分ける境目の日を指定する。16ならtrainが1-16日、validationが17-20になる",
    )
    ap.add_argument(
        "-s_r",
        "--shift_row",
        type=int,
        default=None,
        help="数値列の差分列追加。shiftする行数指定する。24なら1日差分",
    )  # 24
    ap.add_argument(
        "--is_fillna_shift",
        action="store_const",
        const=True,
        default=False,
        help="数値列の差分列の欠損うめる",
    )
    ap.add_argument(
        "--is_add_target_shift",
        action="store_const",
        const=True,
        default=False,
        help="12日前の目的変数列を追加",
    )
    ap.add_argument(
        "--is_add_logs",
        action="store_const",
        const=True,
        default=False,
        help="数値列を対数化した列追加",
    )
    ap.add_argument(
        "--is_dummy",
        action="store_const",
        const=True,
        default=False,
        help="文字列/カテゴリ列/bool列をダミー変数化",
    )
    ap.add_argument(
        "--is_add_split_datetime",
        action="store_const",
        const=True,
        default=False,
        help="datetime列を年/月/日/時/分/秒の列にばらして追加",
    )
    ap.add_argument(
        "--is_add_elapsed_day",
        action="store_const",
        const=True,
        default=False,
        help="経過日数列追加",
    )
    ap.add_argument(
        "--is_add_target_mean",
        action="store_const",
        const=True,
        default=False,
        help="同月の目的変数の平均列と標準偏差列を追加",
    )
    ap.add_argument(
        "--is_add_time_count_class",
        action="store_const",
        const=True,
        default=False,
        help="count多い時間などをクラス分けした列を追加",
    )
    ap.add_argument(
        "--is_labelencode_count_class",
        action="store_const",
        const=True,
        default=False,
        help="count多い時間などをクラス分けした列を0,1,2…とラベルエンコードするか",
    )
    ap.add_argument(
        "--is_add_registered_casual_mean_count",
        action="store_const",
        const=True,
        default=False,
        help="各時刻でのregistered, casualの平均値列を追加する",
    )
    ap.add_argument(
        "--is_add_discomfort_index",
        action="store_const",
        const=True,
        default=False,
        help="不快指数列を追加する",
    )
    ap.add_argument(
        "--is_add_am_pm_weather_mean",
        action="store_const",
        const=True,
        default=False,
        help="午前午後列と午前午後の平均天気列を追加する",
    )
    ap.add_argument(
        "--is_log1p_count_cols",
        action="store_const",
        const=True,
        default=False,
        help="countに関する列をnp.log1pする",
    )
    ap.add_argument(
        "--is_add_any_cols",
        action="store_const",
        const=True,
        default=False,
        help="色々特徴量作成",
    )
    ap.add_argument(
        "--is_add_any_cols_v2",
        action="store_const",
        const=True,
        default=False,
        help="色々特徴量作成_v2",
    )
    ap.add_argument(
        "-te_split_type",
        "--cv_target_encoding_split_type",
        type=str,
        default=None,
        help="cvのターゲットエンコーディングの分け方。StratifiedKFold, KFold, TimeSeriesSplit のどれか指定する",
    )
    ap.add_argument(
        "--is_add_next_holiday",
        action="store_const",
        const=True,
        default=False,
        help="次の日holidayかどうかの列を追加する",
    )
    ap.add_argument(
        "--is_add_season_mid_count",
        action="store_const",
        const=True,
        default=False,
        help="四季ごとのcount数の中央値列追加する",
    )
    ap.add_argument(
        "--is_add_xfeat_mul_feature",
        action="store_const",
        const=True,
        default=False,
        help="xfeatで特徴量掛け算した列追加",
    )
    ap.add_argument(
        "--bin_cols", type=str, nargs="*", default=None, help="指定した列をビン化した列追加"
    )
    ap.add_argument(
        "--clip_cols", type=str, nargs="*", default=None, help="指定した列をクリッピング"
    )
    return vars(ap.parse_args())


if __name__ == "__main__":
    matplotlib.use("Agg")

    args = get_args()
    os.makedirs(args["output_dir"], exist_ok=True)

    df_train = pd.read_csv(
        os.path.join(args["input_dir"], "train.csv"),
        dtype={
            "season": "category",
            "holiday": "category",
            "workingday": "category",
            "weather": "category",
        },
        parse_dates=["datetime"],
    )
    df_test = pd.read_csv(
        os.path.join(args["input_dir"], "test.csv"),
        dtype={
            "season": "category",
            "holiday": "category",
            "workingday": "category",
            "weather": "category",
        },
        parse_dates=["datetime"],
    )

    if args["is_boxcox"]:
        print("--- BoxCox変換 ---")
        df_train = boxcox_num_col(df_train)
        df_test = boxcox_num_col(df_test)

    if args["is_num_scaler"]:
        print("--- 数値列を標準化 ---")
        df_train, df_test = scaler_dataset(df_train, df_test)

    if args["is_weather_del_3sigma"]:
        print("--- weather 列の外れ値(3σ以上のデータ)削除 ---")
        print("df_train.shape:", df_train.shape)
        df_train = delete_outlier_3sigma_df_cols(df_train, ["weather"])
        print("df_train.shape:", df_train.shape)

    if args["is_box_outlier"]:
        print("--- 箱ひげ図を使ってtrain_setの数値列の外れ値を削除 ---")
        print("df_train.shape:", df_train.shape)
        for col in ["season", "holiday", "workingday", "weather"]:
            outliers_list = df_train.groupby(col).apply(
                lambda x: detect_box_outliers(x, ["count"], n=0)
            )
        for outlier_indices in outliers_list:
            df_train = df_train.drop(outlier_indices, axis=0)
        print("df_train.shape del after:", df_train.shape)

    if args["shift_row"] is not None:
        print("--- 数値列の差分列追加 ---")
        df_train, df_test = add_shift_cols(
            df_train,
            df_test,
            n_row=args["shift_row"],
            cols=["temp", "atemp", "humidity", "windspeed"],
            is_fillna=args["is_fillna_shift"],
        )

    if args["is_add_logs"]:
        print("--- 数値列を対数化した列追加 ---")
        df_train = add_logs(df_train, cols=["temp", "atemp", "humidity", "windspeed"])
        df_test = add_logs(df_test, cols=["temp", "atemp", "humidity", "windspeed"])

    if args["is_add_any_cols"]:
        print("--- 色々特徴量作成 ---")
        df_train, df_test = add_any_cols(df_train, df_test)

    if args["is_add_any_cols_v2"]:
        print("--- 色々特徴量作成_v2 ---")
        df_train, df_test = add_any_cols_v2(df_train, df_test)

    if args["is_add_season_mid_count"]:
        print("--- 四季ごとのcount数の中央値列追加 ---")
        df_train, df_test = add_season_mid_count(df_train, df_test)

    if args["is_add_elapsed_day"]:
        print("--- 経過日数列追加 ---")
        df_train, df_test = add_elapsed_day(df_train, df_test)

    if args["is_add_am_pm_weather_mean"]:
        print("--- 午前午後列と午前午後の平均天気列を追加する ---")
        df_train, df_test = add_am_pm_weather_mean(df_train, df_test)

    if args["is_add_time_count_class"]:
        if args["segment_target"] == "count":
            print("--- count多い時間などをクラス分けした列を追加 ---")
            df_train, df_test = add_time_count_class(df_train, df_test)

        elif args["segment_target"] == "registered":
            print("--- registered多い時間などをクラス分けした列を追加 ---")
            df_train, df_test = add_time_count_class_registered(df_train, df_test)

        elif args["segment_target"] == "casual":
            print("--- casual多い時間などをクラス分けした列を追加 ---")
            df_train, df_test = add_time_count_class_casual(df_train, df_test)

        if args["is_labelencode_count_class"]:
            print("--- count多い時間などをクラス分けした列を0,1,2…とラベルエンコードするか ---")
            for c in ["month_class", "dayofweek_class", "hour_class"]:
                enc, _ = label2onehot(df_train[c].values)
                df_train[c] = enc.reshape(-1)
                enc, _ = label2onehot(df_test[c].values)
                df_test[c] = enc.reshape(-1)

    if args["is_add_registered_casual_mean_count"]:
        print("--- 各時刻でのregistered, casualの平均値列を追加する ---")
        df_train, df_test = add_registered_casual_mean_count(df_train, df_test)

    if args["is_add_discomfort_index"]:
        print("--- 不快指数列を追加する ---")
        df_train, df_test = add_discomfort_index(df_train, df_test)

    if args["is_add_split_datetime"]:
        print("--- datetime列を年/月/日/時/分/秒の列にばらして追加 ---")
        df_train = add_split_datetime(df_train)
        df_test = add_split_datetime(df_test)

    if args["clip_cols"] is not None:
        print(f"--- {args['clip_cols']}列をクリッピング ---")
        df_train = clipping_cols(df_train, num_cols=args["clip_cols"])

    # print(args['bin_cols'])
    if args["bin_cols"] is not None:
        print(f"--- {args['bin_cols']}列をビン化した列追加 ---")
        for c in args["bin_cols"]:
            df_train, df_test = add_bin_col(
                df_train, df_test, bin_col=c, n_bin=8, target_col="count"
            )

    if args["is_add_target_mean"]:
        print("--- 同月の目的変数の平均列と標準偏差列を追加 ---")
        df_train, df_test = add_target_mean(
            df_train, df_test, segment_target=args["segment_target"]
        )

    if args["is_add_target_shift"]:
        print("--- 12日前の目的変数列を追加 ---")
        df_train, df_test = add_target_shift(df_train, df_test, n_row=24 * 12)

    if args["is_add_target_shift"]:
        print("--- 12日前の目的変数列を追加 ---")
        df_train, df_test = add_target_shift(df_train, df_test, n_row=24 * 12)

    if args["is_add_next_holiday"]:
        print("--- 次の日holidayかどうかの列を追加する ---")
        df_train = add_next_holiday(df_train)
        df_test = add_next_holiday(df_test)

    if args["is_add_xfeat_mul_feature"]:
        print("--- xfeatで特徴量掛け算した列追加 ---")
        # cols = df_test.columns.to_list()
        # cols.remove("datetime")
        cols = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "dayofweek",
            "hour",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "count_season_mid",
            "count_cat_cols_mid",
            "am_pm",
            "weather_am_pm",
        ]
        df_train["weather"] = df_train["weather"].astype(int)
        df_test["weather"] = df_test["weather"].astype(int)
        df_train = add_xfeat_mul_feature(df_train, cols)
        df_test = add_xfeat_mul_feature(df_test, cols)

    if args["is_dummy"]:
        print("--- 文字列/カテゴリ列/bool列をダミー変数化 ---")
        df_train, df_test = dummy_encode(df_train, df_test)
        print("Shape train: %s, test: %s" % (df_train.shape, df_test.shape))

    if args["day_threshold"] is not None:
        print("--- 時系列でtrain/validation set分ける ---")
        df_train, df_val = train_val_ts_split(
            df_train, day_threshold=args["day_threshold"]
        )

        df_val = df_val.sort_values(by=["datetime"])
        df_val.to_csv(os.path.join(args["output_dir"], "validation.csv"), index=False)

    # csv出力
    df_train = df_train.sort_values(by=["datetime"])
    df_test = df_test.sort_values(by=["datetime"])
    df_train.to_csv(os.path.join(args["output_dir"], "train.csv"), index=False)
    df_test.to_csv(os.path.join(args["output_dir"], "test.csv"), index=False)

    if args["cv_target_encoding_split_type"] is not None:
        print("--- ターゲットエンコーディング列追加する ---")
        for df in [df_train, df_test]:
            df["year"] = df["datetime"].dt.year.astype(str)
            df["month"] = df["datetime"].dt.month.astype(str)
            df["day"] = df["datetime"].dt.day.astype(str)
            df["dayofweek"] = df["datetime"].dt.dayofweek.astype(str)
            df["hour"] = df["datetime"].dt.hour.astype(str)

        target = "count"
        cat_cols = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
        ]

        train_x = df_train.copy()
        train_x = train_x.drop([target], axis=1)
        train_y = df_train[target]
        test_x = df_test.copy()
        # target_encoding
        train_x, test_x = target_encoding(train_x, train_y, test_x, cat_cols, target)
        _df_train = pd.concat([train_x, train_y], axis=1)
        _df_test = test_x
        # csv出力
        _df_train = _df_train.sort_values(by=["datetime"])
        _df_test = _df_test.sort_values(by=["datetime"])
        _df_train.to_csv(os.path.join(args["output_dir"], "train.csv"), index=False)
        _df_test.to_csv(os.path.join(args["output_dir"], "test.csv"), index=False)

        train_x = df_train.copy()
        train_x = train_x.drop([target], axis=1)
        train_y = df_train[target]
        test_x = df_test.copy()
        # cv分けてtarget_encoding
        tr_dfs, va_dfs = cv_target_encoding(
            train_x,
            train_y,
            test_x,
            cat_cols,
            target,
            split_type=args["cv_target_encoding_split_type"],
        )
        # csv出力
        i = 0
        for tr_df, va_df in zip(tr_dfs, va_dfs):
            tr_df = tr_df.sort_values(by=["datetime"])
            va_df = va_df.sort_values(by=["datetime"])
            tr_df.to_csv(
                os.path.join(args["output_dir"], f"train_cv-{i}.csv"), index=False
            )
            va_df.to_csv(
                os.path.join(args["output_dir"], f"val_cv-{i}.csv"), index=False
            )
            i += 1
