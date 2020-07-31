import argparse
import datetime
import logging
import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

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


class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def ts_plot_trainset_true_pred(
        cls, pred_y, df_train, path_name, target_col="count", is_Agg=True,
    ):
        """横軸時系列でtrain setの正解と予測plot"""
        if is_Agg:
            import matplotlib

            matplotlib.use("Agg")

        # 正解と予測の結果csvも出す
        df_train["pred_y"] = pred_y
        df_train.to_csv(f"{path_name}.csv", index=False)

        plt.figure(figsize=(20, 10))
        plt.plot(df_train.index, df_train[target_col], linestyle="dashed", label="true")
        plt.plot(df_train.index, df_train["pred_y"], linestyle="dotted", label="pred")
        plt.xlabel("index")
        plt.ylabel(target_col)
        plt.legend()
        plt.savefig(
            f"{path_name}.png", bbox_inches="tight", pad_inches=0,
        )
        plt.show()
        plt.clf()
        plt.close()
        return


class Logger:
    def __init__(self):
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler()

        _pwd = pathlib.Path.cwd()  # カレントディレクトリ
        file_general_handler = logging.FileHandler(os.path.join(_pwd, "general.log"))
        file_result_handler = logging.FileHandler(os.path.join(_pwd, "result.log"))

        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score{i}"] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(self, dic):
        return "\t".join(["{}:{}".format(key, value) for key, value in dic.items()])


class Submission:
    """bike sharingコンペ用"""

    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv("../input/sampleSubmission.csv")
        pred = Util.load(f"../model/pred/{run_name}-test.pkl")
        for i in range(pred.shape[1]):
            submission[f"Class_{i + 1}"] = pred[:, i]
        submission.to_csv(f"../submission/{run_name}.csv", index=False)

    @classmethod
    def output_submit(
        cls,
        pkl_path: str,
        out_submit_csv: str,
        transform=None,
        orig_submit_csv=r"C:\Users\yokoi.shingo\my_task\Bike_Shareing\data\orig_InClass\bikesharing-for-education_col_edit\sample_submission.csv",
        # orig_submit_csv=r"C:\Users\yokoi.shingo\my_task\Bike_Shareing\data\orig\bike-sharing-demand\sampleSubmission.csv",
    ):
        """
        submission.csv作成
        pkl_pathは予測結果のファイルパス
        """
        df_sub = pd.read_csv(orig_submit_csv)
        predictions = Util.load(pkl_path)
        print("predictions.shape:", predictions.shape)

        if transform is not None:
            # 対数化を戻す
            print("INFO:transform", transform)
            predictions = np.array([transform(x) for x in predictions])

        df_sub["count"] = predictions
        # df_sub['count'] = round(df_sub['count'], 0)  # 四捨五入っぽくする

        # inf対策
        df_sub["count"] = df_sub["count"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # df_sub['count'] = df_sub['count'].astype('int')  # intにする
        df_sub.loc[df_sub["count"] < 0, "count"] = 0  # マイナスの値は0にする

        # InClassのデータでは列名cntにしないとだめ
        df_sub = df_sub.rename(columns={"count": "cnt"})

        df_sub.to_csv(out_submit_csv, index=False)

        return df_sub

    @classmethod
    def blend_submit_csv(
        cls, csvs: list, ratios: list, output_dir: str, out_csv_name=None
    ):
        """submit.csvの目的変数をブレンドする（ratioを掛けて混ぜる）"""
        dfs = [pd.read_csv(c) for c in csvs]
        pred = None
        for r, df in zip(ratios, dfs):
            # _p = r * df["count"]
            _p = r * df["cnt"]  # InClassのデータでは列名cntにしないとだめ
            pred = _p if pred is None else pred + _p
        print(pred)

        # submission = pd.DataFrame({"datetime": dfs[0]["datetime"], "count": pred})
        # InClassのデータでは列名cntにしないとだめ
        submission = pd.DataFrame({"datetime": dfs[0]["datetime"], "cnt": pred})

        if out_csv_name is None:
            submission.to_csv(
                os.path.join(output_dir, "blend_submission.csv"), index=False
            )
        else:
            submission.to_csv(
                os.path.join(output_dir, f"{out_csv_name}.csv"), index=False
            )


class Xfeat:
    @classmethod
    def label_encode_xfeat(df):
        """
        xfeatでobject型の列すべてラベルエンコディング
        """
        df_cate = Pipeline(
            [SelectCategorical(), LabelEncoder(output_suffix=""),]
        ).fit_transform(df)
        df_num = Pipeline([SelectNumerical(),]).fit_transform(df)
        df = pd.concat([df_cate, df_num], axis=1)
        return df

    @classmethod
    def feature_engineering(cls, df):
        """
        xfeatで特徴量エンジニアリング
        参考: https://megane-man666.hatenablog.com/entry/xfeat
        """
        cols = df.columns.tolist()

        encoder = Pipeline(
            [
                ArithmeticCombinations(
                    input_cols=cols,
                    drop_origin=False,
                    operator="+",
                    r=2,
                    output_suffix="_plus",
                ),
                ArithmeticCombinations(
                    input_cols=cols,
                    drop_origin=False,
                    operator="*",
                    r=2,
                    output_suffix="_mul",
                ),
                ArithmeticCombinations(
                    input_cols=cols,
                    drop_origin=False,
                    operator="-",
                    r=2,
                    output_suffix="_minus",
                ),
                ArithmeticCombinations(
                    input_cols=cols,
                    drop_origin=False,
                    operator="+",
                    r=3,
                    output_suffix="_plus",
                ),
            ]
        )
        return encoder.fit_transform(df)

    @classmethod
    def get_feature_explorer(
        cls,
        input_cols,
        target_col,
        lgbm_params=None,
        fit_params=None,
        objective="regression",
        metric="rmse",
    ):
        """
        xfeatの特徴量選択用lgmモデルを返す(threshold_rangeのbestな値を探索するための)
        submit投げるlgmの学習に使うパラメータではないので注意
        """
        if lgbm_params is None:
            lgbm_params = {
                "objective": objective,
                "metric": metric,
                "learning_rate": 0.1,
                "verbosity": -1,
            }
        if fit_params is None:
            fit_params = {
                # "num_boost_round": 100,
                "num_boost_round": 500,
            }
        selector = GBDTFeatureExplorer(
            input_cols=input_cols,
            target_col=target_col,
            fit_once=True,
            threshold_range=(0.6, 1.0),  # 採用する特徴量のfeature_importanceの閾値。ベストの閾値を探る
            # threshold_range=(0.3, 1.0),  # 採用する特徴量のfeature_importanceの閾値。ベストの閾値を探る
            lgbm_params=lgbm_params,
            lgbm_fit_kwargs=fit_params,
        )
        return selector

    @classmethod
    def get_feature_selector(
        cls,
        input_cols,
        target_col,
        threshold,
        lgbm_params=None,
        fit_params=None,
        objective="regression",
        metric="rmse",
    ):
        """
        xfeatの特徴量選択用lgmモデルを返す
        submit投げるlgmの学習に使うパラメータではないので注意
        """
        if lgbm_params is None:
            lgbm_params = {
                "objective": objective,
                "metric": metric,
                "learning_rate": 0.1,
                "verbosity": -1,
            }
        if fit_params is None:
            fit_params = {
                "num_boost_round": 100,
            }
        selector = GBDTFeatureSelector(
            input_cols=input_cols,
            target_col=target_col,
            fit_once=True,
            threshold=threshold,
            lgbm_params=lgbm_params,
            lgbm_fit_kwargs=fit_params,
        )
        return selector


if __name__ == "__main__":
    # submitファイルブレンドをコマンドラインで実行
    # python util.py -o ../model/blend -c1 ../model/v5_casual/lgb-submission.csv -c2 ../model/v5_registered/lgb-submission.csv
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o", "--output_dir", type=str, help="blend results output dir path."
    )
    ap.add_argument("-c1", "--csv1", type=str, help="blend submit csv1 path.")
    ap.add_argument("-c2", "--csv2", type=str, help="blend submit csv2 path.")
    ap.add_argument(
        "-r1", "--ratio1", type=float, default=1.0, help="blend submit csv1 ratio."
    )
    ap.add_argument(
        "-r2", "--ratio2", type=float, default=1.0, help="blend submit csv2 ratio."
    )
    ap.add_argument(
        "-o_c_n", "--out_csv_name", type=str, default=None, help="output csv name."
    )
    args = vars(ap.parse_args())
    Submission.blend_submit_csv(
        [args["csv1"], args["csv2"]],
        [args["ratio1"], args["ratio2"]],
        args["output_dir"],
        out_csv_name=args["out_csv_name"],
    )
