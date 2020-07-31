import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

from model import Model
import util
from util import Logger, Util

logger = Logger()


def rmsle(y_true, y_pred):
    """
    対数平方平均二乗誤差（RMSLE: Root Mean Squared Logarithmic Error）
    https://rautaku.hatenablog.com/entry/2017/12/22/195649
    """
    assert len(y_pred) == len(y_true)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


class Runner:
    def __init__(
        self,
        run_name: str,
        model_cls: Callable[[str, dict], Model],
        features: List[str],
        params: dict,
        target: str,
        train_csv: str,
        test_csv: str,
        n_fold=4,
        output_dir="../model/pred",
        split_type="Kfold",  # どうcv分けるか
        val_eval="rmse",  # 各cvのvalidation setの評価指標。early stoppingで使うtrain setの評価指標はparamsにある。別々に指定するので注意
        input_cv_dir=None,  # cvのvalidation setのファイル置き場。指定したらvalidation setファイルからロード
        del_features=None,  # 削除する特徴量名
    ):
        """コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = n_fold
        self.target = target
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.output_dir = output_dir
        self.split_type = split_type
        self.val_eval = val_eval
        self.input_cv_dir = input_cv_dir
        self.del_features = del_features

    def train_fold(
        self, i_fold: Union[int, str]
    ) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != "all"
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # 学習データ・バリデーションデータをセットする
            if self.input_cv_dir is None:
                # train.csvからcvのfoldに分ける
                tr_idx, va_idx = self.load_index_fold(i_fold)
                tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
                va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
            else:
                print("INFO: load cv csv")
                # あらかじめcvのfoldに分けたtrain,valをロード
                _train_csv = os.path.join(self.input_cv_dir, f"train_cv-{i_fold}.csv")
                _val_csv = os.path.join(self.input_cv_dir, f"val_cv-{i_fold}.csv")
                tr_x, va_x = self.load_x_cv_train_val(_train_csv, _val_csv)
                tr_y, va_y = self.load_y_cv_train_val(_train_csv, _val_csv)
                # 適当な値渡しておく
                va_idx = list(range(va_x.shape[0]))

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)
            # print(model.params)

            # バリデーションデータへの予測・評価を行う
            va_pred = model.predict(va_x)
            score = None
            if self.val_eval == "log_loss":
                score = log_loss(
                    va_y, va_pred, eps=1e-15, normalize=True
                )  # 分類のlog_loss=cross_entropy(0~1の予測値を入力してモデルの性能を測る)
            elif self.val_eval == "mae":
                score = mean_absolute_error(va_y, va_pred)  # 回帰のMAE(絶対値の平均)
            elif self.val_eval == "mse":
                score = mean_squared_error(va_y, va_pred)  # 回帰のMSE(絶対値の2乗平均)
            elif self.val_eval == "rmse":
                score = np.sqrt(mean_squared_error(va_y, va_pred))  # 回帰のRMSE(MSEのルート)
            elif self.val_eval == "rmsle":
                score = rmsle(va_y, va_pred)  # 回帰のRMSLE(+1してからlogとった値のRMSE)

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self):
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f"{self.run_name} - start training cv")

        scores = []
        va_idxes = []
        preds = []
        models = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f"{self.run_name} fold {i_fold} - start training")
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f"{self.run_name} fold {i_fold} - end training - score {score}")

            # モデルを保存する
            model.save_model(model_dir=self.output_dir)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)
            models.append(model)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f"{self.run_name} - end training cv - score {np.mean(scores)}")

        # 予測結果の保存
        Util.dump(preds, os.path.join(self.output_dir, f"{self.run_name}-train.pkl"))

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

        # 一応tsvファイルでもモデルのパラメと評価結果の値(rmseとかの値)残しておく
        # モデルのパラメ
        pd.DataFrame.from_dict(models[-1].params, orient="index").to_csv(
            os.path.join(self.output_dir, f"{self.run_name}-cv_param.tsv"), sep="\t",
        )
        # 評価結果の値
        pd.DataFrame(
            {
                "cv_fold": list(range(self.n_fold)),
                f"valid_mean_score({self.val_eval})": scores,
            }
        ).to_csv(
            os.path.join(self.output_dir, f"{self.run_name}-cv_score.tsv"),
            sep="\t",
            index=False,
        )

        return models, np.mean(scores)

    def run_predict_cv(self):
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction cv")

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f"{self.run_name} - start prediction fold:{i_fold}")
            model = self.build_model(i_fold)
            model.load_model(model_dir=self.output_dir)
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f"{self.run_name} - end prediction fold:{i_fold}")

        # 予測の平均値を出力する
        # if self.split_type == "TimeSeriesSplit":
        #    # TimeSeriesSplitの時はfold番号大きいほうがtrainデータ数多いので傾斜つけたほうがよさそうだが。。。
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, os.path.join(self.output_dir, f"{self.run_name}-test.pkl"))

        logger.info(f"{self.run_name} - end prediction cv")

    def run_train_all(self, i_fold="all"):
        """学習データすべてで学習し、そのモデルを保存する"""
        logger.info(f"{self.run_name} - start training all")

        # 学習データ全てで学習を行う
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(model_dir=self.output_dir)

        # 一応tsvファイルでもモデルのパラメ残しておく
        pd.DataFrame.from_dict(model.params, orient="index").to_csv(
            os.path.join(self.output_dir, f"{self.run_name}-{i_fold}_param.tsv"),
            sep="\t",
        )

        logger.info(f"{self.run_name} - end training all")

        return model

    def run_predict_all(self, i_fold="all") -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction all")

        test_x = self.load_x_test()

        # 学習データ全てで学習したモデルで予測を行う

        model = self.build_model(i_fold)
        model.load_model(model_dir=self.output_dir)
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(
            pred, os.path.join(self.output_dir, f"{self.run_name}-{i_fold}-test.pkl")
        )

        logger.info(f"{self.run_name} - end prediction all")

    def run_predict_all_train(self, i_fold="all") -> None:
        """学習データすべてで学習したモデルにより、学習データの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction train all")

        train_x = self.load_x_train()

        # 学習データ全てで学習したモデルで予測を行う
        model = self.build_model(i_fold)
        model.load_model(model_dir=self.output_dir)
        pred = model.predict(train_x)

        # 予測結果の保存
        Util.dump(
            pred, os.path.join(self.output_dir, f"{self.run_name}-{i_fold}-train.pkl")
        )

        # 横軸時系列でtrain setの正解と予測plot
        train_y = self.load_y_train()
        df_train = pd.concat([train_x, train_y], axis=1)
        pred_y = pred.copy()
        if self.val_eval == "rmse":
            # 対数化を戻す
            pred_y = np.expm1(pred_y)
            df_train[self.target] = df_train[self.target].apply(np.expm1)
        path_name = os.path.join(self.output_dir, "ts_plot_trainset_true_pred")
        Util.ts_plot_trainset_true_pred(
            pred_y, df_train, path_name, target_col=self.target
        )

        logger.info(f"{self.run_name} - end prediction train all")

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        if self.features is None:
            df = pd.read_csv(self.train_csv)
            # 削除したい列あれば削除
            if self.del_features is not None:
                for c in self.del_features:
                    if c in df.columns.to_list():
                        df = df.drop(c, axis=1)
            # ラベル列あれば削除
            if self.target in df.columns.to_list():
                df = df.drop(self.target, axis=1)
            return df
        else:
            return pd.read_csv(self.train_csv)[self.features]

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        return pd.read_csv(self.train_csv)[self.target]

    def load_x_cv_train_val(self, _train_csv, _val_csv) -> pd.DataFrame:
        """クロスバリデーション用に作った学習・検証データの特徴量を読み込む

        :return: 学習・検証データの特徴量
        """
        if self.features is None:
            tr_df = pd.read_csv(_train_csv)
            va_df = pd.read_csv(_val_csv)
            # 削除したい列あれば削除
            if self.del_features is not None:
                for c in self.del_features:
                    if c in tr_df.columns.to_list():
                        tr_df = tr_df.drop(c, axis=1)
                        va_df = va_df.drop(c, axis=1)
            # ラベル列削除
            tr_df = tr_df.drop(self.target, axis=1)
            va_df = va_df.drop(self.target, axis=1)
            return tr_df, va_df
        else:
            return (
                pd.read_csv(_train_csv)[self.features],
                pd.read_csv(_val_csv)[self.features],
            )

    def load_y_cv_train_val(self, _train_csv, _val_csv) -> pd.Series:
        """クロスバリデーション用に作った学習・検証データの目的変数を読み込む

        :return: 学習・検証データの目的変数
        """
        return (
            pd.read_csv(_train_csv)[self.target],
            pd.read_csv(_val_csv)[self.target],
        )

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        if self.features is None:
            df = pd.read_csv(self.test_csv)
            # 削除したい列あれば削除
            if self.del_features is not None:
                for c in self.del_features:
                    if c in df.columns.to_list():
                        df = df.drop(c, axis=1)
            # ラベル列あれば削除
            if self.target in df.columns.to_list():
                df = df.drop(self.target, axis=1)
            return df
        else:
            return pd.read_csv(self.test_csv)[self.features]

    def load_index_fold(self, i_fold: int, random_state=71) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        if self.split_type == "StratifiedKFold":
            # StratifiedKFold:ラベルの比率が揃うようにtrainデータとtestデータを分ける
            sms = StratifiedKFold(
                n_splits=self.n_fold, shuffle=True, random_state=random_state
            )
            # logger.info(f"StratifiedKFold n_splits={self.n_fold}, random_state={random_state}")
        elif self.split_type == "TimeSeriesSplit":
            # 時系列クロスバリデーション
            # データは必ず時間の昇順に並べておく必要がある！！！
            # TimeSeriesSplit はtrain を先頭から順番にデータ切っていき、後ろのレコードをvalidation するだけなので！！！
            sms = TimeSeriesSplit(n_splits=self.n_fold)
            # logger.info(f"　TimeSeriesSplit n_splits={self.n_fold}")
        else:
            # K-分割交差検証
            sms = KFold(n_splits=self.n_fold, shuffle=True, random_state=random_state)
            # logger.info(f"KFold n_splits={self.n_fold}, random_state={random_state}")
        return list(sms.split(dummy_x, train_y))[i_fold]
