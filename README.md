# LightGBMでモデル作成＋パラメータチューニングする自作パッケージ

## configディレクトリのymlファイルでパラメータを渡し、batやshファイルで実行する
- 回帰モデルのみ作成可能
	- チューニングするパラメータは https://github.com/Y-oHr-N/OptGBM/blob/master/optgbm/sklearn.py#L194-L221 を参考にした
- xfeatが必要
	- githubディレクトリに置いてる
- util.pyのSubmissionクラスはコンペのデータに合わせて変更しないとエラーになる

