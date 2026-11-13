# 修正後LeRobot入力での本学習・再検証

2026-11-13。リモートの `c7391ad` までを取り込み、GB10で新規Stage 1学習を開始した。旧NPZ由来のcheckpointは再開に使わない。ローカルのREADME追記は維持し、既存warmup/max_stepsの変更はリモートと同じ値だったため重複を解消した。

## 実行条件

[GB10設定](../configs/stage1_lerobot_gb10.yaml)は修正後RoboMIND設定を継承し、原本データとDINOのローカルパスだけを指定する。seed42、batch2、warmup5000、10000更新、BF16 mixed、IDM24層・LaWM12層・幅1024・z32。学習対象は約6.4億パラメータ、DINOは固定。validationは100更新ごとに全950ペアを評価する。

GPU使用率0%を確認して開始した。他用途の常駐プロセスには変更を加えていない。出力は `runs/stage1-lerobot-rpy14-minmax-gb10-20261113`。20更新ごとの `progress.json`、`training.log`、Lightning checkpointを保存する。生のcheckpointはGitへ含めない。

保存先を訂正し、現在の実体は `/media/sadasue/SSD1/foveated-world-models/runs/stage1-lerobot-rpy14-minmax-gb10-20261113` に置く。上記リポジトリ内パスはSSD1へのシンボリックリンクである。移動前後のcheckpoint SHA256一致を確認し、800更新から再開した。通常の保持対象は最良1件と最新1件、終了時にfinalを追加する。旧パスが同じ実体へ解決される場合は、再開時にcheckpoint管理のパスだけを正規化し、最良モデルの記録を維持する。

ユーザーの指示により、旧データ処理で学習したモデル164ファイル（47,393,223,957 bytes）を削除した。旧基準モデル、その複製、外付け補正器、内部adapter、IDM headやprobeの学習済み重みを含む。原本データ、DINOの事前学習重み、特徴cache、過去の評価記録は維持した。旧研究のcheckpointパス・hashは履歴であり、今後その重みを読み直すことはできない。[削除記録](../research/results/legacy-model-deletion-20261113.json)、[本学習checkpoint移動記録](../research/results/ssd1-checkpoint-migration-20261113.json)。

## 事前検証

- 取り込み後および実行準備後のテストは164件成功。
- 同じフルモデルで3更新、検証、checkpoint保存が成功。学習・検証・途中保存を含むfit時間64.73秒、GPU割当メモリピーク約13.2GB。初期化・保存コストを含むため、この時間をそのまま10000更新へ外挿しない。
- そのcheckpointから修正後validation/test各4ペアを読み、通常z・ゼロz・循環入替zによる再検証コードの実行が成功。3更新の結果は精度結論には使わない。
- ローカル原本から生成されたデータ契約は、リモートのloader修正検証に保存された契約と完全一致した。

[事前評価記録](../research/results/lerobot-gb10-preflight-evaluation-20261113.json)。

## 完了後の再検証

[実行スクリプト](../research/run_corrected_stage1.py)は本学習が正常終了した場合だけ、[評価スクリプト](../research/evaluate_corrected_stage1.py)を起動する。途中失敗では `pipeline_status.json` に失敗段階・終了コードを残して停止する。

最終10000更新checkpointを固定し、修正後のvalidation950ペアとtest1081ペアについて、通常posterior mean z・ゼロz・全評価例内で1つ循環移動したzを比較する。MSE、Smooth L1、MAE、cosine similarity、特徴変化量上位10%／下位50%領域のMSEとzの成分別分散を保存する。評価はfloat32 highest、画像augmentationなし。定数zを使った学習対照や新規encoder改善の実験ではない。

旧画像処理のcacheや数値を直接比較しない。真の未来から推定するzを使う再構成評価であり、実行時の行動条件付き予測とは区別する。test分割は既存研究で参照済みのため未観測の最終テストとは呼ばない。

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u research/run_corrected_stage1.py \
  --config configs/stage1_lerobot_gb10.yaml \
  --output /media/sadasue/SSD1/foveated-world-models/runs/reproduce-stage1-lerobot-full
```

出力先は新規ディレクトリを指定する。`pipeline_status.json` が `complete` になり、`run_summary.json` と `revalidation.json` が揃って初めて本学習・再検証完了と扱う。
