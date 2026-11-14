# 3000更新後の共通FP32評価の準備

[専用評価スクリプト](../../evaluate_robotwin_loss_ablation_fp32.py)を追加した。学習コードは変更していない。旧10000更新と公式モデルによる限定GPU動作確認は完了した。最終評価プロセスPID `126286`を起動し、4条件の3000更新完了をGPU未使用で待機している。最終checkpointの評価結果はまだない。[起動引数と準備検証の証拠](readiness.json)。

4条件の500更新時点の512組・415エピソード・交換先が完全一致すること、SSD1の実マウント、設定の差分、公式とローカルの画像計算コードの一致をCPUだけで検証した。小さい合成データで、対応付きエピソードbootstrap・KL/MSE周辺効果と交互作用・部分集合・公式状態評価の除外を確認した。旧10000更新checkpointのメタデータとhyperparametersのCPU検証も通った。[限定smokeスクリプト](smoke.py)をB1×2組で実行し、旧10000更新・公式の両モデルの全660テンソルをstrict load、公式APIとキャッシュ経由の特徴・μ・予測の完全一致、全モジュールの評価前後ハッシュ不変を確認した。[smoke実測結果](smoke-results.json)。所要81.0秒、GPU最大割当2.80GiB。初回smokeではraw collateにdec_videosが存在しないことを検出し、公式のGPU評価正規化を呼ぶ修正を行った後に再試験が通った。

実行時は4つのfinal3000 checkpointのglobal_step・hyperparameters・data contractを確認し、全重みをstrict loadする。DINOを含む各モジュールのハッシュを記録し、評価前後の不変性を検証する。公式モデルも実際の公開重みを全量ロードし、DINO重みの一致を要求する。公式の14次元状態ヘッドをRoboTwinの16次元状態で採点しない。

DINO・encoder・VAE・decoder・状態ヘッドを最高精度FP32、TF32無効で実行する。512組のDINO特徴と実際の未来ラベルをSSD1へ一度だけ保存し、4条件と公式の全512組の画像評価で共有する。実μ・0・全512組の平均μ・別エピソードμ・current_only μ=IDM(x,x)を比較する。current_onlyでもdecoderの現在特徴xと正解未来yは固定し、分布外になり得る介入として扱う。未来特徴MSEと状態MSEに加え、変化上位10%のパッチ、画像変化量上位25%の組、腕の移動5cm以上、完全な時間幅かつ移動5cm以上を評価する。部分集合は入力と正解だけから固定し、全モデルで一致させる。

以下のコマンドで、親エージェントのレビュー・実行許可・smoke成功後に起動済み。リポジトリのルートを作業ディレクトリとしている。`--wait-for-checkpoints`は4条件の完了と保存を待ち、次にホスト16GiB以上・CUDA8GiB以上の余裕を待つ。学習の起動・再開・設定変更は行わない。待機中にwatch_status.failed_armsがあれば失敗を報告して終了する。

```sh
.venv/bin/python -u research/evaluate_robotwin_loss_ablation_fp32.py \
  --runs-root /media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114 \
  --output /media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114/evaluations/final3000-highest-fp32 \
  --batch-size 2 --bootstrap 4000 --wait-for-checkpoints
```

保存される`results.json`には、全体と各部分集合の同じエピソード再標本化による区間、4条件の対応付き周辺効果・交互作用を含む。各条件の個別行は`<arm>/results.json`、μなどは`<arm>/posterior-codes.pt`に残す。平均μは検証512組に基づく診断用定数であり、別途学習した予測モデルではない。区間は固定重み・平均・交換順列に条件づけたもので、学習seed、交換順列、モデル精度の不確実性を表さない。BF16定期検証とはDINO特徴も異なるため、この共通FP32評価の中で比較する。

起動直後に`/proc/126286/maps`にlibtorch/libcuda/libcudartがないこと、GPUプロセス一覧にPIDがないこと、ログが`waiting_without_CUDA`であることを確認した。起動時のscript/config/source/dataハッシュを`launch-pins.json`へ固定し、checkpoint完了時に再検証する。最終スクリプトSHA256は`e9c1282f15e01832d4c12a01b8e74fdf5229bbf8859d5e48e71c1a84b11b5b01`。実行ログと起動記録はSSD1出力先の`evaluator.log`・`evaluator-launch.json`に保存した。
