# g損失の初期縮退への寄与：500更新の対比較

**準備済み。GPU preflight・500更新学習とも未開始。** [CPU確認・固定SHA・未確認事項](readiness.json)、[v2からのrunner差分](runner-vs-causal-v2.patch)。既存のv1/v2・継続学習・評価スクリプトは変更していない。

実行ファイルは[train_robotwin_g_onset.py](../../train_robotwin_g_onset.py)と、新規[FP32評価helper](../../robotwin_g_onset_probe.py)。設定は[g_on](../../../configs/robotwin_g_onset_20261114/g_on.yaml)／[g_off](../../../configs/robotwin_g_onset_20261114/g_off.yaml)。v2からの学習runner差分は追加48行・削除24行で、学習forward、AdamW、clipping、データ変換、ε監査を継承する。

## 条件と目的

共通条件はfresh seed42、B4、MSE、KLβ=1e-5、warmup1000、peak LR3e-4、g出力幅2*tanh。変えるmodel設定は`lambda_aux=1`／`0`だけ。500更新で自動停止し、後続ジョブ・本学習を起動する処理はない。500更新はwarmup途中であり、その時点でpeak LRには達しない。

初期のg→encoder勾配が画像勾配より強かった実測と、ローカルg出力幅変更の履歴が動機。ただし**公式もAdaLNをゼロ初期化するので、初期g優勢だけでは公式との差を説明できない**。データは多数のtask・episodeを含み、状態差分も一方向ではない。「単調なデータが原因」「gripperが原因」とは仮定しない。

g0はgの重み付き損失を0にする。gのforward自体は残すため、入力や乱数経路を変えない。gパラメータにはAdamW decayが作用し得るが、g損失からencoderへ勾配は流れない。gあり側で初期縮退を再現し、g0側で防げるかを調べる診断であり、g0を本学習の解決策として採用する試験ではない。

## 同条件を保証する記録

- 初期の全trainable parameter fingerprintは、元candidate runの`c933311d638c1b880d370ff85195220bdd06394cdb4ef62710fa8fa4d522e8a0`との一致を実行時に要求する。
- 各更新のepisode/base、画像拡張後のtensor、消費したGaussian εのSHAを、元candidateのseed42 prefixと照合。不一致なら停止する。
- 同じ検証episode split・DINO最高精度FP32 cacheを再利用し、cache SHA、状態、差分、行ID、モデルのDINO fingerprintを確認する。
- 元v2の実損失・ノード／選択パラメータ勾配・層別B4 geometry監査を、更新前0,1,9,49,99,249,499に残す。勾配ノルムをそのまま更新速度や原因と解釈しない。

## onset評価

|更新後step|固定評価ペア数|
|---|---:|
|0,10,50,100,250|同じprefix64|
|500|512。先頭64のgeometryも別項目で集計|
|6更新preflight|0と6で同じ64|

画像のDINO特徴は既存の共通FP32 cacheから読み、encoder/VAE/decoderをFP32、TF32無効で実行する。μ、logvar、pre-LN nodes、current-only μ/logvarを各`evaluations/stepXXXXXX/posterior-codes.pt`へ保存。全共分散固有値・方向別SNR・μの分散・平均／交換／zero／current-only介入の画像誤差を保存する。最終512の先頭64を集計するため、標本数をそろえた経時比較も可能。

評価前後でPython・NumPy・Torch CPU/CUDAのRNG、全moduleのtraining flag、parameter versionとrequires_grad、matmul precision・TF32 flagが不変であることをassertする。評価は損失のbackwardやoptimizer操作をしない。計算精度の切り替えだけでGPU演算のbitwise一致まで保証されるとはしないため、6更新preflightで実数値も確認する。

CPU確認中、PyTorch2.11では`medium`の復元後に旧`allow_tf32` setterを呼ぶと精度API混在エラーになることを検出した。新helperはlegacy flagを先に、`set_float32_matmul_precision(saved)`を最後に復元する。CPUで`medium`へ戻ることを確認済み。既存継続runnerにも同じ復元順があったことはrootへ報告したが、本準備では既存ファイルを変更していない。

## 保存先と確認状況

成果物の実体は `/media/sadasue/SSD1/foveated-world-models/runs/robotwin-g-onset-b4-20261114`。SSD1が`/dev/sda1`としてmountされていることを確認した。`prepared-v2/g_on`と`prepared-v2/g_off`でCPU prepare-onlyを実行し、CUDA未初期化、データ契約、固定cache、episode境界の検査が完了。コード構文・configのfactor差・CPU geometryの既存独立集計との一致も確認した。

GPUでの初期重み一致、実行数値、評価のRNG不変性、メモリ・時間は未確認。root/trainerが既存4継続ジョブと資源調整した後、両条件の6更新preflightで確認する。通常実行では最終500のcheckpointを1件だけ保存し、途中は小さな事後行列を保存する。

既存の実測2.55〜3.15秒/更新から、500更新は1条件約21〜26分。FP32評価分を含め約24〜31分/条件が暫定見積もりで、並列時の時間は資源共有に依存する。既存train peak allocatedは約13.32GiB/条件、reserved約13.98GiB。新しい評価込みpeakはpreflightで測る。full checkpointは既存形式で約8.03GB/条件、2条件で約16.1GB。共通特徴cacheを複製しない。

## 未実行のコマンド

リポジトリ直下。以下は準備した手順で、実行済みという意味ではない。

```sh
onset_group=/media/sadasue/SSD1/foveated-world-models/runs/robotwin-g-onset-b4-20261114
.venv/bin/python research/train_robotwin_g_onset.py --config configs/robotwin_g_onset_20261114/g_on.yaml --output "$onset_group/preflight/g_on" --preflight
.venv/bin/python research/train_robotwin_g_onset.py --config configs/robotwin_g_onset_20261114/g_off.yaml --output "$onset_group/preflight/g_off" --preflight
```

preflight確認・資源調整後の500更新実行は、`--preflight`を外し、空の出力先`$onset_group/g_on`と`$onset_group/g_off`を指定する。既存出力を上書きしない。縮退をgあり側で再現できなければ、g0との差による原因判定は保留する。全体損失が下がることだけを成功条件としない。
