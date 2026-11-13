# 差分IDMの比較・学習延長計画

2026-11-13。ユーザー承認済み。方針を先にcommit/pushし、その後に実装・GPU smoke・本学習を行う。

## 目的と現在の根拠

新規差分IDMは `z = head((Q(C,F) - Q(C,C)) / scale)` を元decoderに直接入力し、IDMとdecoderを共同学習する。C/Fは固定DINOの現在／未来特徴、Qは共有encoder。400更新・3 seedの予備実験は、既観測96例で元モデルよりMSEが14.74%低く、encoder固定対照より1.63%低かった。ただし差分構造自体の効果、未使用場面への汎化、制御性能は未証明。[実測結果](../research/README.md)。

今回の主目的は、同じ条件で差分なしIDMと比較し、400→800→1600更新で優位が持続するか調べること。正解未来を入力するposterior再構成の実験であり、行動条件付き未来予測・制御の実証とは区別する。

## 比較条件

| 項目 | 事前方針 |
|---|---|
| 主比較 | 差分あり `Q(C,F)-Q(C,C)` と差分なし `Q(C,F)`、ともにencoder/head/元decoderを学習 |
| 学習量 | 各1600更新の連続軌跡から400・800・1600更新を保存・評価。更新ごとの学習率を一定にして共通prefixを比較 |
| seed | 42/43/44。方式間で同seedのhead初期値とminibatch順を共有 |
| 初期重み | 同じlocal final10000 encoder/decoder。差分専用の学習済みheadを流用せず、両方式で同じランダム初期headを使う |
| 入力校正 | 訓練特徴だけで各方式の固定scalar RMSを求める。校正値・初期code分布・初期誤差を記録し、校正の違いも明示する |
| 共通設定 | head1024→256→32、決定論的MSE、batch8、encoder/head/decoder LR=1e-5/1e-4/3e-5、matrix weight decay .01、bias/norm0、gradient clip1、float32 highest |
| 固定するもの | DINO、訓練512例、データ順、損失、decoder構造、評価指標。同一step比較であり同一FLOPsではない |
| 補助対照 | まず主比較6 runを完了。以前のencoder固定実験は初期headが異なるため別記。新条件の固定encoder対照は必要に応じ追加 |

同一seedは共同学習の全初期重みが独立という意味ではなく、事前学習encoder/decoderは全seed共通。今回の新規headの初期化とminibatch順が変化する。以前のhead初期化での400更新値とは直結した学習曲線として扱わない。

## データ分割と評価

1. 利用可能manifestと元checkpointの訓練／validation manifestを監査する。episode IDだけでなく元データの同一性も確認し、未使用評価候補をモデル出力を見る前に固定する。
2. 既存validation64とobserved96は開発・診断用。新たな独立testが確保できた場合は選択に使わず、全run終了後に事前指定の1600更新モデルを一度評価する。未使用splitが存在しない場合は、その不足を明示して主比較を進め、既存集合を新規testと呼ばない。
3. 400・800・1600の訓練／開発MSE、Smooth L1、MAE、cosine similarity、seed別結果を記録。コード交換と同一画像組コードも診断する。差分なしの同一画像組は実際に `Q(C,C)` を計算する。
4. 主効果は同step・同seed・同episodeの差。seed別差とseed平均のpaired episode bootstrap区間を併記する。3 seedで全不確実性を推定できるとは主張しない。小変化例の悪化と追加計算費用も報告する。
5. train512・batch8では400/800/1600更新が約6.25/12.5/25周相当。訓練誤差だけが下がる場合は過学習を疑い、更新増より訓練データ拡大を優先する。

## 実装・実行・成果物

GPU使用率・空きメモリを確認し、他ジョブを変更せずに使用する。差分分岐の両側への勾配、差分なし経路、共通初期値、途中checkpointの再読込・再開をテストする。実モデルの少数更新で数値・勾配・VRAM・時間を確認してから本学習へ進む。

既存400更新runは上書きせず、新しい出力ディレクトリを使う。20更新ごとの進捗、100更新ごとの再開用checkpoint、400/800/1600更新の評価用checkpointを保存する。データ・ソース・初期重み・batch順のhashと固定protocolを残す。GPU2枚が利用可能なら方式を並行実行する。これは複数プロセスの学習であり、エージェント委任を必要としない。

実装・測定結果・再現手順はGitへ保存する。大容量の学習済み重みと特徴cacheはローカルに保存し、Gitには含めない。計画時点で結果を断定しない。

## その後の研究段階

主比較の結果を受けて、データ規模拡大、zからの相対変位・把持状態の予測、行動条件付き予測・制御成功率を検証する。今回の実行範囲はデータ監査と主比較6 runおよび評価まで。制御実験や新しいデータ収集は今回のMSE結果を踏まえて設計する。

## 実装・データ監査の進捗（2026-11-13）

方針はcommit `283bd50`で実装に先立ってpushした。主比較のtrainer、2 GPUの実行キュー、全6 run完了を条件にした最終test評価と学習曲線集計を実装した。

未使用評価候補として既存`test.jsonl`の全170エピソードから、各1個の中間・非末尾pairをモデル出力を見ずに固定した。元モデルのtrain1444／validation160エピソードとのepisode・動画group・動画パス・arrayパス・動画内容hashの重複はない。元runに保存されたtrain/validation manifest hashとも一致する。testは7動画shard・既知の6タスク群・同一カメラであり、未知タスク・未知カメラへの汎化評価ではない。リポジトリにtestスコアを用いた研究記録は見つからなかったが、外部環境を含む完全な使用履歴までは保証しない。[固定した選択と監査](../research/results/difference-idm-ablation-20261113/test-selection.json)。

CPU/GPU回帰テスト155件通過。GPUの4更新smokeでは両方式の初期encoder/head/decoder hashとbatch順が一致し、全24層への非ゼロ勾配を確認した。差分ありの両分岐は逆符号の勾配を受け、差分なしはQを1回だけ計算する。2・4更新checkpointの再読込MSE差は最大2.09e-7だった。学習時最大allocatedは差分あり19.27GiB、差分なし14.25GiB。更新時間は約0.85秒対0.48秒（保存・評価を除く）。[smoke記録](../research/results/difference-idm-ablation-20261113/smoke.json)、[テスト記録](../research/results/difference-idm-ablation-20261113/tests.txt)。

testの特徴抽出では、元`VJEPA_LAM`コンストラクタがmatmul precisionを`medium`へ変更するため、構築後に`highest`へ戻す必要があった。既存trainキャッシュとの照合で発見し修正した。抽出はDINOだけを使い、testの予測スコアを計算しない。

```sh
# 図の生成に使う追加依存（既存環境ではpython -m pip install matplotlib==3.11.2でも可）。
uv sync --group research
# 出力先は大容量ディスク上の新規ディレクトリを指定する。
.venv/bin/python research/prepare_idm_ablation_test.py audit
.venv/bin/python research/prepare_idm_ablation_test.py extract --device cuda:0
.venv/bin/python research/run_idm_ablation.py \
  --runs /data2/yuki_sadasue/runs/foveated-world-models/difference-idm-ablation-20261113
```

初回だけaudit/extractを実行する。既存成果物は上書きしない。学習キューは`queue-status.json`、各runの`progress.json`とログに進捗を保存する。途中終了時は同じキューを再実行すると、完了runをスキップして保存済みの同一protocolから再開する。全run完了後に最終評価・18 checkpointの再読込検証・`learning-curves.png`作成まで自動実行する。完了したtest結果は自動で再評価しない。
