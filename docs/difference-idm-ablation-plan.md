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

**13:40 JSTに本学習を開始し、14:58に全6 runの学習が完了。集計処理の修正後、15:09に最終評価・集計も正常終了した。** 実装commitは`027668a`。[起動記録](../research/results/difference-idm-ablation-20261113/launch.json)、[固定protocol](../research/results/difference-idm-ablation-20261113/protocol.json)、[最終結果](../research/results/difference-idm-ablation-20261113/results.json)。ローカルの`runs/difference-idm-ablation-20261113/`は大容量ディスク上の成果物ディレクトリへのリンク。

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

## 最終評価（2026-11-13 15:09 JST）

差分ありは固定terminal1600のtest170で差分なしよりMSEが2.21%低い。3 seedすべてで改善したが、開発集合より効果は小さい。全18 checkpointの再読込による既存評価の照合は、各8例のMSE最大絶対差1.79e-7で通過した。[結果JSON](../research/results/difference-idm-ablation-20261113/results.json)、[学習曲線](../research/results/difference-idm-ablation-20261113/learning-curves.png)。

| 集合（1600更新・3 seed平均） | 差分ありMSE | 差分なしMSE | 差分ありの相対改善 |
|---|---:|---:|---:|
| train512 | 0.043909 | 0.041667 | -5.38% |
| validation64 | 0.101808 | 0.109047 | 6.64% |
| observed96（開発用） | 0.090690 | 0.100044 | 9.35% |
| test170 | 0.107915 | 0.110353 | 2.21% |

testの差分あり−差分なしのMSE差は-0.002438。paired episode bootstrapの95%区間は[-0.003624, -0.001194]、動画shard単位では[-0.003915, -0.001339]。どちらも0をまたがないが、共有事前学習重み・既知6タスク群・同一カメラ・7動画shardという条件付きの評価であり、少数clusterの区間精度には限界がある。seed42/43/44のMSE差はそれぞれ-0.002711/-0.001765/-0.002837。

両方式とも400→800→1600更新で訓練誤差は下がるが、3 seed平均のvalidation/observed96誤差は上がる。評価した3時点では400更新が開発集合で最良であり、追加更新より訓練データ拡大と開発集合による早期停止の検証を優先する。testは事前指定の1600更新だけに使用し、この結果を見て400/800更新を追加採点していない。

差分ありのtestコード交換MSEは0.108926、同一画像組コードでは0.108861（通常0.107915）で、正しい未来由来コードの利得は約1%と小さい。rawは通常0.110353、交換0.110472、同一画像組0.110139。未来由来コードの利用や物理的な操作表現の有効性を強く主張できる結果ではない。正解未来を入力するposterior再構成であり、制御性能は未評価。

保存・評価を除く1600更新の平均学習時間は差分あり1497.6秒、raw879.8秒で約1.70倍。最大allocated GPUメモリは19.27対14.25GiB。同step比較であり同計算予算比較ではない。

初回集計は`options.decoder.grid_hw`がcheckpoint内ではtuple、JSON内ではlistになるため、同値の設定の直接比較で停止した。集計側だけをJSON表現で比較するよう修正し、全フィールドの一致チェックを維持した。回帰テスト7件通過。再実行は完了済み学習をすべてスキップし、重み・学習設定を変更せず評価した。初回エラーはtest採点より前であり、再実行時に各terminal checkpointを初めてtest採点した。修正後集計スクリプトのhashは結果JSONに記録される。

### Stage 1でのzの追加効果と外付け方式との関係

今回の差分IDMでは同一画像組の差分が0になるため、`z(C,C)=head(0)`は各学習済みモデル内で全例共通の定数。学習済み重みを固定してこのコードへ差し替えた結果は以下。rawの`z(C,C)`はCに依存するため、rawの同一画像組診断を定数コード対照とは呼ばない。

| 集合・1600更新・3 seed平均 | 通常z MSE | 定数head(0) MSE | 定数化による悪化 | 別例zへの交換による悪化 |
|---|---:|---:|---:|---:|
| validation64 | 0.101808 | 0.118935 | 16.82% | 23.40% |
| observed96 | 0.090690 | 0.095815 | 5.65% | 10.18% |
| test170 | 0.107915 | 0.108861 | 0.88% | 0.94% |

開発集合では予測精度に対するzの利得があるが、testでは小さい。これは正解未来に対するMSEの差であり、通常zと定数zの予測テンソル間の距離を直接測った値ではない。「出力が同一」や「decoderがzを完全に無視する」とは結論できない。全例をずらすコード交換も、現在条件を揃えた介入ではない。

以前の外付け版の16.33%改善は、旧encoder2900＋prior10000条件・既観測96での元モデル比（0.103631→0.086709、400更新・3 seed）。今回の2.21%はtest170で同条件のraw追加学習モデルに対する改善であり、比較相手も評価集合も違う。旧外付け版の既観測96では、コードを定数にして別途学習した同容量対照が0.095100に対し通常コードが0.086709で8.82%低かった。この対照はzの追加効果を支持するが、今回の推論時差替えとは実験が異なる。[旧外付けの定数対照](../research/results/nested512-controls-summary-observed96-20261113.json)。

また、このPCのfinal10000から再構築した外付け版は400更新で既観測96 MSE 0.093284だった。今回の共同学習版は400更新0.087940、1600更新0.090690で、それより低い。旧外付け版の最良値だけで外付け構造の優位とは判断できない。外付け版と共同学習版の優劣を確定するには、同じencoder/head初期値・データ・学習予算・評価集合と、各方式の定数コード対照を揃えた比較が必要。[再構築条件の比較](../research/README.md#内部統合版のgpu検証2026-11-13再構築条件)。
