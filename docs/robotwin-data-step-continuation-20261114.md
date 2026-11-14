# RoboTwin：既存ペア反復と未投入ペアの追加学習

2026-11-14。ユーザー指定により、同じ3000更新checkpointから「既に学習に使ったペアの反復」と「未投入ペアの追加」を比較する。seedを変えたもう1組も実施し、合計4本とする。本学習のbatch64・30000更新とは別の短期比較である。

## 固定条件

開始点は `robotwin-loss-ablation-b4-20261114/kl_mse/checkpoints/final.ckpt`。画像MSE・KL1e-5の候補で、同じ512組のFP32評価ではz交換による誤差増加が1.84%、公式配布重みは46.60%だった。候補は修復済みではない。[公式を含む全対照](../research/results/robotwin-loss-ablation-3000-fp32-20261114/analysis.md)。

| 条件 | 継続seed | 追加学習のペア | 追加更新 |
|---|---:|---|---:|
| seed43_replay | 43 | 元3000更新で使用した12000件を並べ替えて反復 | 3000 |
| seed43_new | 43 | 元の学習で使っていない12000ペア | 3000 |
| seed44_replay | 44 | 同じ反復集合を別の順序で使用 | 3000 |
| seed44_new | 44 | 同じ未投入集合を別の順序で使用 | 3000 |

batch4、LR3e-4、AdamWのmoment・更新回数・schedulerを引き継ぐ。ウォームアップは再開しない。学習forwardとGPU augmentationは固定commitの上流実装を使い、BF16、勾配clip1を維持する。seed内では両集合に同じ位置の並べ替えを使い、augmentation前後のCUDA RNG状態と実際のVAE epsilonを照合する。画像が異なるので、拡張後の画素hashの一致は要求しない。

4本の開始重みとoptimizerは同一である。seedで変えるのは継続中の順序・augmentation・VAE乱数であり、初期学習から独立なseed再現実験ではない。

## データの証拠

[生成コード](../research/build_robotwin_continuation_manifests.py)で元の全3000バッチのID hashを再構成し、保存ログと一致させた。反復12000件は11988固有ペア・9096episodeを含む。未投入側は12000固有ペア・9171episodeで、反復側とのペア重複は0。

「未投入」はepisodeと現在frameの組の新しさを指す。両集合は3694episodeと272端点frameを共有し、未投入側283件では少なくとも片方の端点が元集合に含まれる。新しいepisodeや新しいデータセットを増やす実験とは区別する。horizonが48frame未満の件数は反復2634・未投入2635で、時間差0は72・53。

固定リストはSSD1の `runs/robotwin-data-step-continuation-b4-20261114/manifest.json` と `manifests/seed*.json` に保存した。各読み込みでもepisode/frame IDを照合し、IO retryによる別ペアへの置き換えを黙認しない。

## 更新・評価・保存

[実行コード](../research/train_robotwin_data_step_continuation.py)は重み・Adam・schedulerを明示的にロードし、ローカル更新1–3000を通算3001–6000として記録する。元のLightning checkpointはepoch=1を保存しているため、学習loopやdataloader位置の自動再開に依存しない。開始前にモデル全stateとoptimizer全stateのhash一致、Adam449状態の更新回数3000、scheduler3000、LR3e-4を確認する。

500追加更新ごとに同じ512組・415episode、同じ交換先、最高精度FP32の共通DINO特徴、評価batch2で通常z・z=0・平均z・交換z・現在入力のみを比較する。公式配布重みの対照間の関係と効果量を毎回の基準にし、単に0との差が有意というだけで修復と判定しない。評価は学習のRNG状態を保存・復元して行う。

checkpoint実体はSSD1。500追加更新ごとに最新1件を更新し、最終版は同じ実体へのhard linkを作る。4本の同時保存による一時メモリ増加を避けるため、保存をロックで直列化する。元の3000・10000更新checkpointは変更しない。

比較の読み方は、開始点対反復で「同じペア集合に対する追加更新の効果」、反復対未投入で「同じ追加更新数でペア集合を入れ替えた効果」を見る。新旧ペアの内容差も含む比較であり、データ数だけの普遍的効果や長期収束を一組の結果から断定しない。seed43/44で方向・効果量が再現するかも確認する。

## 起動状況

4本それぞれの6更新preflightをGPUで完了した。全4本の開始重みとAdam stateのhashが一致し、同seedの両条件で6更新すべてのaugmentation前後CUDA RNG・VAE epsilonが一致した。各更新で全VAE学習パラメータの勾配存在・有限性を確認し、schedulerは3006、LRは3e-4だった。1本あたりGPU最大割当12.31GiB・reserved12.87GiB、初回を除く更新中央値2.77–2.79秒。[事前確認の全記録](../research/results/robotwin-data-step-continuation-20261114/preflight-summary.json)。

その後、元の3000更新checkpointから改めて4本の追加3000更新を開始した。preflightの6更新は本比較へ継ぎ足していない。PIDはseed43_replay=343513、seed43_new=343514、seed44_replay=343515、seed44_new=343516。[起動記録](../research/results/robotwin-data-step-continuation-20261114/launch.json)。実行ファイルSHA256は `17f4279f2761dcdfc9ce3e620d15e63bfad5e4f343b42a5225790a3f0d50f801`。更新だけで約2.3時間、FP32評価と保存を含め約3時間を見込むが、並行処理とIOで変わる。

公式・旧10000・今回の候補・3000対照群に対する共通線形プローブは完了した。全モデルのDINOからVAEまでを同じFP32条件に揃え、train2048・val512、episode分離、train内5foldでridge係数を選択。外付けheadだけを学習し、公式gは使っていない。重み不変・native/cache一致を確認し、保存済みheadから全予測と信頼区間を独立再計算した。[全比較と検証記録](../research/results/robotwin-matched-state-probes-fp32-20261114/analysis.md)。

状態だけにμを追加した正規化状態差分MSEの改善は、公式25.525%、候補3.015%、基準3000更新6.047%、旧10000更新1.505%だった。候補の `IDM(x,y)` と `IDM(x,x)` の全状態MSE差は95%区間が0を跨いだが、stateと組み合わせたXYZ成分には小さな正の差がある。未来情報が完全に0とは結論しない。raw z=0の外部ridgeでは局所モデルの狭い非ゼロμ分布から大きく外挿して誤差が極端に増える。これは画像decoderのz=0評価とは別であり、zの意味や利用が回復した根拠にはしない。

CPUの[監視・比較コード](../research/watch_robotwin_data_step_continuation.py)もPID347052で起動した。[監視の起動記録](../research/results/robotwin-data-step-continuation-20261114/watcher-launch.json)。4本のプロセス同一性、開始state、seed内のaugmentation・VAE epsilon一致、seed間の乱数差を継続確認する。各500更新の評価が揃うと、共通episodeで再標本化した未投入−反復の差と、開始3000更新・公式配布重みとの比較をSSD1の実行group内 `paired-analysis` に保存する。学習の再起動・停止や本学習の自動開始は行わない。2026-11-14 11:26 JST時点で4本とも追加297–299更新、記録された監視エラーは0件。

## 開始点の潜在表現

[同じ512組の方向別監査](../research/results/robotwin-posterior-direction-audit-20261114.md)では、候補のμ変動の99.9044%が第1主成分に集中し、公式は22.2283%だった。事後ノイズに対する方向別の信号比が1を超える方向も候補1・公式11で、単純な全体スケール差だけでは説明できない。候補の画像予測にzの寄与が検出されたことを、潜在表現の問題が解消した根拠とはしない。今後のcheckpointでも、この偏りとdecoderのz介入効果を合わせて確認する。原因がデータ・更新回数・損失設計のどれかは、開始点の幾何だけでは決められない。

[固定重みのCPUスペクトル検査](../research/results/robotwin-bottleneck-weight-spectra-20261114.json)では、候補のVAE μ Linearは特異値二乗の第1成分比10.21%・PR16.61で、LayerNormのaffineと中心化を含めた固定線形部分でも10.74%・PR16.02だった。最後のμ Linear自体がほぼrank 1になったという説明は支持されない。出力の共分散は入力分布とLayerNormの非線形性にも依存するため、これは原因の確定ではなく、encoder内部と投影前後の実際の入力共分散を調べる根拠となる。

[同じRoboTwin 64組の層別計測](../research/results/robotwin-z-underuse-20261114/layer-covariance-primary-summary.json)を完了した。候補は最初に観測したcross-attentionブロック出力で第1主成分93.87%、最終Transformer層96.45%、32次元投影後85.21%、VAE LayerNorm後99.83%、μ99.90%。公式の同じ箇所は36.94%、26.93%、27.90%、25.68%、20.30%だった。64組の値を512組の値と混同しない。観測処理の有無で全バッチのnodes・μが完全一致し、前後の重みhashも一致した。

[正規化の処理別分析](../research/results/robotwin-z-underuse-20261114/layer-covariance-normalization-decomposition.json)では、候補の投影後を各サンプル内で平均中心化しただけでは85.20%、標準偏差で割ると99.84%、学習済みaffineを適用すると99.83%だった。偏りが強まる計算箇所は特定できたが、LayerNorm一般の欠陥や、その手前の学習上の原因を確定したものではない。最初の観測ブロックにも複数の演算が含まれ、attention単体の責任とは同定していない。

## 実行中に検出した障害

初回500追加更新の比較時に、CPU watcherの変数名衝突により監視だけが終了した。旧source・status・logを保存し、[修正記録](../research/results/robotwin-data-step-continuation-20261114/watcher-v2-repair.json)のv2へ切り替え、実512組の比較で復旧を確認した。この障害で学習は停止していない。

その後、4本すべてが追加1000更新（通算4000）の2回目評価入口で終了した。評価後のFP32精度設定を復元する順序がPyTorch 2.11の新旧APIを不整合にし、次の `get_float32_matmul_precision()` が例外を出すことを[独立再現](../research/results/robotwin-precision-restoration-20261114.md)した。損失・勾配の非有限化ではなく、この例外から重み破損や実演算精度の変更は断定できない。

最後の完全checkpointは全4本とも追加500更新（通算3500）で、660個のモデルstate、449個すべてのAdam step=3500、scheduler=3500、LR3e-4を確認した。[checkpointとstream監査](../research/results/robotwin-data-step-continuation-20261114/recovery-checkpoint-and-stream-audit.json)。旧checkpointにはRNG stateがないため、元seed・実DataLoader・GPU augmentation・VAE乱数を500更新分再生し、保存済み乱数hashとの一致を確認する復旧を準備している。旧版を残した別versionで復旧し、保存されていないCPU RNGや拡張後画素について、過去hashとの照合ができたと主張しない。新規g有無実験のGPU起動は、この復旧まで保留する。

復旧用[runner v2](../research/train_robotwin_data_step_continuation_recovery_v2.py)の4本のGPU preflightが完了した。各500バッチのdata ID・CUDA RNG前後・VAE epsilon・累積hash、および501直前のCUDA RNGが旧記録と全件一致。各自の3500更新model/Adamは再生で不変で、6更新後はscheduler3506、3回連続評価後もmedium精度・全RNGを復元できた。同seedの両条件では、再生500と追加6の計506バッチのCPU augmentation RNGも一致した。[独立確認](../research/results/robotwin-data-step-continuation-20261114/recovery-preflight-independent-check.json)。旧CPU RNGや画素hashとの直接照合ではない。

13:16 JST時点で、旧3500更新checkpointから4本を改めて起動した。PID436057–436060、出力は元group内 `recovery-v2/seed*`、起動記録は `recovery-v2/recovery_training_launch.json`。preflight6更新を継ぎ足さず、残り2500更新を行って通算6000を目指す。新checkpointは評価前に保存し、RNGとデータ位置も含める。旧ログ・checkpointと、未保存だった旧501–1000更新の記録は残すが、後者を有効な学習履歴には含めない。[watcher v3](../research/watch_robotwin_data_step_continuation_v3.py)もPID436588で起動し、実launchとの一致と元3500の比較を確認。起動時点は `reconstructing_resume_rng`、errorsは空だった。

## 3500更新時点のμ

[保存済み全512組の再集計](../research/results/robotwin-continuation3500-geometry-20261114.md)では、第1主成分比率はseed43既存99.9596%・新規99.9826%、seed44既存99.7956%・新規99.9585%。方向別SNRが1を超える方向も全4本で1のままで、公式は11。元3000と行・交換先が一致し、watcherの介入集計とも一致した。μの分散量とdecoderへの影響は変わったが、追加500更新時点で多方向への回復は確認できない。新規ペアの交換効果の優劣はseedによって逆転している。

## 2026-11-14 14:10 JST時点の保存状況

復旧後の4本は通算4500–4588更新を実行中で、watcherのエラーは0件。6000更新完了とは報告しない。学習・checkpointはSSD1に保存し、このGit snapshotにはコード・設定・集計・調査記録を含める。

[4000更新の同じ512組の再集計](../research/results/robotwin-continuation4000-geometry-20261114.json)では全4本のμ PC1は99.816–99.931%、SNR>1方向数は1のまま。画像予測誤差は改善したが、未投入ペアの優位性はseed間で一致しない。

[最初のattentionの詳細測定](../research/results/robotwin-z-underuse-20261114/first-aggregation64-audit.md)では、開始点3000更新の候補は1つのheadにweighted value分散の81%が集まり、そのhead内PC1が99.43%だった。同じ64例の公式は最大headでも8.85%。現在のforwardで偏りが生じる箇所は分かったが、学習上の原因は未確定である。

ユーザーの指摘を受け、準備済みのg有無比較はGPU preflight・500更新とも開始しない。gを維持した[損失・教師の監査](../research/results/robotwin-z-underuse-20261114/loss-reduction-and-target-audit.md)と、[公式との実行条件の比較](../research/results/robotwin-g-preserving-official-audit-20261114.json)を保存した。batch64の本学習も開始しない。[現時点の原因評価](../research/results/robotwin-collapse-cause-assessment-20261114.md)に確認済み事項と未確定事項をまとめた。
