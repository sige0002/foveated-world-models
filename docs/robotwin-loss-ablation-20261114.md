# RoboTwin z問題：3000更新の損失比較

2026-11-14。ユーザー指示に基づき、batch4・3000更新の短期実験を4本並列実行する。batch64・30000更新の本学習は停止したまま。確認したいのは、公開コードと論文にあるKL係数・画像損失の違いが、z利用の失敗に寄与するか。ユーザーは改めて「解決してから本学習に移る」と指定した。3000更新の完了だけでは本学習へ進まず、zの問題が解決したという評価を先に行う。

## 条件と比較範囲

| 条件 | 画像損失 | KL係数 |
|---|---|---:|
| baseline | SmoothL1（切替幅0.1） | 5e-5 |
| kl_only | SmoothL1（切替幅0.1） | 1e-5 |
| mse_only | MSE | 5e-5 |
| kl_mse | MSE | 1e-5 |

出典の区別：MSE・KL1e-5は[論文Eq.4とAppendix E.6](https://arxiv.org/html/2606.15768v2#A5.SS6)の記載。今回固定した公開コードの[VAE既定値](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/vq.py#L27)はKL5e-5で、公開YAML・配布重み付属YAMLの画像損失はSmoothL1である。後者2つのYAMLはKL係数を明示していない。会話中の「KLを公式値に直した」という表現は、論文と公開コードを区別しておらず不正確だった。KL5e-5をローカルが独自に5倍へ変更した証拠にはならない。また、付属YAMLだけから配布重みの実学習履歴を確定しない。

共通条件はseed42、新規初期化、batch4、warmup1000、最大LR3e-4、AdamW、同じepisode split、同じ順序の画像・augmentation・VAE乱数。失敗した旧batch4 runのwarmup1000を保ち、損失以外の変更を減らす。停止中のbatch64計画のwarmup10000とは区別する。旧runの非同期loaderを、4本とも同じ順序を保証するloaderへ変更するため、旧runとのbitwise再現ではない。

gは状態差分のMSE、係数1、出力2*tanhで固定。4本は損失因子の比較であり、論文完全再現ではない。論文との状態教師・decoder層数・時間幅などの違いは[独立監査](../research/results/robotwin-loss-theory-audit-20261114.md)を参照。

設定：[baseline](../configs/robotwin_loss_ablation_20261114/baseline.yaml)、[KLのみ](../configs/robotwin_loss_ablation_20261114/kl_only.yaml)、[MSEのみ](../configs/robotwin_loss_ablation_20261114/mse_only.yaml)、[両方](../configs/robotwin_loss_ablation_20261114/kl_mse.yaml)。[学習runner](../research/train_robotwin_stage1.py)の任意診断モードを使い、通常学習設定は変更しない。

## 評価・保存

500更新ごとに同じvalidation 512組・同じ別episodeの交換先で評価し、通常z・全例共通の平均z・交換z・z=0を比較する。学習目的の数値はSmoothL1とMSEで単位が違うため、条件間比較には共通の未来特徴MSEと状態MSEを使う。

通常の予測誤差と、平均化・交換による悪化を併せて確認する。zの分散や交換差だけが増えて、通常予測が悪化する結果を成功とは扱わない。μ分散、事後ノイズ分散、VAE前の大きさも記録する。平均zは診断用validation平均であり、別途訓練した定数zモデルではない。

本学習へ進む判断では、少なくとも未使用episodeで正しいzが平均zと別episodeのzの両方より未来予測を改善し、最高精度FP32でも差が残ることを確認する。episode単位の区間が0を跨がないだけでなく、予測誤差に対して実用上の大きさがあるかを公式配布モデルの同条件評価と併せて判断する。z=0だけの悪化、極小の丸め誤差、通常予測を悪化させて作った交換差は修復の証拠としない。候補を選べたら別seedでの再現と状態変化を読む検査を行い、原因に対応した変更かを確認する。

学習更新1/10/100/500/1000/2000/3000では、実際の同じforwardから各損失のIDM出力activationに対する勾配norm・cosineを取得する。追加のoptimizer更新や別の乱数drawは行わない。初期重み、毎更新の入力identity・augmentation・実際のepsilonをhashで照合し、validationは学習用乱数を保護する。

今回の短期実験は3000更新時点のfull checkpointを各1件保存する。500更新ごとの途中記録は評価値で、checkpointではない。本学習の5000更新保存設定は維持する。

## 事前実行と開始記録

4本同時の各6更新＋validationはすべて成功。初期重み・入力・augmentation・epsilonのhashが全条件で一致し、最初の比較は `paired_snapshot`。1本あたりGPU最大割当約13.31GiB、4本の合計約53.25GiB。更新中央値は各2.49–2.53秒で、3000更新は評価・保存を除き約2.1時間。

初回のIDM出力勾配normは、baselineで画像0、g 0.0196288、weighted KL 0.0000301126。KLのみ変更するとweighted KLは0.00000602213で、gと画像は同じだった。公式AdaLNのゼロ初期化で初回の画像→IDM勾配が0になることを実際の学習経路でも確認した。初回でKLがgを圧倒しているとの仮説は支持されないが、その後や学習軌跡への影響はこれだけでは否定しない。

05:03 JSTに各3000更新の実験を開始した。[起動記録](../research/results/robotwin-loss-ablation-launch-20261114.json)。実体はSSD1の `runs/robotwin-loss-ablation-b4-20261114/{baseline,kl_only,mse_only,kl_mse}`。本書作成時点では実行中で、3000更新の結果は未確定。

[集計コード](../research/summarize_robotwin_loss_ablation.py)は初期値・学習stream・評価例の一致を検査し、KL/MSEの単独効果と交互作用を算出する。[読み取り専用の監視コード](../research/watch_robotwin_loss_ablation.py)は同stepの結果が揃うたびに `comparison_step*.json` を保存し、全条件完了後はepisode単位4000回bootstrapを含む `final_comparison.json` を作る。学習の自動再開や追加実験は行わない。

単一seed・3000更新なので、原因を限定する初期検査である。全条件で同じ失敗なら「KLは無関係」と即断せず、KL=0、サンプリングの停止、gの有無などを別の一因子比較として検討する。少なくとも、結果を見る前に長期学習へ戻さない。

## 500更新時点の観測

同じ512組・415episodeの比較で、4条件すべて正しいzと平均z・交換zの未来特徴MSE差は数e-6程度だった。MSEへの変更は通常の未来特徴MSEを約0.534から約0.483へ下げたが、個別zの有用性は確認できていない。KLだけを5分の1にした条件も同様。これは500更新時点での結果で、3000更新までの推移を否定するものではない。

VAE直前の座標標準偏差は各条件約119–128で、公式配布重みでの約122と同程度だった。それでもmuの例間分散は約1.8e-5–2.8e-5と小さく、z交換の利益が見えない。したがって、旧10000更新の巨大な正規化前スケール約19270は、この初期の不使用に必要な条件ではない。ただし標準偏差だけでは方向分布や未来情報量は比較できない。

途中評価のencoder/decoderはBF16である。bootstrapの区間は保存した予測のepisode変動を表し、BF16の系統的な丸めを含まない。最終checkpointでFP32の同条件再評価が必要。

## 1000更新時点の観測

同じ512組で、通常の未来特徴MSEは引き続き改善したが、個別zを交換した影響は極小だった。4条件の学習streamは1000更新まで一致した。

| 条件 | 通常zの未来特徴MSE | 平均z−通常z | 交換z−通常z | mu分散の和 |
|---|---:|---:|---:|---:|
| baseline | 0.423770 | 2.20e-6 | 6.21e-7 | 1.80e-5 |
| kl_only | 0.424024 | 1.86e-7 | 1.14e-9 | 3.73e-6 |
| mse_only | 0.411581 | 2.53e-6 | 1.20e-6 | 1.25e-5 |
| kl_mse | 0.411753 | 3.54e-6 | −7.07e-9 | 1.50e-5 |

KL係数を下げれば単調にmuの分散が増えるという結果でもない。正規化前の座標標準偏差は約370–415、事後ノイズ分散の和は約5.57–6.65だった。この時点も修復とは判定せず、3000更新までの結果と最終FP32再評価を待つ。数e-6の差に関する丸めの注意は500更新時点と同じ。

## 最終FP32評価と次の切り分けの準備

[最終評価コード](../research/evaluate_robotwin_loss_ablation_fp32.py)は、4本が3000更新で完了してから、同じ512組を最高精度FP32のDINO特徴で比較する。公式配布重みも同じ特徴で評価し、平均z・交換z・z=0・IDMに現在を2枚渡したzを、同じ現在入力と正しい未来教師に対して比較する。動きのある例・変化の大きいpatchも分ける。公式gは状態契約が違うので採点しない。[準備・実行記録](../research/results/robotwin-loss-ablation-final-evaluator-20261114/readiness.md)。この評価はモデルを更新しない。

次の一因子比較用として、[独立したv2 runner](../research/train_robotwin_stage1_causal_v2.py)と設定を準備した。既存の実行中runnerは変更していない。

| 設定 | 変更 |
|---|---|
| [baseline_replay](../configs/robotwin_causal_ablation_20261114/baseline_replay.yaml) | v2の対照群。元の損失・samplingを維持 |
| [beta_zero](../configs/robotwin_causal_ablation_20261114/beta_zero.yaml) | KL係数だけ0 |
| [mu_only](../configs/robotwin_causal_ablation_20261114/mu_only.yaml) | 同じepsilonを消費した上で、接続されたmuを両decoderへ渡す。KLは維持 |
| [g_zero](../configs/robotwin_causal_ablation_20261114/g_zero.yaml) | g損失係数だけ0 |
| [official_init](../configs/robotwin_causal_ablation_20261114/official_init.yaml) | 配布encoder/VAE/画像decoderの439テンソルをロード。local gとoptimizerは新規 |

5設定すべてGPUで6更新・検証に通過した。fresh4設定は初期重みが一致し、全5設定の入力・augmentation・epsilonのhashも全6更新で一致。画像/g/log分散に対する各変更の勾配経路を確認した。GPU最大割当約13.32GiB、reserved約13.74GiB。[実測報告](../research/results/robotwin-z-underuse-20261114/causal-v2-preflight-summary.json)。

v2対照群と元runnerの6更新では最大4.71e-4のloss差があった。小さいが原因は未確定で、3000更新後も小さい保証はない。追加の因果比較ではv2のbaseline_replayを3000更新の対照群にする。4本同時の枠はbaselineと3因子へ使う。公式初期化は6更新の確認までで、3000更新の自動起動対象に含めない。ここまでで追加の3000更新は起動しておらず、現行KL/MSE比較の結果を先に確認する。

mu_onlyのlog分散は再構成に使わないためnoise_fractionを修復指標にしない。g_zeroのgは状態予測を学習していないため、その状態MSEも修復判定に使わない。g除去は共有IDMへの信号だけでなくglobal clippingにも影響し得るので、改善しても単一の経路に原因を限定しない。公式初期化の6更新でzへの画像勾配と個別zの効果は残ったが、短い動作確認であり、修復や長期安定の証拠ではない。

## 結果を受けた短期比較の継続

06:16 JST、[継続コード](../research/continue_robotwin_causal_diagnostics.py)をPID138348で起動した。[起動記録](../research/results/robotwin-z-underuse-20261114/causal-continuation-launch.json)、[固定条件v3](../research/results/robotwin-z-underuse-20261114/causal-continuation-contract-v3.json)、[独立検査33件](../research/results/robotwin-z-underuse-20261114/causal-continuation-independent-tests.json)。現在はCPUだけで元4本と最終FP32評価の完了・プロセス終了を待っており、追加の学習は未起動。06:16時点の元4本は1400–1420更新である。

元4条件のいずれかで、平均zまたは交換zによる未来特徴MSEの増加が通常MSEの1%を超え、episode bootstrapの95%区間下限も0を超えたら、候補のレビュー待ちで止める。この基準は追加実験の実行を振り分けるためのもので、修復判定ではない。単一seed・全512組での指標であり、多重比較の補正もしていない。

その候補がない場合に限り、v2のbaseline_replay・beta_zero・mu_only・g_zeroを各batch4・3000更新で一度だけ起動する。元ジョブの終了、GB10の使用状況、統合メモリの空き90GiB以上、SSD1の空き70GiB以上、ソース・設定・評価結果の一致を確認してから進む。実体はSSD1の `runs/robotwin-causal-v2-b4-20261114` に保存する。

起動後は初期値・入力・augmentation・epsilonの一致と進捗を記録し、3000更新後はFP32評価を要する状態で止める。本学習、公式初期化の3000更新、再試行、さらなる実験群を自動起動する機能は含めない。失敗や部分起動時にも既存ジョブを停止せず、二重起動を防いでレビューを要する状態を残す。

## 3000更新完了とFP32評価の復旧

10:32 JSTの確認で、元4本はすべて3000更新を正常完了し、各final checkpointを保存していた。初期重み・全3000更新の入力・augmentation・epsilonは一致した。[3000更新のBF16結果](../research/results/robotwin-loss-ablation-3000-bf16-20261114/analysis.md)では、KL1e-5とMSEの組合せだけ、平均zへの置換による未来特徴MSE増加が通常誤差の1.232%、交換zでは1.855%となり、両方のepisode区間下限が0を超えた。主な増加は2500–3000更新で起きた。他3条件は依然極小の差だった。BF16での候補であり、修復の認定ではない。

旧FP32評価器は、cache経路と公式APIの一致検査で停止していた。継続プロセスも07:47 JSTに評価失敗を検出して停止し、追加学習の起動はなかった。旧ログ・lock・契約は保存している。

[原因の実測](../research/results/robotwin-loss-ablation-final-evaluator-20261114/parity-diagnosis.json)では、baseline・組合せ条件・公式配布重みのすべてで入力特徴とmuは完全一致したが、decoderへ渡す特徴の連続配置と公式4-frame配列のview配置の違いにより、予測値に最大3.34e-6・2.86e-6・9.54e-6の差が出た。公式と同じstrideに戻すと3モデルとも予測も完全一致した。重みは前後hashで不変。

[評価器v2](../research/evaluate_robotwin_loss_ablation_fp32_v2.py)は公式と同じview配置を再構成する。許容誤差1e-6は変更していない。B1/B2/B4の値・stride・storage offsetに対する[独立確認](../research/results/robotwin-loss-ablation-3000-bf16-20261114/fp32-v2-independent-review.json)後、PID321101で全5モデル・512組のFP32評価を再開した。[起動記録](../research/results/robotwin-loss-ablation-final-evaluator-20261114/v2-launch.json)。元と同じDINO特徴cache・評価ペア・交換先を使い、出力は `evaluations/final3000-highest-fp32-v2` に保存する。この追記時点では評価中で、追加学習・本学習は起動していない。
