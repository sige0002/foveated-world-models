# Stage1の潜在flow予測実験

研究ブランチ: `research/stage1-latent-flow-matching`。
目的は未来のDINO特徴の予測精度を改善すること。中間画像の生成や行動チャンクの導入は、この実験の前提にしない。再構成MSEの改善は以下の限定された実験で確認したが、新手法としての独自性、大規模学習・実行時の改善は未証明。

## 元モデル単独との直接速度比較（2026-11-13）

ユーザー依頼により、既存の差し引き概算を元モデル単独の直接測定で置き換えた。CPU2スレッド・float32 highest・inference mode、各条件3回warmup後に12回、順序を交互に変更して測定。同じ現在特徴を入力し、元decoderにはfinal10000のnative posterior mean、改善版には参照42コードを与えた。元decoder、zero-code prior、prior＋固定400補正器を比較した。実行中だった既知の研究CPU学習4件は測定中のみSIGSTOPし、finallyで全件SIGCONTした。既に終了した2件には信号を送っていない。

| batch | 元decoder単独（native code） | zero-code prior単独 | prior＋補正器 | 元decoder比 |
|---|---:|---:|---:|---:|
| 1 | 338.77 ms | 338.17 ms | 361.65 ms | +6.75% |
| 8（batch全体） | 2553.71 ms | 2552.24 ms | 2704.47 ms | +5.90% |

中央値。バッチ1での追加時間は22.88ms。prior再計算とcacheの最大差は両batchとも0。画像特徴抽出・IDM・読み込み・cold startupを含めず、GPUは未測定。全システムの遅延増加率とは呼ばない。[各回の測定結果](../research/results/original-decoder-timing-20261113.json)、[CPU学習の一時停止・復帰記録](../research/results/original-decoder-timing-isolation-20261113.json)。ベンチ実装は既存FFN検証worktreeの `research/benchmark_original_decoder.py` に保存した。研究ゴールの一時停止状態は変更していない。

## 固定IDMコードの分散と追加情報の診断（2026-11-13）

train512の参照コード32次元を調べると、中心化共分散は相対閾値1e-10でrank32、正部分空間の条件数260.94、entropy effective rank12.35だった。上位8方向に分散の80.06%、上位16方向に93.63%がある。平均成分のエネルギーは全コードの1.03%。したがって「全コードがほぼ同じ定数に潰れている」という状態ではない。ただしこの分散・rankは意味情報量や操作の自由度ではなく、decoderのHessian条件数でもない。現在の学習にwhitening等を追加していない。[診断結果](../research/results/condition-code-geometry-20261113.json)、[再現コード](../research/audit_condition_code_geometry.py)。

並列IDM案として、固定z0を必ず残し、追加の独立Gaussian channelに最大1natだけ未来情報を渡す実験を設計した。future query、現在＋z0の再符号化、独立noiseを同じdecoder構造で比較する。single-sample riskを主評価とし、4sampleのloss平均はMonte Carlo推定、予測平均やmean-only推論は1nat保証の対象にしない。これは追加VIB latentに近い既知の構成で、新規手法や情報不足の証明とは呼ばない。[提案と限界](../research/idm/bounded-innovation-idm-proposal-20261113.md)。実装準備中で学習未起動。新headの最終層と追加decoder列を双方ゼロにする期待勾配の停止を避けるため、headを非ゼロ初期化しdecoder追加列のみゼロにする修正を実装前レビューで指定した。

## 凍結した出力LayerNormの幾何的誤差下限（2026-11-13）

元のdecoderの最後のLayerNormは、各tokenの出力を `gamma*u+beta`、`sum(u)=0`、`||u||²<D` に制約する。入力を自由に選べるものとして閉球 `||u||²<=D` に緩和し、正解特徴までの最短距離を計算した。これは予測器ではなく、固定した出力正規化だけに由来する誤差の下限を調べるoracle診断である。内部ネットワークがこの最短点を生成できるとは限らない。

| split | zero-code prior MSE | 出力制約のMSE下限 | prior MSEに対する比率 |
|---|---:|---:|---:|
| train512 | .1022484 | .00231915 | 2.27% |
| dev64 | .1191655 | .00224894 | 1.89% |
| observed96 | .1042186 | .00226324 | 2.17% |

96例のtokenの98.99%で球面制約がactiveだが、下限は現在の予測誤差に比べて小さい。出力LayerNormの幾何的制約そのものだけで大部分の誤差を説明する証拠はない。これはLayerNorm除去による改善幅の上限ではなく、除去が学習・表現に与える影響は別途未検証。

計算は対角重み付き二次目的に対するLagrange dualを使用した。各tokenで主問題の実行可能性とprimal-dual gapを検査し、最大gapは全splitで9.4e-15未満。等方的なケースの閉形式解とも最大1.8e-15で一致した。訓練は行わず、学習中の3条件にも変更を加えていない。[診断コード](../research/audit_frozen_output_norm.py)、[全例の結果](../research/results/frozen-output-norm-audit-20261113.json)。

## 内部condition入力の再学習（2026-11-13、実行中）

元の12層Transformerを凍結し、IDMコードの入力写像だけを `c(z)=c0+Uz` に変更した。Uは1024×32のゼロ初期値で、最初は全コードについて元のzero-code予測と一致する。実データの独立勾配監査でUへの勾配が非ゼロと確認した。これは既知のadapter型設計の検証であり、新規性や精度改善はまだ示していない。

訓練512例・400更新・seed42・共通batch1042で、実コード、訓練平均の定数コード、その定数を訓練コードのRMSノルムに拡大したコードの3条件をCPUで実行中。第3条件は訓練入力のみの監査でRMSノルム4.489対平均ノルム0.456の差を認め、学習開始前に追加した。定数対照の実効自由度やAdamW正則化まで一致させるものではない。4テストと実データpreflightは通過。凍結重み、zero-code出力、データ分割、ソースhashを検査する。

[学習前protocol](../research/results/early-condition512-protocol-20261113.json)、[独立勾配監査](../research/idm/early-condition-gradient-audit-20261113.json)。学習するパラメータは32768だが12層全体を逆伝播するため、軽量学習・推論高速化の証拠にはしない。評価は未来条件付き再構成かつ既観測64+96例である。

## 100論文調査からの設計判断（2026-11-13）

現在の有望案は、frozen current-only予測 `b(X)=decoder10000(X,0)` に、参照差分IDMの情報を使って直接補正を加える方式。主学習は10000で完了し、main val_loss選択bestは9802、finalは10000である。以下はfinal10000を一貫した比較対象とし、評価セットごとにcheckpointを選んでいない。

| 方法 | 補正器のtrain例／更新数 | seed範囲 | dev64 MSE | observed96 MSE |
|---|---|---|---:|---:|
| 元のfinal10000 | 元のmain学習 | 既存checkpoint | .118654 | .103631 |
| 参照差分IDM＋直接補正 | 128／400 | head/decoder42–44平均 | .103624 | .090697 |
| 参照差分IDM＋直接補正 | 512／400 | head/decoder42–44平均 | **.099222** | **.086709** |
| 定数z＋同容量補正器 | 512／400 | decoder42–44平均 | .108919 | .095100 |
| 参照差分IDM＋直接補正 | 512／1600 | head/decoder42のみ | **.095954** | **.083304** |
| Flow補正・4標本平均 | 512／400 | head/decoder42のみ | .113575 | .100645 |

固定400更新・3 seed平均の既存final比改善は96例で**16.33%**、128→512の追加データ効果は同じ3 seedで**4.40%**。固定1600更新のseed42は既存final比**19.61%**低いが、3 seed平均ではなく追加更新費用もある。[データ規模・更新数・seed別集計](../research/results/nested512-direct-scaling-summary-20261113.json)。512/400の定数z対照に対する3 seed平均の追加改善は8.82%、paired差95%区間[−.010807, −.006293]。[同容量対照](../research/results/nested512-controls-summary-observed96-20261113.json)。**512例では定数z補正自体も元モデルを改善するため、128例で得た「容量追加だけでは改善しない」という結果を一般化しない。**

Flowの512/400における個別標本平均MSEは.110110、4標本平均は.100645で、同じseed42の直接補正.085574を超えない。固定endpointの512/1600比較も完了し、96例で直接補正.083304、定数z補正.102540、flow個別標本.112929、flowの4標本平均.100131だった。同じseed42の直接補正に対してflow平均は20.20%高いMSEとなり、今回の設定では追加更新でも差は埋まらない。更新数は揃えているがFLOPs一致ではなく、全flow方式の否定ではない。[1600更新の対照集計](../research/results/nested512-fixed1600-controls-summary-20261113.json)。[flow追加更新](../research/results/nested512-flow1600-protocol-20261113.json)、[定数z追加更新](../research/results/nested512-constant1600-protocol-20261113.json)。128例で学習した固定flowでは、別乱数16標本平均でも.105757対直接.089675だった。iid標本を仮定する `B_hat=(N*loss(mean_N)-mean(loss(sample)))/(N-1)` による平均予測リスク推定は4標本で.104932、16標本で.104955。無限標本の実測ではないが、その固定モデルでは生成分散の削減だけで差を埋めにくい。[16標本監査](../research/results/final-prior-flow-mc16-vs-direct-20261113.json)。

全て正解未来から得るposterior zによる再構成で、実行時の行動条件付き予測ではない。encoder2900・prior10000は共有、headと校正は128例で固定したまま補正器だけを512例に拡張した。seed平均はモデルごとのリスク平均であり予測ensembleではない。主validationは全160エピソードで64+96の全てを既に研究観測済み、独立final testではない。区間は学習済みモデルに条件づけたepisode bootstrapで、学習・選択・適応的再利用の不確実性を含まない。追加head学習、参照query計算、補正器を含むため元モデルと同じ総学習費用ではない。操作意味、カメラ不変性、新規性、トップカンファレンス相当の価値は未確立。

以前の6501基準の14.85%や、128例final基準の12.48%は当時の比較として保存している。[6501基準](../research/results/prior-correction-unused96-summary-20261113.json)、[128例final基準](../research/results/final-prior-correction-summary-20261113.json)。96例を初めて使った時点の「研究未使用」という記述は歴史的な選択状況であり、現在の評価を未使用testと呼ぶ根拠にはしない。

[調査CSV](../surveys/flow-architecture/survey.csv)に100件を統合した。全件で本文に基づく抽出とホストによる主要手法・結果・限界の原文照合が完了。うち4件は同じ版の既存調査を照合して再利用した。flowそのものに加え、条件付き予測、表現空間、物理予測、最適化の基礎論文を含む。各論文の全式を独立証明したという意味ではない。検索・選定は目的に沿ったもので、網羅的な系統的レビューではない。書誌情報の不整合や理論の適用条件は各行に残した。

現時点の推奨は、**強い直接MSE予測器を基準に、IDMが提供する条件情報を改善する**こと。flowの置き換えだけで大幅なMSE改善を得るという期待は、現在の実験では支持されない。Transformerはネットワーク構造、flow matchingは学習・生成の枠組みであり、二者択一ではない。比較ではTransformerの容量と入力経路を揃え、目的関数と推論方式の差を測る。

判断の理論的理由は目的の違いにもある。条件Cを固定し、生成乱数が正解Yと条件付き独立なら、二乗誤差は `E||Y−Yhat||²|C = Var(Y|C) + ||E[Y|C]−E[Yhat|C]||² + Var(Yhat|C)`（分散はtrace）に分解できる。直接MSE回帰は条件付き平均を狙い、flowは条件付き分布の生成を狙うため、単一標本MSEの改善は自動的には得られない。ただし平均化後にも今回の差が残るので、生成分散だけが原因とは言わない。[Flow Matching原論文§3](https://arxiv.org/html/2210.02747v2)の保証は確率経路と速度場についてであり、有限容量でのpaired endpoint MSE優位の保証ではない。

| 優先度 | 設計・仮説 | 根拠と現在の判断 | 必須の比較 |
|---|---|---|---|
| 1 | 現在特徴を直接保持し、全特徴次元を出せる残差head | 狭い速度出力には初期ノイズが残る構造的制約を確認。広いheadでは直接予測も改善し、同容量flowより低MSE。これは既知の設計改善で新規手法ではない | 同じhead・初期重み・凍結zで3 decoder seedを完了。全seedで直接回帰が優位 |
| 2 | IDMの追加情報を、現在から予測できない実測操作変位で評価・学習 | 参照差分IDMは3 seedで再構成を改善したが、カメラや操作の分離は未証明。条件付き平均を引くと、正しい基準予測の下でcurrent-only codeの追加利得は0 | 再構成のみ、通常の変位教師、条件付き残差教師。同じラベル量・容量で実測変位と未来特徴MSEを評価 |
| 3 | 初期条件からのMSE branchでIDMを学習し、flow側の未来補間からencoderへの勾配を分離 | 補間状態がzの追加情報を代替する情報論的仮説。encoder崩壊や改善の証明ではない。現在の固定IDM実験だけでは判定不能 | 同一branchでflow勾配あり／停止／直接回帰のみ。学習費用とパラメータ数を揃える |
| 条件付き | 直接予測後の少数回補正 | PDE-Refinerは長期rolloutを改善するが、今回の4回決定論的補正・source anchoringは直接回帰を明確に超えていない | 同じ計算量を使う深い1回予測、自己生成入力での補正、原座標MSE。GT補間上の改善は採用根拠にしない |
| 保留 | 平均予測＋残差flow、consistency/蒸留、solver変更 | 分布の多様性や速度には根拠がある。正しい平均にゼロ平均残差を加えてもMSEは改善しない。solverは学習済み場の誤差を直さない | 平均branchのみ、同容量直接予測、同じ総NFE。best-of-Nを主指標にしない |
| 非採用 | z復元だけのcycle、z交換gapの直接最大化、無条件なカメラ独立化 | 既存手法との重複、情報埋め込みや背景依存での達成、行動と頭部運動の相関を消す危険がある | 操作の実測情報・同じ物理遷移の視点介入なしには操作意味を主張しない |

直接関連する[FlowWM](https://arxiv.org/abs/2606.29059v2)のmode coverage／best-of-3の改善はpaired MSEの証拠ではない。[DYffusion](https://arxiv.org/abs/2306.01984v2)は海面温度で平均MSEがDropout対照より悪く、[CorrDiff](https://arxiv.org/abs/2309.15214v4)も変数によってensemble平均MAEが直接回帰を超えない。一方、[PDE-Refiner](https://arxiv.org/abs/2308.05732v2)の長期安定性改善は、反復補正を評価する理由になる。これらの異なる評価設定を一つの「拡散は高精度」という結論にはまとめない。

### 空間差分branchのアーキテクチャ検証

現在特徴の各パッチで横・縦方向の中心差分を取り、予測した2係数でその方向の補正を足す。自由な768D残差headも残し、特徴差分だけでは表現できない変化を制限しない。係数は各軸 `0.5*tanh`、外周パッチは補正0とする。追加1538パラメータ。DINO特徴が画像移動に対して同変であるという保証はなく、局所移流の一次近似を学習上の仮説として使う。

先行oracle診断では、direct1600の残差に正解未来を使って係数を合わせると96例MSEが約7.95%低下した。これは表現可能性であり、学習可能性・予測性能ではない。[oracle記録](../research/results/spatial-tangent-after-direct1600-20261113/results.json)、[anchor表記の補足](../research/results/spatial-tangent-after-direct1600-20261113/anchor-clarification.json)。

512例・400更新・seed42の固定endpointを完了。同じ2列を共通の符号付きchannel permutationで変換した対照を置き、各パッチのGram行列・branch容量・初期共通重みを揃える。現在特徴と既存posterior zから係数を予測し、oracle係数・未来targetを入力しない。直接残差とこの対照の両方を超えるかで判断する。3単体テストで初期重み・出力等価性、線形空間場の差分と境界、対照Gram一致、係数勾配を確認した。[実験プロトコル](../research/results/spatial-tangent-learning-protocol-20261113.json)、[実装](../research/spatial_tangent_decoder.py)。空間変換による予測自体は既知であり、このbranchだけを新規性としない。

結果はdev64で直接.098379→tangent.098166（rotated.099064）、observed96で直接.085574→tangent.085305（rotated.085785）。直接比0.314%、rotated比0.560%の小さい改善にとどまる。96例のpaired差95%区間は直接比[−.0005474, −.00000166]、rotated比[−.0008303, −.0001427]で、学習済み単一seedに条件づけた区間にすぎない。わずかな差を大幅改善やseed頑健性の証拠とはしない。oracle約8%の表現余地と学習したbranchの改善を区別し、この結果だけで主方式に採用しない。[固定400結果](../research/results/spatial-tangent512-summary-20261113.json)、[独立実装監査](../research/results/spatial-tangent-learning-review-20261113.json)。

### 予測残差の128次元部分空間

VFMFとRLA-WMの圧縮空間でのflowを踏まえ、学習512例の `future−current-only prior` の非中心化二次モーメントから固定128D基底を作った。これはIDMの32Dを変える実験ではない。元の768D現在特徴とposterior zを条件に保ち、ノイズ・予測補正・flowの速度とteacher endpointを `baseline+Range(P)` に揃える。自由なネットワーク容量は全方式13,410,048で共通、basisは学習しない。次元削減の効果を切り分けるため、ネットワーク自体の計算削減は行っていない。[実装](../research/projected_prior_decoder.py)、[プロトコル](../research/results/projected-prior512-protocol-20261113.json)。

基底は評価前に学習データだけから保存し、rank128を固定した。保持する残差エネルギーは約76%、oracle再構成誤差はdev64 .027937／96 .024739。これは予測精度ではない。[圧縮診断](../research/results/prior-subspace512-20261113/results.json)。VFMFはPCAより非線形VAEが良いことも報告しており、今回の線形対照はその再現や否定ではない。

同じ400更新・seed42で投影directと投影flowを比較する。directの元空間MSEには固定の捨てた成分の誤差が含まれるが、flowの速度損失には含まれないため、学習loss値ではなく元の768D終点MSEを比較する。active方向のnoise標準偏差は.1のまま、総分散はfull flowの1/6になる。単一標本誤差の低下だけを学習改善としない。full-rank同値性、直交誤差分解、基底検査の3テストと独立レビューを完了。[レビュー](../research/results/projected-prior512-review-20261113.json)。

既存full flowのmean4とdirectの誤差差分の83.08%は保持する128D内にあり、捨てる方向だけでは不利を説明できない。既存flow予測を後から射影するとMSE .100645→.099554、既存directは.085574→.087033へ悪化する。これは学習なしの診断で、元のdirectは捨てる方向にも有用な補正を学んでいる。[固定モデル誤差分解](../research/results/projected-prior512-original-decomposition-20261113.json)、[差の集計](../research/results/projected-prior512-gap-decomposition-20261113.json)。

固定400の結果は96例で、投影direct .089030（full direct .085574、4.04%悪化）、投影flow単一標本 .104409（full .110110、5.18%改善）、投影flow mean4 .101560（full .100645、0.91%悪化）だった。投影flow mean4は同空間directにも14.07%劣る。4標本からの条件付き平均予測リスク推定 `(4*mean4−sample)/3` はfull .097490、投影 .100610で、平均化の不足だけでは説明しにくい。無限ensembleの実測値ではない。[比較集計](../research/results/projected-prior512-summary-20261113.json)。

投影学習後の誤差分解でも、直接の保持空間誤差.064291、flow mean4 .076821、双方の捨てた成分は.024739となった。独立の評価・分解経路でMSEが1e−10以内に一致した。**固定PCA128は精度改善案として棄却する。** 単一標本の改善はあっても平均予測は改善しない。これは非線形圧縮・別のデータ規模・別の学習設定全般の否定ではない。[投影学習後の分解](../research/results/projected-prior512-trained-decomposition-20261113.json)。

### 物理時間幅を明示する対照

データは末尾の短いpairも含み、学習512中79、dev64中11、observed96中27が1.2秒未満だった。datasetはdtを返すがproduction `_compute_step` はlamに渡さない。そこでflowの生成時刻とは別に、`r=dt/1.2` から `(r−1,r²−1)` を作り、zero-init・biasなしの2→128射影で条件に加えた。追加256パラメータ。共通初期重みとglobal RNGを維持し、1.2秒では追加入力が0になる。[実装](../research/horizon_prior_decoder.py)、[固定400プロトコル](../research/results/horizon-prior512-protocol-20261113.json)、[監査](../research/results/horizon-prior512-review-20261113.json)。

固定400・seed42の96例で、直接は.085574→.085704（0.152%悪化）、flow mean4は.100645→.100632（0.014%改善、paired区間は0を跨ぐ）。短時間27例のflowは.077547→.077337（0.271%改善）だが、nominal69例は.109684→.109747へわずかに悪化した。horizon射影weight normはdirect .526、flow .328で、条件経路が未学習だったわけではない。現在のflow/direct差を埋める主要な改善にはならず、この設定を主方式として採用しない。[結果](../research/results/horizon-prior512-summary-20261113.json)。

current-onlyの別対照では、`b+(1−r)(current−b)` の単純補間は96例.104219→.104245で改善せず、trainのみで曲率cを最小二乗した `b+[(1−r)+c*r*(1−r)](current−b)` はc=−1.505969、.102770となった。nominalは不変、短時間だけに約6.66%の改善があるが、学習済み直接予測を超えない。[校正対照](../research/results/horizon-prior-calibration512-20261113/results.json)。

ここでdtは指定されたpairの予測時間として扱う。未来の未知の終端を使って固定1.2秒forecastを短縮してよいという意味ではない。短いpairはepisode末尾と相関するため、改善を一般的な時間力学の獲得とは呼ばない。wrong-z gapも異なる時間幅との不整合を含む。2単体テストと独立監査を実施。初期AdaLNがzeroなので初回条件勾配は0だが、3更新後にhorizon経路へ勾配が流れることを確認した。学習更新とvalidationは既存と同じで、今回から最終train-set推論だけを省略できる任意flagを用意し、両新armに同じ設定を適用した。デフォルト挙動は従来通り。

### 非線形残差codecと、その空間での予測

線形PCAだけでは不十分というVFMF/RLA-WMの知見を踏まえ、current-only prior残差のPCA経路に、現在特徴とprior差分を条件とするMLP encoder/decoder adapterを加えた。adapter末尾をzero-initにして同rank PCA復元から開始し、512例・1000更新・seed42で固定した。primaryは各patch32ch（256patchで8192値）、比較用は128chで、既存global IDM z32とは別のw表現である。VAEでもRLA-WMのTransformer再現でもなく、圧縮可能性を調べる小型モデル。[codec仕様](../research/results/nonlinear-codec512-protocol-20261113.json)、[実装](../research/conditional_residual_codec.py)。

96例の達成復元MSEは32chでPCA .049088→.038631、128chで.024739→.021107。zero-code復号はそれぞれ.102934/.103600で、current-only補正だけで復元改善を説明する結果ではないが、zero-codeは独立に学習したcurrent-only対照ではない。encoderが正解未来を見るので、これを予測精度と呼ばない。非線形decoderでは達成復元誤差を厳密な誤差下限とも呼ばない。[codec結果](../research/results/nonlinear-codec512-summary-20261113.json)。3単体テスト・独立レビューを完了し、3split間のepisode非重複と同じfinal10000 priorのhashも確認した。実行済みtrainerのsourceは[snapshot](../research/source_snapshots/train_conditional_residual_codec-executed-20261113.py)に保存し、現trainerには後から同provenance検査を追加した。

続いてprimary32 codecを凍結し、currentと既存posterior IDM zだけからwを予測した。学習wは教師として使うが、dev/96のw cacheは読まず、評価時にfuture encoderを呼ばないことをテストした。train-only mean/stdでwを正規化し、同じ幅128・2層・721,696学習パラメータ・seed42・400更新で4方式を比較した。codecは673,184固定パラメータで、追加1000更新を要する。strong full-space correctionの13,410,048パラメータとは同容量ではない。[予測プロトコル](../research/results/nonlinear-codec-predictor512-protocol-20261113.json)。

| 32ch codec上の予測方式 | dev64 feature MSE | observed96 feature MSE |
|---|---:|---:|
| 復号後feature MSEで直接学習 | .105165 | .092886 |
| 正規化latent MSEで直接学習 | .105000 | .092000 |
| Flow σ=1、復号後4標本平均 | .116843 | .101735 |
| Flow σ=.1、復号後4標本平均 | .121832 | .105950 |

どれもstrong full-space direct .085574を超えず、最良の直接方式でも7.51%高いMSEだった。flowの『復号後平均』と『latent平均を復号』も分けて記録した。低い復元誤差を予測改善へ読み替えず、この固定小型構成を採用しない。一方、full-space方式との容量差、および新predictorはbaselineを復号時だけに使い、strong correctionはstate/raw経路にも使うという入力経路差が残るため、非線形圧縮一般の失敗とは結論しない。[予測結果](../research/results/nonlinear-codec-predictor512-summary-20261113.json)。

初回4実行は古い128行にendpoint_indicesがないため、更新前の照合でexit1になった。episode/index/dtと任意endpoint_indicesを、source/codec checksumと併せて照合するよう修正し、実512行・正規化値の完全一致と4テストを確認した。失敗logs/sourceを保存し、別のv2ディレクトリで再実行した。[schema互換性修正記録](../research/results/nonlinear-codec-predictor512-row-amendment-20261113.json)。評価はすべて再使用済みデータで、既存IDM zが未来条件付きである点も変わらない。

### 非線形codecの入力条件・容量対照

[事前固定した5群](../research/results/codec-context-capacity512-protocol-20261113.json)が全て正常終了した。現在特徴から作った凍結基準予測を条件入力に追加し、小型128幅・2層と大型384幅・4層で直接回帰とflowを比較した。codec32は固定1000更新、予測器はseed42・512訓練例・固定400更新。大型には復号後の特徴MSEを直接学習する対照も含めた。

[結果と対応例差の集計](../research/results/codec-context-capacity512-summary-20261113.json)：

|方式|観測済み96例のMSE|
|---|---:|
|小型・潜在MSE直接回帰|0.090569|
|小型・flowの4標本予測平均|0.100313|
|大型・潜在MSE直接回帰|0.085270|
|大型・flowの4標本予測平均|0.095421|
|大型・特徴MSE直接回帰|0.086690|
|既存の全特徴空間直接回帰400更新|0.085574|

基準予測の入力追加だけで小型直接回帰は1.55%、小型flowは1.40%改善。さらに大型化すると、それぞれ5.85%、4.88%改善した。ただし大型の同構成間でもflowは直接回帰より11.90%悪い。差0.010151の固定モデルに条件づけたpaired bootstrap95%区間は[0.008407,0.012080]。大型codec直接回帰と既存全特徴直接回帰との差は−0.000304、区間[−0.000821,0.000218]で、明確な優位はない。非線形圧縮自体を失敗とする結論は修正し、既存方式とほぼ同等に到達したと扱う。

同一容量内では初期共通重み・batchが一致する。容量間では初期化の乱数消費が異なるのでflowの時刻・noiseまで一致するとは主張しない。学習可能パラメータは小型820,000、大型11,417,120、別途固定codec673,184。既存全特徴直接回帰13,410,048とは容量・FLOPs・codec事前学習費用が一致しない。全て未来条件付きIDMによるposterior再構成で、再利用した評価例・1seed。bootstrapは学習や適応的モデル選択の不確実性を含まない。

現段階ではflowへの置換を採用しない。次はIDMの情報抽出とdecoderによる利用を分離し、両者の適応で既存の直接回帰を超えられるかを検証する。既知のjoint学習そのものを新規性とはしない。

### 次の対照：IDMと予測器の局所適応を分離

[論文と実装に基づく追補仕様](../research/idm/future-information-utilization-minimal-experiment-addendum.md)と[固定実験計画](../research/idm/local-repair512-protocol-20261113.json)を保存した。既存の400更新direct correctionとreference42 headを共通の開始点とし、H（headのみ）、D（decoderのみ）、HD（両方）、K（train平均の定数codeでdecoderのみ）を各400更新・3minibatch seedで比較する。全群でpretrained重みは共通なので、独立な事前学習seedの追試ではない。元の400更新からの継続であり、既存の最良1600更新モデルの性能を上回ったとはまだ主張しない。

[独立した開始点監査](../research/idm/local-repair-start-audit-20261113.json)で512/64/96例のrow対応・全episode集合の非重複・final10000 prior共有・head/calibration/checkpoint・有限性を確認。固定CPU highest/B8でheadから再計算したμを使う初期MSEはtrain0.080275635、dev64 0.098379539、観測済み96例0.085571649。以前のcached μを同じdecoderで使った96例0.085573666との差は約−0.000002。μ自体の最大差は0.01207あり、同一bitの再現とはしない。初期評価を根拠に精度設定を変更しない。

H/HDはμ経路の視覚MSEと、固定した元posterior varianceを使うKL drift（係数0.001）を最小化する。元headのsampled MSE＋standard-normal KLの継続ではない。平均train driftが1natを超えたら局所適応としての範囲を外れたと扱う。将来のGaussian学習、human動画への転移、実行可能なaction-conditioned予測は別検証が必要。[専用trainer](../research/idm/train_local_repair.py)と5合成テストが通過し、[sourceを固定した実行記録](../research/idm/local-repair512-launch-20261113.json)を保存してseed42の4群を起動した。各群は同じ学習済み重みから新規AdamWを作る。[途中集計](../research/idm/local-repair512-summary-20261113.json)ではD（予測器のみ）の観測済み96例MSEがseed42 0.082191、seed43 0.084293で、共通F 0.085572を下回る。Kは0.094103／0.095574、HDは0.081780／0.084557だが両seedともtrain平均driftが事前上限1natを超え、採用条件不成立。H42も0.085213・drift超過。残りの群は実行中で、3seedの結論は未確定。

### IDM局所適応12群と独立評価の完了

全12群が固定400更新で正常終了した。[集計](../research/idm/local-repair512-summary-20261113.json)の観測済み96例における3モデルのMSE平均はH 0.085189、D 0.083170、HD 0.082970、K 0.094909。共通Fは0.085572。これは予測を平均したensembleではなく、各modelのriskの平均である。H/HDは全6群でtrain平均driftが1natを超過したため採用条件を満たさない。Dの改善は固定コードの追加利用と整合するが、headが最適・情報が十分という証明ではない。

[独立した物理変位probe](../research/idm/local-repair-probes-20261113/results.json)も完了。train-only CV・scales・foldsを固定し、current-onlyの過去結果を再現した。96例のnormalized MSEはcurrent-only0.402629、F0.275594、H42/43/44が0.278322／0.275878／0.278045、HDが0.274707／0.268663／0.271811。Dのcodeは全splitでFと完全一致。Hの強い未来依存や小幅な視覚改善は、独立した効果予測の改善には直結しなかった。HDのprobe改善も、既観測データ・softdrift逸脱・既知readoutによる限定的結果であり、一般的なaction表現改善として採用しない。

[同じcurrentでfutureを交換する診断](../research/idm/local-repair-swaps-20261113/results.json)も全指定modelで完了した。交換は64例内の既存cyclic対応から指定8例を使用し、元headのvariance関数を固定した。future-swap KLやdecoder感度はcamera等を含む未来依存の指標であり、実行可能なaction counterfactualやcamera分離の証明ではない。[soft制約の失敗とhard-bound候補](../research/idm/local-repair-soft-drift-and-hard-bound-addendum.md)は別途整理した。

### LaWMの探索範囲を速度と内部構造へ拡張

ユーザーの追加方針に従い、flowへの置換に限定せず、Transformer内部、SSM、小型事前学習モデルなどを候補に含める。SLMの意味は確認中であり、SSMと同義とは扱わない。既存の100件はflow関連を中心とした調査で、この追加候補群の調査完了を意味しない。

[実行されたmodule形状からの演算量監査](../research/results/decoder-dense-macs-20261113.json)では、研究用13.41Mパラメータdirect correctionは1例あたり約2.579G dense MAC。768幅の出力側Transformer blockが約74.3%、全attentionの二次的token相互作用部分が約5.2%だった。bias・softmax・正規化・メモリ転送・凍結prior・vision encoderを含まない。これはproductionの12層decoder全体の計測でも、実時間のprofileでもない。二次attentionだけを減らせば大幅高速化する、とは現状では期待しにくい。

次の順序で検証する。

1. 現行AdaLNBlockはattention weightを使用しないがMultiheadAttentionへneed_weights=Falseを渡していない。PyTorchの[公式資料](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)に沿い、同じ重みでSDPA経路の出力・勾配・実時間を比較する。実装効率の対照であり新規手法ではない。実行中の比較trainerの共有コードは変更しない。
2. 演算量が大きい出力blockのFFN拡張率と幅を分けて比較する。特にFFN縮小は全768出力の自由度を維持できる一方、head全体の低次元化は出力補正の表現力を制約し得る。両者を同時に変更しない。
3. 全体相互作用と局所的な変化を分ける候補として、少数のglobal tokenと局所gated convolutionの混合を比較する。全体カメラ運動も予測に必要なため、局所性だけに限定しない。既存multiscale/Perceiver/gated CNNとの重複を調べてから新規性を判断する。
4. [Mamba-2](https://arxiv.org/html/2405.21060v1)はSSMとattentionの関係および効率的実装を示すが、現在の短い256空間tokenの一段予測で速い・正確とは保証しない。[MambaOut](https://arxiv.org/html/2405.07992v1)の視覚タスク比較も踏まえ、SSMのみでなくgated CNNとattentionを対照にする。実際の時間履歴を入力する実験は、現cacheの端点対だけの実験と区別する。

独立worktreeで同一重みのattention経路を比較した。[数値同等性の結果](../research/results/attention-backend-equivalence-20261113.json)はCPU2・float32 highest・B2で推論出力最大差0、学習出力最大差1.43e-6、パラメータ勾配の相対L2差2.69e-7。元モデルの重み・共有sourceを変更していない。小さい入力集合での許容誤差内検証であり、全入力のbitwise一致や学習後精度の保証ではない。[速度測定条件](../research/results/attention-backend-prebenchmark-20261113.json)を保存し、競合学習の終了後に[隔離CPU速度比較](../research/results/attention-backend-timing-summary-20261113.json)も完了した。推論中央値は基準39.424ms／候補39.487ms、forward/backwardは122.058ms／122.034msで、実質的な高速化はない。このCPU条件では速度改善の根拠として採用しない。GPUや全体パイプラインの速度は未計測。

[FFN比較の固定計画](../research/results/slim-ffn512-protocol-20261113.json)と専用worktreeの候補を準備した。768幅headのFFNだけを3072から1536へ縮小し、全768次元出力・attention・raw current/prior経路を保持する。共通初期重みとRNG、初期prior一致、FFNへの勾配を2テストで検証。削減パラメータは2,360,832、dense MACの計算上の削減は23.42%。既存FFN4とFFN2を同じreference42コード・512例・400更新・decoder seed42/43/44で比較する。形状が異なるFFNはnative初期化なので、その部分の初期重みが同一とは主張しない。seed42/43の4群をCPU4ずつ起動した。FFN4 seed42の100更新時の訓練MSEは既存対照と完全一致し、新trainerの共通経路を確認した。速度目的だけで精度目標を置き換えず、精度と全体推論費用を併記して採否を決める。

速度は同じ装置・精度・batchで学習と競合しない条件で測る。候補単体と凍結prior等を含む全体を分け、MSEと実測時間の両方を報告する。候補編集は独立worktreeで行い、現在の学習済み重みと未commitの研究sourceを正確に固定して比較する。

### FFN縮小6群と前段込みの速度測定

[FFN4/2×3seedの固定400更新](../research/results/slim-ffn512-summary-20261113.json)が全て正常終了した。FFN4 seed42は過去対照の全4時点の訓練lossと、最終train/dev64/観測96 MSEを再現。3modelの観測96 risk平均はFFN4 **0.0860523**、FFN2 **0.0866184**で、縮小版が0.658%悪化した。各seedでも悪化し、平均の対応例差の条件付き95%区間は[0.000357,0.000796]。固定モデル・既観測例の記述比較であり、学習や適応的選択の不確実性を含まない。

競合学習がない状態での[CPU2速度測定](../research/results/slim-ffn-timing-20261113.json)は、補正器単体でB1 21.525→16.985ms、B8 153.189→119.877ms（batchあたり）。ただし凍結current-only priorを毎回計算するとB1 **351.528→347.107ms**、B8 **2660.951→2631.970ms**で、短縮は約1.26%／1.09%に留まる。DINO encoderとIDMは含めていない。GPU速度は未評価。prior再計算とcached baselineの最大差はB1/B8とも0。[計測条件・source](../research/results/slim-ffn-timing-provenance-20261113.json)も保存した。

したがってFFN縮小は補正器の速度と小幅な精度低下のトレードオフであり、精度改善の主案としては採用しない。次は全体で支配的なcurrent-only priorを軽量化する。例えば大きな凍結prior b(X)を小型g(X)へ蒸留し、既存の補正器を固定したまま最終MSEを評価すれば、teacher近似誤差と補正器がそれを増幅する影響を分離できる。教師出力に近いだけで最終精度が保たれる保証はない。

この案は[知識蒸留](https://arxiv.org/abs/1503.02531)という既知の枠組みを用いる。world-model蒸留にも[TWIST](https://arxiv.org/abs/2311.03622)など先行例があり、単なる蒸留を新規性とは主張しない。現時点は次の検証候補であり、学習は未起動。元の目標である大幅なMSE改善は引き続き未達で、速度目標だけに置き換えない。

### Current-only priorの圧縮比較を開始

[固定計画](../research/results/prior-compression512-protocol-20261113.json)に基づき、現行current-only prior b(X)を小型g(X)へ置き換え、400更新の補正器D(X,Z,b)を固定する対照を開始した。gは現在特徴だけを受け取り、teacher prior・未来特徴・IDM codeは推論入力に含めない。教師への近似誤差、g単体の未来MSE、D(X,Z,g)の最終未来MSE、元D(X,Z,b)との予測差を別々に報告する。最終補正には未来条件付きZを使うので、最終MSEは従来どおりposterior reconstructionである。

最初の[train-only線形teacher fit](../runs/linear-prior512-20261113/results.json)は、全train512×256patchのfloat64共分散と固定ridge0.01で算出した。観測済み96例のteacher-prior近似MSE0.033789、g単体の未来MSE0.136207、最終MSE **0.115335**。元の最終0.085574より悪く、今回の線形置換は不採用。線形モデル全般の最良値や圧縮不能性を証明するものではない。

[非線形3群](../research/results/prior-compression512-launch-20261113.json)は同じ13.41Mパラメータ・初期値・batch・seed42・400更新で、(A) ||g−b||²、(B) ||g−Y||²、(C) ||D(X,Z,g)−D(X,Z,b)||²+0.1||g−b||²を比較する。gはcurrent residualから始め、初期出力は現在特徴と一致する。Cは小さな誤差δ=g−bについて局所的にδᵀJ_DᵀJ_Dδを測るので、補正器が増幅する方向を重視する動機がある。ただし局所近似であり、汎化や真の未来MSE改善を保証しない。Cは凍結Dの追加forward/backwardが必要で、同FLOPs比較ではない。

推論入力の制限、初期出力、線形fitの回復、凍結補正器を通るstudentへの勾配を3テストで確認した。学習中の共有sourceは固定し、独立worktreeで実行している。追加の新規性は未主張で、大幅なMSE改善は依然未達。

### Prior圧縮400更新の結果と次の利用経路

[全群集計](../research/results/prior-compression512-summary-20261113.json)では、teacher-priorを学ぶ非線形studentの最終MSEは0.108358、正解未来を学ぶstudentは0.122881で、元の0.085574を維持できなかった。両群は400更新で正常終了し、補正器不変・source・batch・初期値の監査が通過した。補正後のteacher予測差／prior近似誤差の平均MSE比は0.959／0.936であり、平均誤差の大幅増幅よりも、priorの近似誤差自体が大きい。これは平均的な観測値で、全入力のLipschitz保証ではない。

正解未来studentはprior単体の未来MSEがteacher-fit studentより低いが、固定補正器を通した最終MSEは悪い。固定補正器が元priorに適応している影響を含み、再適応すれば改善不能という結論ではない。補正後の一致を学ぶ3群目も正常終了し、最終MSE0.107290、teacher-prior近似MSE0.024732、最終teacher予測差MSE0.022962だった。通常のteacher fitより小幅に改善するが、元モデルの精度を維持できていない。今回の512例・400更新・単一seedの圧縮recipeは不採用。圧縮全般の限界や、追加データ・学習時間での改善不能性は示していない。

別の利用経路として[既存12層decoder内部への条件adapter](../research/idm/early-condition-adapter-readonly-audit.md)をread-onlyで監査した。元decoderの視覚知識を凍結し、action入力をc0+Uz（U:32→1024、biasなし、zero-init）へ置き換える。初期は全codeでzero-code priorと一致し、z=0なら学習後もそのpriorを保存する。元の出力補正器を増設する方式と注入位置が異なるが、既知FiLM/adapterの範囲で、新規性や精度向上は未証明。元action projectionの条件部分空間から外れる点、定数code対照の実効容量差、凍結12層のbackward費用を明示した。現時点は仕様のみで、実装・学習は未実施。

### 最も検証価値がある研究仮説

データ規模の監査では、128例の400→1600更新でtrain MSEが.062047→.022347へ下がる一方、96例は.089675→.098438へ悪化した。512例では同じ固定更新数の比較が.085574→.083304となり改善した。小データでの追加更新の結果だけでアーキテクチャの限界とは判定しない。[128例更新数対照](../research/results/final-prior128-400-vs1600-20261113.json)。512cacheは旧128の全fieldと検証64のbytehashを保持し、追加384のhead再計算が誤差0、評価160とのepisode非重複を独立検証した。[選択・比較プロトコル](../research/results/nested-train512-comparison-protocol-20261113.json)、[cache監査](../research/idm/nested512-independent-audit.md)。

操作情報のprobeでも学習を512例へ増やした。全256パッチとcurrent stateを使うtrain-only選択の空間RBF対照に対し、zの追加で共通旧128 scaleの8D変位MSEはdev .420013→.321819、96 .402629→.316632（23.38%／21.36%改善）。右gripperは128時点より改善したが、512のcurrent-onlyにはまだ劣る。問題の大正変位の学習supportは3→18例に増えたが、振幅の過小予測が残る。[512操作probe](../research/idm/nested512-displacement-note.md)、[誤差分解](../research/idm/right-gripper-error-note.md)。画像特徴MSEの改善を精密操作全般の改善と混同しない。

次元の機序診断では、保存μ32の代わりにQ差分1024を加えたreadoutで96変位MSEが.316632→.286871、train-fold PCA32では.306865となった。しかし右gripperはいずれも修復されず、「32D headでgripper情報が落ちる」という仮説は支持されない。全群のkernel強度、train-only選択、旧target scaleを揃え、元Model経路で保存μの完全再現も確認した。これは指定readoutでのアクセス可能性であり、条件付き相互情報の測定や次元拡大の採用根拠ではない。[Q／μ／PCAの比較](../research/idm/query-accessibility512-note.md)。

候補は、**未来補間を使う学習と、現在から条件付き未来を予測する学習の間で、zに求められる情報が変わる**という問題を、診断と学習設計の両方で扱うこと。詳細な情報恒等式は後述する。単なるauxiliary lossやstop-gradient追加では新規性は弱い。研究として成立させるには、既存IDM/flowの複数設定で再現する失敗、情報の代替に関する予測、同容量・同計算量での改善、実行時に使える行動条件での有効性が必要になる。

[IDMの次実験案](../research/idm/next-objective-proposal.md)は、未来教師によるz局所修正を診断として使い、encoderの抽出不足とdecoderの反応不足を調べる。修正結果の蒸留はtrainだけで行い、validationで未来を使って最適化した値を通常予測性能として報告しない。semi-amortized inferenceに先行研究があり、原理自体は新規ではない。実測変位の残差教師も、基準予測の誤差や行動とカメラの条件付き相関で失敗し得るため、採否条件を仕様化した。

### 初期step2900比較の判断と、その後の検証

初期比較のcacheはtrain128／validation64遷移で、元encoderもwarmup前のstep2900だった。改善IDMを固定した広いhead比較は400更新・seed42で直接MSE0.143357、flow0.170028、4標本平均0.157610。ここから大規模収束後の優劣やトップカンファレンス相当の価値は断定できない。

元の学習済みTransformer decoderは同じ64例でMSE **0.143712**。[元モデルとのpaired比較](../research/results/reference-idm-vs-original-seed42-20261113.json)では、新しい直接予測の差は−0.000354、episode bootstrap 95%区間は[−0.006046, 0.005690]で、明確な優位はない。小型scratch対照からの改善を元のStage1への大幅改善と混同しない。元モデルとは学習歴・容量・追加head学習が異なるため、この比較も同計算量のアルゴリズム比較ではない。区間は学習済みモデルに条件づけており、設計選択やseedの不確実性を含まない。

その後、追加2 decoder seedの比較と、当初の研究未使用96エピソード評価を完了した。今後データ数・学習量を増やす際にも、新しい評価分割を確保し、現在のvalidationを見て選んだ設計を同じvalidationだけで確証しない。改善率に加え、episode単位の差、seedばらつき、学習・推論計算量、小さな操作の精度を報告する。現時点で「flowによる圧倒的なMSE改善」は未達成であり、そのような主張はしない。

## 実装

[潜在flow](../vendor/lawam/latent_action_model/core/utils/latent_flow.py) は、現在特徴 `h` と潜在行動 `z` を条件に速度場を学習する。`x0 = h + sigma * epsilon`、`x1 = future`、`x_tau = (1-tau)*x0 + tau*x1` とし、速度教師は `x1-x0`。生成計算の時間 `tau` と、データの物理時間幅は異なる。

- `direct`: 従来の一回のTransformer予測。既存設定の既定値。
- `refinement`: 現在特徴から同じ更新ネットワークを反復する決定論的予測。終点回帰で学習する。
- `flow` / `current`: 現在特徴の周囲のノイズから未来特徴へ輸送。
- `flow` / `noise`: ガウスノイズから未来特徴へ輸送。現在特徴は条件入力として保持。

flowの学習ではランダムな生成時刻の速度MSEを最適化し、IDMのKLと状態補助損失を加える。推論はEuler積分。速度の大きさを潰さないよう、速度出力にはLayerNormをかけない。検証では未来特徴を速度場に渡さず、実際にサンプリングした終点を評価する。

学習時の一段の終点推定は、正解を含む補間点を使う。そこで `train/interpolation_*` と表示し、真の未来予測精度として扱わない。`val/feature_mse` は生成した終点の誤差、`val/copy_feature_mse` は現在特徴をそのままコピーした誤差。`val/recon_loss` は従来と同じ特徴損失。生成モデルでbest-of-Nだけを報告しない。

IDMと状態予測器の初期値は、同じseedなら全方式で一致するようにした。ただし共同学習後のIDMは方式ごとに異なる。現在のStage1評価は正解未来を見たIDMコードを条件にするため、実行時の予測能力の証明ではない。

[UPSTREAM.json](../vendor/lawam/UPSTREAM.json) は公開コードの元ハッシュを保持し、変更した2ファイルのハッシュを [RESEARCH_PATCHES.json](../vendor/lawam/RESEARCH_PATCHES.json) に記録する。ハッシュ検査は元版と研究用変更の対応を検証する。

## 実行

リポジトリのルートで実行する。`/path/to/...` は既存のローカルモデルと準備済みデータに置き換える。ダウンロードは行わない。

```sh
.venv/bin/python -m pytest -q tests

.venv/bin/python research/run_latent_flow_pilot.py \
  --vision-model /path/to/dinov3-vitb16-pretrain-lvd1689m \
  --prepared-root /path/to/preprocessed \
  --output runs/latent-flow-pilot-seed42 \
  --accelerator cpu --steps 100 --val-batches 32 --seed 42
```

[pilotスクリプト](../research/run_latent_flow_pilot.py) は小型構成（IDM/予測器各1層、幅64）、同じデータ分割、augmentationなしで5方式（1段の残差回帰を含む）を順番に学習する。出力ディレクトリの再利用を拒否する。各方式の設定・ログ・checkpointと、集約した `comparison.json` を保存する。100 step・検証先頭32ペア・1 seedは予備実験にすぎず、方式の優劣を確定しない。

通常サイズの設定は [現在特徴flow](../configs/stage1_flow.yaml)、[ノイズflow](../configs/stage1_flow_noise.yaml)、[決定論的反復](../configs/stage1_refinement.yaml)。`train_stage1.py` の `--vision-model`、`--train-manifest`、`--val-manifest`、`--output` で環境に合わせる。本学習は別途設定・実行する。接続確認専用の [flow smoke設定](../configs/stage1_flow_smoke.yaml) もある。

## 論文として成立させるための判断基準

単なるflow化を新規性にしない。次の既存研究は直接関連する。

- [VFMF, §3–4](https://arxiv.org/html/2512.11225v1): VFM特徴の確率的予測と特徴圧縮。高次元DINO特徴をそのまま生成する難しさも議論。現在の直接特徴flowは、圧縮を使う強い方式との比較が必要。
- [RLA-WM, §3](https://arxiv.org/html/2605.07079v1): DINO差分から圧縮表現を学び、その表現を行動条件付きflowで予測する。残差化と低次元化自体を独自性にできない。
- [Flow-JEPA, §4](https://arxiv.org/html/2608.29029v1): 現在観測と行動系列を条件に将来の潜在系列をflowで共同生成する。未来系列全体のflow生成も既存。

### 現在特徴始点の位置づけ

条件 `h` を固定して `y_tau = x_tau-h` と置けば、現在特徴始点のflowは、`y0 = sigma*epsilon` から `y1 = future-h` へのflowと等価な輸送問題になる。したがって「現在特徴から開始する」ことだけは、新しい確率モデルの証明にならない。実装上のパラメータ化、最適化、計算効率が変わる可能性を実測する。

### 検証する仮説

1. **残差の予測をしやすくする効果**: current-flowの優位性が、確率的生成ではなく現在特徴の保存に由来する可能性。コピー、1段の残差回帰、反復回帰と比較する。
2. **不確実性を表現する効果**: 条件が十分なら直接回帰で足り、条件が不十分なときにflowが効く可能性。IDMの潜在幅、固定したIDM、未来を使わない行動条件を別々に評価する。単一サンプルMSEだけで多峰性を評価しない。
3. **局所操作を予測する効果**: 全特徴の平均誤差が下がっても、静的背景の再現だけかもしれない。独立に定義した操作領域・物体位置・接触結果で改善を測る。予測誤差が大きい画素を選んで操作領域と呼ばない。

これらは研究仮説であり、新規性の確定ではない。採択水準の主張には、既存方式との差、失敗機序、独立した操作評価、複数データ・seedでの再現が必要。

### 本比較の未完了項目

- 同じ学習済みIDMを凍結した予測器単独比較と、共同学習の比較を分ける。
- SmoothL1/MSEの損失差、最終LayerNormの差、モデル容量を揃える対照を追加する。
- 同じ推論時間／計算量で比較し、積分1/2/4/8/16回を測る。反復回帰は逆伝播も反復するため学習計算量も異なる。
- ノイズ強度と始点の種類を交差して比較する。現在のpilotはcurrent=0.1、noise=1.0なので始点だけの比較ではない。
- 時間幅別・episode別に評価する。固定seedの複数サンプル、平均予測、分布評価を分離し、episode単位の信頼区間を出す。
- 現在の先頭32ペアから全validation・未見タスクへ広げる。test splitは選択に使わない。
- 実行可能な行動条件で評価し、未来画像からの情報による改善と区別する。

## 検証記録

2026-11-13: CPU上で実RoboMIND＋凍結DINOを用いたflowの2 step学習、サンプリング評価、checkpoint保存、3 step目へのoptimizer込み再開が成功。23件のテストが成功。GPUはCUDA初期化時のメモリ不足で失敗し、既存の本学習には介入していない。単体テストは一定速度の厳密輸送、行動条件への勾配、固定ノイズでの再現、未来教師を受け取らない生成API、不正設定、IDM初期値の一致を検査する。

接続テストは性能改善の証拠ではない。

100 step、seed 42、validation先頭32ペアの初期結果（小さいほどよい）:

| 方式 | 特徴MSE |
|---|---:|
| 現在特徴をコピー | 0.24791 |
| 直接予測 | 0.86478 |
| 1段残差回帰 | 0.25714 |
| 4段反復回帰 | 0.25629 |
| 現在特徴からのflow、4段 | 0.26676 |
| ノイズからのflow、4段 | 1.50916 |

この予備結果はflowの優位性を支持しない。現在特徴を保存する3方式は近く、いずれもコピーを上回らない。直接予測も未収束と考えられ、これを本サイズモデルの性能と解釈しない。始点保存の効果と確率的学習の効果を分離する必要がある。

設定・分割ハッシュ・測定値を含む [保存した実験記録](../research/results/latent-flow-pilot-20261113.json)。

詳細ログ: `runs/latent-flow-pilot-20261113/comparison.json`、`runs/latent-flow-residual-pilot-20261113/comparison.json`。

追加診断は [評価スクリプト](../research/evaluate_latent_flow.py) で行う。validation全域から等間隔の64ペアを選び、同じノイズを使って正しいz・ゼロz・次のペアのzを比較する。サンプル平均のMSEと各サンプルのMSEを分離する。ゼロz／別zは条件依存の診断であり、実行可能な反実仮想の正解を与える実験ではない。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/evaluate_latent_flow.py \
  --runs runs/latent-flow-pilot-seed42/direct \
         runs/latent-flow-pilot-seed42/flow_current \
  --output runs/latent-flow-pilot-seed42/paired-evaluation.json \
  --pairs 64 --samples 4
```


### validation全域の64ペア診断

全方式で同じペア・同じ教師特徴を使用したことを検査済み。flowは固定seedで4サンプル。

| 方式 | サンプルMSE | 4サンプル平均のMSE | 別ペアのzでのMSE |
|---|---:|---:|---:|
| direct | 0.860298 | — | 0.860301 |
| refinement | 0.213813 | — | 0.213814 |
| flow_current | 0.224591 | 0.216648 | 0.224591 |
| flow_noise | 1.500026 | 0.733001 | 1.500026 |
| residual | 0.214481 | — | 0.214481 |

コピーMSEは0.205827。全方式で別ペアのzへの交換による誤差差はごく小さい。これは初期モデルが動画ごとのzの違いを十分に利用していない可能性を示すが、未収束の小型モデルなので、本サイズStage1やflow一般の欠陥とは結論しない。

[ペア別の測定値](../research/results/latent-flow-paired-20261113.json) を保存した。サンプル誤差＝平均予測誤差＋サンプル分散の恒等式も数値的に検査済み。現在特徴flowの平均予測もコピーを上回らないため、分散を表現したことだけで精度改善とは言えない。CPU所要時間は他プロセスと共存した診断値であり、正式な速度比較ではない。

次の判断に必要なのは、十分に学習した同一IDMを固定した予測器比較。現時点ではこの案を有効な新手法として採用しない。

## 固定IDMによる予測器単独比較

[特徴抽出](../research/cache_stage1_features.py) と [予測器の学習](../research/train_cached_decoders.py) を追加した。動いている本学習から名前付きstep checkpointを読み出し、元ファイルが削除されても再利用できる重みスナップショットを保存する。`last.ckpt` は書換え中の可能性があるため入力として拒否する。checkpoint・manifest・キャッシュのハッシュを記録する。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/cache_stage1_features.py \
  --run /path/to/existing-stage1-run \
  --checkpoint /path/to/existing-stage1-run/checkpoints/epoch=0-step=2900.ckpt \
  --output runs/frozen-idm-step2900 --train-pairs 128 --val-pairs 64

CUDA_VISIBLE_DEVICES='' .venv/bin/python research/train_cached_decoders.py \
  --cache runs/frozen-idm-step2900 --output runs/frozen-idm-decoder-comparison \
  --steps 400 --batch-size 8 --dim 128 --layers 2 --seeds 42
```

- trainとvalidationはepisodeが重ならないことを検査する。両者とも全時間位置から等間隔に選び、augmentationなし・観測済みendpointだけを使う。
- DINOとIDMは凍結し、IDMのGaussian meanを保存する。同じキャッシュを全予測器に与え、ミニバッチ選択用乱数を生成モデルの乱数から分離する。
- 直接回帰は最終LayerNormあり・なしの2種類。1段残差回帰・4段反復回帰・現在特徴flowと比較する。残差／反復／flowには同じ初期ネットワーク重みを与える。
- 回帰は終点MSE、flowは速度MSE。IDMを含まないためKLや状態補助損失はこの段階では使わない。
- 100 stepごとのvalidation、最後のtrain再構成誤差、z交換診断、パラメータ数、実行時間を保存する。flowは4サンプルの平均予測と各サンプルの誤差を別記する。best-of-Nは使わない。
- 128ペア・小型予測器・1 seedは機序を調べる予備実験であり、実データ全体や本サイズへの汎化を証明しない。元の本サイズ予測器の値は参考診断であり、学習量・容量が違う新予測器の公平な対照ではない。

使用開始した2,900 step checkpointはwarmup 5,000 stepの途中であり、IDMの収束を仮定しない。100 stepの共同学習との違いを確かめる段階で、最終判断には成熟したcheckpointと複数seedが必要。固定IDMの有効性自体をz交換診断で検査する。

### 固定IDM比較の結果（2026-11-13）

2,900 stepのIDMを凍結。train 128 episode／128ペア、validation 64 episode／64ペア。新予測器は幅128・2層、400更新、batch 8。主要3方式は各910,848パラメータで、同じ初期値・同じ入力・同じミニバッチ順序を使った。

| 方式 | seed数 | validation MSEのseed平均 | seed間標準偏差 |
|---|---:|---:|---:|
| コピー | — | 0.205827 | — |
| 直接回帰、最終LNあり | 1 | 0.231413 | — |
| 直接回帰、最終LNなし | 1 | 0.212655 | — |
| 1段残差回帰 | 3 | **0.181039** | 0.000515 |
| 4段反復回帰 | 3 | 0.181222 | 0.001728 |
| flow、4段・個々のサンプル | 3 | 0.204777 | 0.001777 |
| flow、4サンプル平均 | 3 | 0.194197 | 0.001719 |

平均化したflowは16回のネットワーク評価を必要とし、1回の残差回帰より計算量が多い。flowと残差回帰のMSE差は全3 seedで正（flowが悪い）。学習したseedに条件づけたepisode bootstrapでも、単一サンプルの差の95%区間は[0.02119, 0.02641]、4サンプル平均の差は[0.01068, 0.01581]。この区間は未知の学習seed全体や別データへの汎化を保証しない。

seed 42のflowについて積分回数1/2/4/8/16も評価した。4回から16回に増やすと、個々のサンプルMSEは0.20448から0.20976へ悪化した。積分回数不足だけでこの差を説明する証拠はない。

**判断:** この小規模・固定IDM・特徴MSEの条件では、試した速度出力型current-flowを採用する根拠は得られなかった。後述の出力ランク制約があるため、同じ幅・パラメータ数だけではflowの適切な対照にならない。残差回帰を対照とし、制約を解消したflowを追加比較する。flow一般の否定、最適化の収束、データ全体での優劣は主張しない。

**別の重要な観測:** 固定IDMのvalidationコードは、平均ベクトルの二乗ノルム6.262に対し、動画間の分散の合計が0.00000911だった。元の本サイズ予測器も、正しいzでのMSE 0.143712と別episodeのzでのMSE 0.143711がほぼ同じ。このcheckpointと評価条件では、動画ごとのコードの違いが予測に寄与する証拠がない。

ゼロzでは元の予測器のMSEが0.212439に悪化するが、これは「行動情報を使っている」証拠にならない。ほぼ一定の非ゼロコードがAdaLNの共通条件として働き、ゼロ化がその共通条件を壊す可能性がある。これは構造からの解釈であり、学習全体を通じた潜在崩壊を確定したものではない。

[集計・区間](../research/results/frozen-idm-summary-20261113.json)、[seed 42の全記録](../research/results/frozen-idm-seed42-20261113.json)、[seed 43/44](../research/results/frozen-idm-seeds43-44-20261113.json)、[積分回数](../research/results/frozen-idm-integration-20261113.json)、[元予測器のz交換](../research/results/frozen-idm-reference-20261113.json)、[コード分散](../research/results/frozen-idm-diagnostic-20261113.json)、[キャッシュ来歴](../research/results/frozen-idm-cache-manifest-20261113.json)。

追加した集計・診断の実行入口は [paired集計](../research/summarize_cached_comparison.py)、[元予測器の再評価](../research/evaluate_cached_reference.py)、[積分回数比較](../research/sweep_cached_flow.py)。固定IDM比較を含む25件のテストが成功。

### 出力ランクの診断と追加対照

小型flowの速度出力は `v=Wq+b`、Wは768×128である。Wの列空間に直交する射影をPとすると、区間長1の積分後にも `P(x_final-h-b)=0.1 Pε` が残る。更新回数を増やしても、この部分空間の初期ノイズを消せない。独立な標準正規ノイズによる期待MSE寄与は `0.1²(768-128)/768=0.0083333`。これは真値との差や他の成分を含む総誤差の説明ではなく、4サンプル平均ではこの寄与も1/4になる。

[保存重みの診断](../research/results/flow-rank-obstruction-20261113.json)では、16ノイズ標本の当該成分は0.00832376、1/4/16更新で射影恒等式の最大誤差は3.2e-6未満だった。本サイズの幅1024には、この幅128由来の制約をそのまま適用できない。[RAE §4.1・付録B.1](https://arxiv.org/html/2510.11690v1)にも狭い出力の速度回帰制約があるが、今回の条件付き予測の性能を保証する定理ではない。

追加した [endpoint_flow](../research/endpoint_flow.py) は、現在特徴hに学習した残差を加えた未来推定 `fθ` を出し、`v=(fθ-x_t)/(1-t)` とする。raw stateを通す解析的な項により速度の固定低ランク制約を解消する。[JiT](https://arxiv.org/abs/2511.13720v2)や[JLT](https://arxiv.org/abs/2605.27102v3)に関連する既知のパラメータ化であり、新規性は主張しない。終点そのものの残差表現には狭い出力の制約が残る。

訓練は終点MSEであり、これは線形経路で `(1-t)²` を掛けた速度MSEに等しい。したがって今回の対照は出力構造と時間重みを同時に変更している。1更新の推論はノイズ付き入力からの直接回帰と等しく、反復の効果を別途調べる必要がある。未来真値は訓練経路だけに使い、推論には渡さない。

直交ノイズ消去・損失の等価性・有限勾配と既存キャッシュ学習のテストは成功。幅128・2層・400更新・seed 42/43/44での追加比較が完了した。

| 方式 | 個々のサンプルMSE（3 seed平均） | 4サンプル平均のMSE |
|---|---:|---:|
| 元の速度出力flow、4更新 | 0.204777 | 0.194197 |
| clean endpoint flow、4更新 | 0.203065 | 0.200648 |
| 1段残差回帰 | 0.181039 | 同一の決定論的予測 |

[追加比較の全記録](../research/results/frozen-idm-endpoint-flow-20261113.json)。clean endpointのseed間標準偏差は個々のサンプル0.002328、4標本平均0.002197。構造上のノイズ残留は解消したが、残差回帰を超える結果ではない。読み出しと時間重みの同時変更なので、各要因の因果効果を分離した比較でもない。

[seed 42の更新回数比較](../research/results/endpoint-flow-integration-seed42-20261113.json)では、1/2/4/8/16更新のMSEが0.191722/0.193799/0.202585/0.216668/0.233957。更新を増やせば予測が良くなるという仮説を支持しない。途中状態への依存と生成軌道上の誤差を次に検査する。

追加対照は同じネットワークで、`noisy_residual`は時刻0だけの終点回帰と1更新推論、`endpoint_anchored`は半分の訓練例を時刻0、残りを一様時刻として4更新推論を使う。後者の0.5は切り分け用の固定値で、最適値という主張はない。前者の予測が訓練時にも未来真値に依存しないことを含む5テストが成功。単なる時間重み変更を新規手法と扱わない。

[seed 42](../research/results/frozen-idm-source-anchoring-seed42-20261113.json)と[seed 43/44](../research/results/frozen-idm-source-anchoring-seeds43-44-20261113.json)の400更新比較が完了した。時刻0だけの回帰は3 seed平均sample MSE 0.180554、4標本平均0.180251。混合時刻の4更新は0.194052、平均0.193660だった。時刻0だけの回帰と通常の残差回帰の差は−0.000485、学習したseedに条件づけたpaired episode bootstrapの95%区間は[−0.001319, 0.000360]で、明確な改善を示さない。混合時刻は全3 seedで残差回帰より悪い。学習時刻と推論更新回数の両方が異なるため、これだけでshortcutが原因と確定できない。[集計と区間](../research/results/source-anchoring-summary-20261113.json)。

次の対照として、幅128のtrunkとは別に768次元の途中状態と現在特徴を直接受け取る幅768のheadを追加した。同じ初期重み・容量の `wide_residual` と `wide_flow` を比較する。出力ランク制約を除く実装であり、広いheadやraw入力経路自体に新規性は主張しない。容量が増えるため、元の幅128モデルとの差だけでflowの効果を判断しない。[実装](../research/wide_head_flow.py)と、重み一致・出力ランク・入力依存・有限勾配を含む関連7テストが成功。

[seed 42・400更新の比較](../research/results/frozen-idm-wide-head-seed42-20261113.json)が完了した。広いheadの残差予測はMSE **0.160909**、flowの個別標本は0.191024、4標本平均は0.177293だった。同容量の残差予測への優位はなく、固定低ランクの速度出力だけを敗因とする説明は不十分。この1 seedから一般の優劣や収束は主張しない。各方式ともz交換による変化が小さく、元の未成熟IDMを固定した条件である。次は同一重みの積分回数診断と、改善したIDMの凍結コードを使う比較で、予測器と条件情報の効果を分離する。

[広いheadの積分回数診断](../research/results/wide-flow-integration-seed42-20261113.json)では、1/4/8更新の個別標本MSEは0.190271/0.191024/0.193521、4標本平均MSEは0.178395/0.177293/0.179042。8更新に増やしても改善しない。この範囲で積分回数不足だけを支持する結果はなく、高精度solverへの収束を確認したわけではない。

[軌道診断](../research/results/wide-flow-trajectory-seed42-20261113.json)では、同一重み・各例1個の固定ノイズから4更新を行い、学習用の真値補間状態と自己生成状態を同じ時刻で比較した。validationの局所終点推定MSEは、時刻0で両者0.190213、時刻0.5で補間状態0.018847対生成状態0.190665、時刻0.75で0.004473対0.190952だった。時刻0.75の状態間MSEは0.106909。trainでも同様の差がある。

補間状態には真の未来が含まれ、時刻が進むほどそれ自体が正解へ近づくため、上記は将来予測性能の改善でもshortcutの因果的証明でもない。一方、初期時刻の推定が既に弱く、局所終点推定は自己生成軌道上でほぼ改善していない。学習用補間の損失低下だけで推論精度を判断せず、初期条件からの予測目的と反復による改善を別々に評価する。[診断実装](../research/diagnose_flow_trajectory.py)では `teacher_endpoint_error=(1-t)^2*teacher_velocity_error` と時刻0の一致を検算し、真値を使う経路と生成経路の違いを確認する2テストが成功。既存の4標本評価とは乱数の消費順が異なるため、個別標本の値は厳密には一致しない。

### 次に比較する設計の根拠

時刻やノイズの変更だけで大幅改善が得られるという証拠は、ここまでの予備実験ではない。次は、強い決定論的予測器を共通の条件ネットワークとして構築し、その上でflowの学習目的を比較する。過去パッチへのcross-attentionを使う[DINO-world](https://arxiv.org/abs/2507.19468v1)は既存の対照候補だが、大規模動画学習の下流評価を今回のMSE改善と取り違えない。

特徴の共分散を使う前処理も候補だが、[Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337v5)の白色化空間MSEは元座標では異なる誤差重みになる。元のDINO特徴MSEを維持して比較し、trainデータのみで統計を推定する必要がある。可逆変換だけでは狭い出力の制約は解消しない。

並列の[IDM調査](../research/idm/README.md)は、現在条件付き参照差分を用いたコードを検査している。これは真の未来を入力するStage1 posteriorの再構成評価であり、未来を観測しない実行時の予測とは別。flow比較に使う場合も、同一の凍結コードを各decoderへ渡して切り分ける。

### IDMとの接続で検証する仮説

固定encoder・新規headと小型decoderを2000更新した[3 seed比較](../research/idm/reference-comparison2000/summary.json)では、posterior平均を使うvalidation MSEはraw 0.178606、global center 0.170286、現在条件付き参照差分0.160962、constant 0.179063だった。参照差分はglobal centerより5.48%低い。これはStage1再構成の改善であり、カメラと操作の分離や実行時の行動条件予測の証明ではない。参照差分のseed42 headを凍結し、元と完全に同じcurrent/future/rowsを保ったコードで4種類のdecoderを比較中。head学習の追加計算も含める必要があり、元IDMからの差だけでアルゴリズムの公平な優位とはしない。

新たな仮説は、**未来を含む学習用flow状態が、IDMのzから得る追加情報を代替する**ことである。現在X、未来Y、確率的posterior Z、独立Gaussian雑音εに対して、`S_t=(1-t)(X+σε)+tY` とする。固定encoderと独立性の下ではchain ruleから

`I(Y;Z|X) = I(Z;S_t|X) + I(Y;Z|X,S_t)`。

Xを条件とするとS_tは `Y+σ(1-t)/t ε` と同値な観測で、tが増すとGaussian観測の雑音が減る。channelを再結合したデータ処理不等式から、状態側の情報 `I(Z;S_t|X)` は非減少。元の相互情報が有限なら、zが追加する未来情報は非増加となる。連続な決定論的μコードでは相互情報が無限になり得るため、この差分式をそのまま数値診断へ使わない。

これはencoder崩壊やMSE悪化の証明ではない。固定IDM実験ではencoderは変化せず、二乗誤差でのzの利益は条件付き平均の変化に依存する。共同学習では、(a)同じ予測branchを加えてflow勾配もencoderへ流す、(b)flowからencoderへの勾配だけ止める、(c)通常の残差予測のみ、を比較する候補になる。現在入力からの予測branchでzを訓練し、flowは固定zを使う構成自体は既知要素の組合せであり、新規性や性能は未確立。実行時評価は履歴・候補行動から得られるcode、または明示したpriorを使う。Stage1 posterior再構成と、実行時に利用可能な行動条件での予測を分ける。

### 改善したIDMを固定したdecoder比較

[400更新・seed42の全4条件](../research/results/reference-idm-decoder-comparison-seed42-20261113.json)が完了した。共通の参照差分IDMを凍結し、train 128 / validation 64の同じ遷移で比較した。

| decoder | 個別予測MSE | 4標本平均MSE | z交換時の個別予測MSE |
|---|---:|---:|---:|
| 幅128残差回帰 | 0.165299 | 同左 | 0.192186 |
| 幅128 flow | 0.196084 | 0.185640 | 0.209123 |
| 広いheadの残差回帰 | **0.143357** | 同左 | 0.177598 |
| 広いheadのflow | 0.170028 | 0.157610 | 0.198063 |

同容量の広いhead同士でも直接回帰が優位。flowは4更新、4標本平均は合計16回のモデル評価を使う。未成熟な元IDMだけがflowの敗因とは言えない。ただし1 seed・小標本・400更新であり、収束後や大規模学習の優劣は未確定。直接回帰の300更新時点0.140166を選ばず、事前指定した400更新を比較した。元IDMからの改善には参照headの2000更新という追加計算がある。

追加のdecoder seed43/44を完了し、[3 seed集計](../research/results/reference-idm-wide-three-seed-summary-20261113.json)を保存した。共通の参照IDM seed42を固定したdecoder初期値の比較であり、IDM学習seedの再現ではない。直接回帰は平均MSE **0.141045**（seed標準偏差0.002003）、flowは0.173144（0.003464）、4標本平均は0.160264（0.003115）。直接回帰との差は個別標本で+0.032099、4標本平均で+0.019219で、全3 seedでflowが悪い。学習済みseedに条件づけたepisode bootstrap 95%区間はそれぞれ[0.028822,0.035588]、[0.016100,0.022542]。これは小標本・固定400更新での結果であり、大規模収束後の結論ではない。

z交換での悪化は利用の証拠だが、操作意味の証拠ではない。[近傍交換診断](../research/idm/neighbor_swap_spec.md)でも、事前指定半径内で全validation例を交換できず、近傍化によりgapが縮小した。今後は実測操作変位への追加予測利得を測り、未来の外観やカメラ情報だけを符号化する解と区別する。

二乗誤差の観点では、条件C=(X,Z)と独立な生成ノイズを使う予測 Ŷ に対し、有限二次モーメントの下で `E[||Y-Ŷ||²|C] = tr Var(Y|C) + ||E[Y|C]-E[Ŷ|C]||² + tr Var(Ŷ|C)`。N個の独立標本を平均すると最後の項だけが1/Nになる。生成分布が正しくても個別標本が条件付き平均よりMSEで有利にはならない。一方、有限容量・最適化下ではflowの平均推定が直接回帰より良くなる可能性までは否定しない。この恒等式は今回の学習失敗の原因を特定するものではなく、分布生成の改善をそのままMSE改善と見なさないための評価上の根拠である。

### 次の対照：自己生成した終点を直接学習する

`research/train_cached_decoders.py` に `wide_terminal` と `wide_refinement` を追加した。前者は `wide_flow` と同じ重み初期化、current+0.1noiseの初期分布、4回Euler更新、ネットワーク容量を使い、最終出力と未来特徴のMSEを全4更新へ逆伝播する。後者は同じ4回更新で初期ノイズを0とする。真の未来を途中状態に混ぜず、損失の教師としてだけ使う。これらはflow matching自体の新手法ではなく、学習目標と生成軌道のずれを切り分ける対照である。

最初は同じ参照IDM seed42コードで、decoder seed42・400更新を比較する。既存の幅128 refinementの失敗だけで、広いheadでも補正が無効とは判断しない。`wide_terminal` と `wide_flow` の固定ノイズforward一致、`wide_refinement` の手動4回Euler一致、全段への有限非零勾配をテスト済み。4段の逆伝播はCFMの1点学習より高価なので、同更新回数の差を同計算量の優位とは表現しない。結果が良ければ、同じ総学習時間または概算のforward/backward回数を使う直接回帰・CFM対照と追加seedが必要。

[終点MSEの400更新結果](../research/results/reference-idm-terminal-objective-seed42-20261113.json)は個別標本0.143665、4標本平均0.136843。通常のwide_flowより良いが、個別標本は同seedの直接回帰0.143357とほぼ同じで、4標本平均には16回のモデル評価を要する。[同じ重みで推論ノイズを0とした探索的評価](../research/results/terminal-zero-noise-seed42-20261113.json)は4回更新で0.134862。これは条件付き平均のMonte Carlo推定ではなく、入力ノイズを変えた決定論的推論点である。いずれもstep6501の元モデル0.125485を超えていない。実測学習時間は約950秒だが、同時CPUジョブの負荷が異なるため、過去runとのwall-clock比を厳密な計算効率比較にしない。

同時に、軌道診断へ `--code-swaps` を追加した。正しいzで生成した状態を固定し、zだけをglobal cyclic交換して速度感度と局所終点MSE差を測る。teacher側では終点gapが速度gapの `(1-t)^2` 倍なので、時刻とともに終点gapが減るだけではzの利用低下とは言えない。この反例とz非依存場でgapが0になるテストを含む4テストが成功。global交換は分布外になり得るため、条件付き相互情報や操作意味の推定には使わない。

[改善IDM・広いflow headの診断結果](../research/results/reference-idm-flow-code-trajectory-seed42-20261113.json)では、validationのteacher速度感度はt=0で0.016650、t=.75で0.004073、生成状態では同じ時刻に0.016650、0.003885だった。両経路でほぼ同じ低下があり、teacher未来補間固有のz利用低下とは言えない。生成状態にも前段の正しいzの効果が蓄積されるので、固定状態での局所交換から全軌道のz利用を判断しない。

teacher速度損失の交換gapは0.024437→0.003744で、終点係数だけではない低下があるが、その原因は未確定。生成状態からの局所終点推定MSEは0.172410→0.169907で改善が小さい。これは各例1個の固定ノイズによる診断で、4標本評価のMSEと乱数消費順が異なる。これらの結果は、終点目的対照を試す根拠にはなるが、IDM共同学習での勾配分離の有効性を証明しない。

### Warmup後のcheckpointで基準を更新する

主学習を変更せず、保存済みstep6501 checkpointから別のcacheを抽出した。[比較監査](../research/results/idm-step2900-vs6501-20261113.json)により、train128／validation64のepisode・index・dtとcurrent/future特徴がstep2900と完全一致することを確認した。新cacheは `runs/frozen-idm-step6501` に保存し、元cacheは保持する。

元decoderの抽出時validation MSEはstep2900の0.143714からstep6501の **0.125486** へ改善した。したがって、step2900由来の改善head＋広い直接予測headの0.143357を、現在のStage1の改善とは扱えない。checkpointの学習進行という、architecture変更以外の強い要因がある。

posterior平均zのvalidation例間分散総和は9.11e-6→1.386e-4、平均コードのノルム二乗は6.262→0.682。zero-z MSEは新checkpointで0.126313だった。いずれもzの操作意味やdecoder利用の証明ではない。新decoderで同じglobal交換評価を追加し、初期checkpointに限った現象かを確認する。新旧の学習済みdecoder比較と、同じ凍結IDMを使う予測器比較を混同しない。

[step6501 decoderの再評価](../research/results/frozen-idm-reference-step6501-20261113.json)はMSE0.125485、global交換0.125495で差は約1.07e-5。抽出値との最大再現誤差は2.96e-5以内。warmup後にもこの交換に対する依存は弱い。ただし観測コードの交換は分布外の可能性を持ち、操作の因果的有用性を直接測ったものではない。微小なコード分散の解釈には演算精度も確認する。

この結果から、次のarchitecture候補は既存予測器を保持した補正である。固定した `b(X)=decoder6501(X,0)` を基準に、`Yhat=b(X)+r(X,Z)` を学習する。bは未来や新しいIDMコードを使わず、補正の出力層を0初期化すれば学習開始時は基準予測と一致する。初期一致は汎化改善の保証ではない。現在特徴Xから一から大きな残差を学ぶ対照と比べて、既存モデルの知識を保持したまま未説明の変化を学べるかを検証する。

flowを比較する場合も、同じbを始点に使い、元のXを条件として保持する。固定基準／決定論的補正／flow補正を同容量・同じコードで比較し、基準decoderの計算を含めて報告する。これはboostingやCorrDiffに近い既知の分解で、新規性は主張しない。

[基準抽出](../research/cache_current_prior.py)と[補正module](../research/prior_correction.py)を実装した。新cache `runs/reference-idm-with-prior6501` は元のcurrent/future/参照IDMコードを保ち、current-only baselineを追加した。train MSE0.110360、validation MSE0.126313。新旧cacheのfeatures/pairs一致、zero-code基準の再現、有限性とhashを確認。基準一致の初期出力、全方式の初期重み一致、flow学習時のcurrent条件保持、samplerが未来targetを受け取らないことなど、関連12テストが通過した。

seed42・400更新の `prior_residual` はvalidation MSE **0.108801**、z交換時0.136350。対応する `prior_flow` は学習中で、比較runの終了後にまとめて保存する。現段階は開発に使った64例・1 seedの結果であり、改善確定とはしない。元のzero-code基準より約13.9%、実posteriorを使う元decoderより約13.3%低いが、後者とは学習歴や追加パラメータ・計算が異なる。

追加検証用に、元validation manifest160 episodesから研究のtrain128／validation64 cacheに含まれない **96 episodes** を選定した。[選定記録](../research/results/research-unused-episode-selection-20261113.json)は各episodeから観測済み端点の遷移を1件ずつ固定seedで選び、画像や予測誤差は参照していない。head・calibration・400更新の選択を固定したまま評価cacheを準備する。これは研究未使用episodeであり、元Stage1学習のvalidation集計に含まれた可能性があるため、独立外部testとは呼ばない。


## 誤差領域の分解と次の座標系対照（2026-11-13）

[既に観測済みの96エピソードの記述分析](../research/results/prior-correction-region-errors-20261113.json)では、各例のDINO特徴変化が大きい上位10%パッチで、zero-code priorのMSE0.486365に対して直接補正3 seed平均0.404394（16.85%改善）。変化の小さい下位50%では0.018567から0.014748（20.57%改善）だった。全領域平均は既存集計を差5.82e-10で再現した。正解特徴で評価領域を定義しており、特徴変化はカメラ・照明も含む。物理的な操作領域、小さな動作、制御成功率の検証ではない。比較対象はoriginal posterior decoderではなくzero-code priorである。

[残差座標の補正器](../research/centered_prior_correction.py)を次の対照として準備した。物理状態x、現在特徴h、凍結予測b(h)に対し、ネットワークが見る状態を `u=h+(x-b(h))` に変える。sourceは `h+sigma*epsilon`、終点は `h+(y-b(h))` となり、出力を `b(h)+(u-h)` で元座標へ戻す。bは積分時刻に依存しないため速度教師は従来と同じ `y-b(h)-sigma*epsilon`。変更は同じ条件付き問題の有限容量ネットワークへの提示方法であり、情報や自由パラメータを追加するものではない。

従来のprior_flowは状態b(h)+noiseと条件hを受け取り、clean bそのものは条件として渡さない。bはhから決定されるが、大きな凍結decoderの関数を小さな補正器が再現しないとsource noiseを識別しにくい可能性がある。新座標では状態から既知条件hを引くだけでsource noiseを取り出せる。ただしclean bの見た目を状態から外す影響もあるため、直接補正の同座標対照が必須。残差拡散の既知の考え方に沿った実装対照であり、新規性も性能向上も未証明。

初期重み・パラメータ数一致、初期prior保持、補間座標／物理終点の整合、source noiseの解析的除去を[3テスト](../tests/test_centered_prior_correction.py)で確認。factoryへprior_centered_flow／prior_centered_residualを追加し、元の補正器も含む9テストが通過した。[固定400更新の座標比較](../research/results/prior-centered-protocol-20261113.json)を開始した。当時実行中だった終点MSE4段／決定論的4段／直接1600更新は変更前のコードを読み込み済みで、その実験条件は変更していない。これら先行比較の結果は後述する。


### 既存flowのゼロノイズ推論

[固定CFM重みの探索的評価](../research/results/prior-flow-zero-noise-seed42-20261113.json)では、source noiseを0にした4段推論でdevelopment64 MSE0.119045、既に観測済みの96例で0.105965。通常の4標本平均の0.121798／0.108678より低いが、直接補正seed42の0.108801／0.093066には届かない。サンプリングの分散だけが差の原因ではない。これは条件付き平均を厳密に計算したものでも、学習した確率分布からの正しい標本でもない。評価済みデータを使った後続診断として扱い、新しい独立検証とは呼ばない。


### 4段終点学習と計算量対照の結果

固定seed42・development64・400更新の[終点MSE4段学習](../research/results/prior-terminal-seed42-20261113.json)は単一標本の平均MSE0.118054、4標本平均0.110737。直接1段400更新の0.108801より悪かった。[paired差の95%区間](../research/results/prior-terminal-vs-direct400-20261113.json)は4標本平均でも[0.000725,0.002946]。学習済みモデルに条件づけた開発データ上の比較である。

[決定論的4段と長期直接学習](../research/results/prior-refinement-compute-comparison-20261113.json)では、4段400更新は0.107910、1段400との差−0.000890の区間[−0.002740,0.000379]で、明確な改善とは判断できない。直接1600更新は0.116248へ悪化し、学習量追加だけでは改善しなかった。4段終点学習は4段を逆伝播するので、通常CFMや直接400更新より学習計算が多い。直接1600はforward/backward回数の概算対照であり、更新数・実FLOPsが厳密に同じではない。


同じ終点MSE4段モデルの[探索的ゼロノイズ推論](../research/results/prior-terminal-zero-noise-seed42-20261113.json)はdevelopment64で0.108412。直接400との差−0.000388、paired 95%区間[−0.001558,0.000610]で明確な優位はない。確率モデルの条件付き平均を正確に計算したものではなく、推論時にsource noiseを0とした診断である。4段の学習・推論費用に見合うMSE利得は現段階では確認できていない。

IDMの[postprojection比較仕様](../research/idm/postprojection-training-spec.md)に基づき、正規化方向、方向＋平均/尺度、train校正raw、および全例同じzero34入力の4群を開始した。全群のGaussian head・補正器の初期値、batch、posterior noise、容量を揃え、固定400更新とする。定数入力群ではmu/logvarが例に依存しないが、sample zはposterior noiseにより異なる。元のproduction経路とは異なる新規head＋補正器のjoint学習であり、元headの交換だけによる同条件改善とは主張しない。入力・校正・公平性の検査と9合成テストは成功、実データの最終結果は未取得。


## 残差座標・正規化介入の確定結果（2026-11-13）

[残差座標400更新](../research/results/prior-centered-seed42-20261113.json)は直接MSE0.111081、flow単一標本平均0.135362／4標本平均0.124708。元座標の直接0.108801、flow0.131709／0.121798を上回らなかった。[paired差](../research/results/prior-centered-vs-original-coordinates-20261113.json)では、flow4標本平均の悪化幅0.002910の区間は[−0.000557,0.006531]であり、全差が確定的とも言えない。source noiseの識別を容易にするだけでは十分でない。変換はclean priorの見た目をnetwork stateから外すため、その影響も混在する。採用根拠は得られなかった。

[postprojection4群400更新](../research/results/postprojection-heads400-summary-20261113.json)のposterior-mean MSEは、方向のみ0.125038、方向＋平均/尺度0.121914、train校正raw0.120407、定数入力0.124325。平均/尺度群と定数群の差−0.002412の95%区間は[−0.004424,−0.000636]、rawと定数は−0.003918 [−0.006524,−0.001547]。開発64例・1 seed・探索的複数比較であり多重性補正なし。全群の初期重み・batch・posterior epsilon hashが一致。headへのMSE勾配は初期zero gate下で0だが、その後は全群で学習経路が開いた。rawもtrain MSE0.066957に対しvalidation0.120407で過学習差が大きい。従来参照差分の0.108801とはencoder・head入力次元・学習歴が異なり、同計算量比較ではない。

### 物理時間と実測変位での追加検査

[96例の物理時間別集計](../research/results/prior-correction-horizon-strata-20261113.json)では、nominal1.2秒（許容1e-4秒）が69例、短い遷移が27例（0.0667〜1.1333秒）。1.2秒の69例に限ったoriginal6501 MSE0.119942に対し直接補正3 decoder seed平均0.102228、14.77%改善。短い遷移での改善15.13%と近く、全体14.85%が短時間例だけによる効果ではなかった。これは既に観測済みの評価データの事後的記述分析である。

[実測変位の固定線形probe](../research/idm/displacement-probe-note.md)はcurrent patch-mean768＋current state20＋code32から、各arm xyz/gripperの8D endpoint差を予測。train128内の共通5foldだけでscalerとridge alphaを選択した。zero/original6501/reference42 codeのnormalized MSEはvalidation64で0.943928／0.908416／0.732079、observed96で0.866747／0.839108／0.603342。referenceはzero比22.44%／30.39%低下、7/8座標で改善し左gripperは微悪化。[一度限りの正則化grid境界監査](../research/idm/displacement-ridge-boundary-note.md)でalpha上限をtrain-CVだけに基づいて1e6へ拡張しても、選択値・最終重み・全予測は変わらなかった。

これは指定した線形・pooled current baselineを補う表現上の予測効用であり、conditional MI、action commandの同定、camera不変性、実行時予測の証明ではない。posteriorは正解未来を見ており、Zが含む非線形current情報で弱いbaselineを補う可能性もある。観測済み96例は独立最終testではない。xyzはm・各arm元座標系、gripperはsource値で物理単位未保証。測定ラベルはaction commandや関節角ではない。

現在の有力案を検証するため、[IDMと補正器の両seedを変える追試](../research/results/reference-idm-joint-seed-replication-protocol-20261113.json)を開始した。従来の3 decoder seedはIDM42を共有していた。今回は既存IDM43/44の各2000更新headを使い、decoder43/44を固定400更新で学習する。現在特徴・未来特徴・rowの完全一致を確認して6501の同じbaselineを再利用した。最終結果は未取得。


[head＋decoder seed追試](../research/results/reference-idm-head-decoder-seed-summary-dev64-20261113.json)のdevelopment64結果は0.108801／0.107296／0.107798、平均0.107965、seed標準偏差0.000766。original6501の0.125485より13.96%低い。IDMの2000更新headと補正器のseedを同時に変えたが、encoder2900とprior6501は共有しているため、encoderを含む全学習の3 seed追試ではない。96例での対応code再抽出・再評価を準備中。

[同一8例に対する学習後4head診断](../research/idm/postprojection-posterior-sensitivity400.md)では、未来交換の平均対称Gaussian KLがdirection5.80e-5、statistics12.7995、raw28.9374、constant0。directionの単一sample識別精度の上界は約50.2%とchanceに近い一方、他2群の大きいKLや緩い上界は高い識別精度・有用な操作情報の証明ではない。特に交換futureはOODになり得る。正規化方向だけからは取り出しにくい未来依存があるという診断と、開発MSEの小幅改善は整合するが、性能改善の全原因を証明するものではない。
