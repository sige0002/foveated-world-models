# RoboTwinのz問題：損失・初期化の独立監査

2026-11-14。対象は未改造の `RLinf/LaWAM` commit `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`。モデルの学習・GPU実行・checkpointの変更は行っていない。公式ソースの読解、式の導出、CPUでの小さいAdaLNブロックの微分だけを実施した。

**現在確認できた最も重要な初期条件は、公式のAdaLNゼロ初期化により、学習開始直前の画像損失からIDMへの勾配が0になること。** その後ゲートは学習できるため、永続的な配線切れではない。KLとgが先にIDMを動かす初期過程は候補になるが、今回の崩壊原因という断定はできない。

## 損失の加算とzの注入は別

公開設定の損失は、画像SmoothL1＋状態差分MSE＋係数付きKLの単純な和である。zの注入は各ブロックのAdaLNであり、単に `visual_tokens + z` とはしていない。[損失合算](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/lam_lightinng.py#L643)、[AdaLN](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_decoder.py#L47)。

論文Appendix E.2も、単純な加算注入ではzのnorm変動が画像トークン全体の変動やloss spikeを生じ得るため、AdaLNを採用したと述べる。これは著者の採用理由であり、AdaLNならzの利用が必ず成功するという主張ではない。[LaWAM論文](https://arxiv.org/html/2606.15768v2#A5.SS2)

## ゼロ初期化の正確な意味

ブロックは概略 `h' = h + gate(z) * F((1+scale(z))*LN(h)+shift(z))`。実装はgateだけでなく、shift/scale/gateを出す最終Linearの全weightとbiasを0に初期化する。[初期化44–45行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_decoder.py#L38)

初期状態では全ブロックが恒等写像なので、decoder全体は現在特徴だけの関数となり `∂D/∂z=0`。画像損失の勾配はIDM、VAE、action_in_projへ届かない。一方、gate出力を生成するLinearの行には、原則として非ゼロの勾配があり、その更新で条件付け経路が開く。shift/scale行やattention/FFNへの最初の画像勾配は0。現在特徴側の入出力投影は最初から学習できる。

公式ソースから `modulate` と `AdaLNBlock` をASTで抽出し、CPU・幅8・head2・seed42・dropout0でMSEを微分した。optimizerは作っていない。

| 検査 | 結果 |
|---|---:|
| ブロック出力と入力 | 完全一致 |
| 条件ベクトルへの勾配の最大絶対値 | 0 |
| shift/scale行の勾配norm | 0 |
| attention側gate行の勾配norm | 0.133886 |
| FFN側gate行の勾配norm | 0.145340 |
| attention weightの勾配norm | 0 |

これはブロックの代数を小規模に実行確認したもので、実画像・全サイズの初期学習を測定した結果ではない。理論上の0勾配は、全サイズでも同じゼロ初期化に従う。

ロボット例ではgがzへ依存するため、初回からg損失がIDMを学習させ得る。人動画のbatchは状態項を飛ばすので、初回はIDMにKLしか掛からない。現在のRoboTwin実験は全例ロボットであり、後者の条件を今回の説明に流用しない。

## SmoothL1はMSEより常に弱いわけではない

1要素の残差を `e = prediction-target` とする。公開コードのSmoothL1は切替幅 `b=0.1`。

```text
d SmoothL1 / de = e/b       if |e| < b
                 sign(e)   otherwise
d MSE / de      = 2e
```

両方とも同じ要素数で平均するなら、その共通係数は比から消える。

| 残差域 | SmoothL1とMSEの勾配絶対値の比 |
|---|---|
| `0 < |e| < 0.1` | 5倍 |
| `0.1 ≤ |e| < 0.5` | `1/(2|e|)`、1より大きい |
| `|e| = 0.5` | 同じ |
| `|e| > 0.5` | `1/(2|e|)`、1より小さい |

したがってMSEへの変更は単に画像損失を強める操作ではない。大きな残差を相対的に強め、小さな残差を弱める。背景と動く領域の残差分布によって、画像・状態・KL間の勾配の比率も変わる。総loss値だけで説明せず、同じbatch上のIDM勾配norm・角度・残差域の割合を計測する必要がある。

## KLが5倍であることから言える範囲

`l = log(sigma²)` とすると、対角Gaussianの1例のKLは `0.5 * Σ(mu² + exp(l) - 1 - l)`。同じμとlに対する勾配は `beta*mu` と `0.5*beta*(exp(l)-1)` である。[公式VAE](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/vq.py#L67)

同一状態なら `beta=5e-5` は `1e-5` の5倍のKL勾配を作る。ただし総勾配、Adam更新、実際のz情報量が5倍変わるとは限らない。他項の向き、適応的二次モーメント、gradient clipping、学習中の分布変化が介在する。

既存の1万更新でKL勾配が小さいことは、初期の影響を否定しない。特に初回画像勾配が0という設計のため、初期の成分比を追う価値がある。βを変更しただけでzの分散が増えても、正しいzで交換zより未来誤差が低くならなければ修復とは判定できない。

## ノイズを受けるdecoderがzを弱める理論上の経路

説明用の局所線形近似として `prediction=f(x)+A*z`、`z=mu+epsilon`、`E[epsilon]=0`、`Cov(epsilon)=Sigma`、`r=y-f(x)` と置く。MSEについては厳密に、

```text
E_epsilon ||r - A(mu+epsilon)||²
    = ||r - A*mu||² + tr(A*Sigma*Aᵀ)
```

となる。μが残差をあまり説明しなければ、右辺第2項を避けるためAを小さくする方向が有利になる。現在特徴からのfが十分強ければ、zから得る利得はさらに小さくなり得る。この自己強化は理論上の候補であり、実際の非線形AdaLNの学習軌跡を証明した式ではない。SmoothL1でも凸性からゼロ平均ノイズは期待損失を改善しないが、上の二次のペナルティ分解をそのまま使えるのはMSEの場合である。

既存の128組ではμの変動がサンプリングノイズに比べ小さく、未来変更によるVAE直前の変化の約95.98%が正規化で抑えられる大きさ方向だった。[現在の重みの診断](../../docs/robotwin-z-conditioning-proof.md)。これは上の候補と整合するが、KL・g・初期化のどれがその状態を作ったかは分離実験が必要。

## 絶対状態と差分状態の比較で混同しないこと

`g_abs(s,z) = s + g_delta(s,z)` と明示的に定義すれば、`||g_abs-s_T||² = ||g_delta-(s_T-s)||²` で、残差headの勾配まで同じになる。単に教師を未来状態へ変更して同じMLPの出力を当てると、この等価性はない。初期値、表現する関数、shortcutが変わる。

公開コードは正規化した状態を差し引いて教師を作る一方、g出力をtanhで[-1,1]へ制限する。[教師生成](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/data_loader/collate.py#L53)、[g出力](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_decoder.py#L209)。各状態成分が[-1,1]なら差分は[-2,2]まで取り得るので、公開設定には原理上の範囲不一致がある。実データでどれほど該当するかは別途必要。ローカルの2*tanh変更は範囲を広げるが、初期の出力微分も変えるので、IDMへのg勾配に無関係ではない。

gは現在状態も入力するので、低いg誤差だけでz利用を証明できない。絶対状態への変更も、現在状態のコピーを容易にする場合があり、必ずzの学習を強めるとは言えない。

## 4条件で分かることと追加候補

baseline、KLだけ、画像MSEだけ、両方の2×2比較は、βと画像損失の主効果・相互作用を同一seedで切り分けられる。一方、この4本で同じ失敗が起きても、KL無関係とは言えない。比較するβが両方非ゼロだからである。

次の1条件として最も変更が小さいのは **β=0、その他baselineと同じ**。同じ予算で失敗するなら、少なくともその条件ではKLが失敗の必要条件ではない。改善するなら、β=1e-5でも強すぎる可能性を追える。ただし1seed・3000更新の陰性結果は長期・他seedを否定しない。

その後、ノイズ仮説には `z=mu` のみ（KL係数は保持）、g初期勾配仮説には `lambda_aux=0` のみという独立介入が対応する。前者はlog分散が再構成から学習されなくなるので、元VAEの再現とは呼ばない。AdaLN初期化を変える場合も、その前にstep0・1・10・100などで画像→IDM勾配の立ち上がりを測り、長期に閉じている事実を確かめる方がよい。

追加の仕様差として、論文E.2はencoder/decoderとも24層、公開YAMLはencoder24層/decoder12層である。また論文E.6のrobot horizonは1.2秒、公開設定は1.6秒。今回の4条件は**損失の因果比較**であって論文の完全再現ではない。[論文E.2/E.6](https://arxiv.org/html/2606.15768v2#A5)、[公開設定](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/config/dino_large_vae.yaml)

## 同じ構造でも大きな共通成分へ進み得る理由

### 実際に使われるQFormerの確認

対象は `QFormer_att` であり、同じファイル内の別クラス `QFormer` ではない。前者はqueryをcross-attentionした後、画像トークンと結合し、`norm_first=True` の `torch.nn.TransformerEncoderLayer` を24層通す。最後のqueryをそのまま返し、1024→32次元の `out_proj`、VAEの32次元LayerNormへ渡す。24層の残差列の直後、1024次元でのfinal normはない。最初の `CrossAttentionBlock` にはnormがあるので、「encoderの全域で正規化なし」でもない。[QFormer_att](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/modules.py#L699)、[実際の呼び出し](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_encoder.py#L49)、[出力投影](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_encoder.py#L123)

残差の各小ブロックを `q_next = q + r` とすると、恒等的に

```text
||q_next||² - ||q||² = 2<q,r> + ||r||²
```

である。Pre-LNはrの入力を正規化するが、この残差和自体を一定normにはしない。各層のrが共通方向へ揃えば共通成分が蓄積する。同じ重みを共有する全標本のquery、attention出力、FFNやbiasなどは候補だが、どこが今回の主因かは未測定である。初期query、cross-attention後、24層の各attention/FFN後を測れば、既に大きい値の持ち越しか、各層での増幅かを区別できる。

### Pre-LNの一次論文が示す範囲

[Xiong et al., ICML 2020](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) のLemma 2は、仮定した初期化でPre-LNの**期待二乗norm**が深さに比例して増える範囲を示す。Lemma 3はLNのJacobianの尺度と入力normの関係を示す。ただし学習後の今回の28,438というnormを予言する定理ではない。同論文の標準Pre-LNは予測前にfinal LNを含み、今回の投影前にfinal LNを追加すべきという直接の証明でもない。

[Sun et al., The Curse of Depth, v3](https://arxiv.org/html/2502.05795v3#S3) は、訓練したLLMで深い残差の分散増加と層間の方向変化の減少を調べる。§3の分布仮定の下で示すのは線形成長の下限と指数成長の上限であり、**全Pre-LNが指数的に発散する定理ではない**。今回への検証可能な予測は、深さとともに共通成分が増え、相対的な残差・未来由来の方向変化が小さくなること。異なる用途のLLMの結果だけからLaWAMの歴史的原因を確定できない。

### 「大きいnorm」と「未来情報が消える」を分ける

32次元の投影をp、座標平均を引く行列を `C=I-11ᵀ/d`、`u=Cp/||Cp||` とする。LNのepsilonを無視できる範囲では、

```text
LN(p) = gamma * sqrt(d) * u + bias
delta LN ≈ diag(gamma) * sqrt(d)/||Cp|| * (I-u*uᵀ) * C*delta p
```

したがってpが大きいだけでは情報欠落は決まらない。方向も標本に応じて十分変わればLN後にも違いは残る。一方 `Cp_i ≈ a_i*v + t_i`、tが小さい場合、標本差の多くがaの変化となり、LNが使う方向uはほぼ同じになる。ここでの「標本に共通するベクトルv」は「1標本内の座標平均」と異なる。

現在の測定では未来変更による投影差のradial成分がローカル95.98%、公式10.72%。これは現在のローカルが上記の不利な幾何にある根拠だが、なぜそこへ進んだかは別問題である。また、投影後を正の定数で縮めるだけではLNの出力はほぼ変わらず、失われた方向差は復元しない。[測定と介入](../../docs/robotwin-z-conditioning-proof.md)

query norm比は約26.8倍なのに、投影後の座標標準偏差比は約157.6倍ある。out_projのFrobenius normが4.77対5.53という値だけで投影を除外できない。入力方向と特異ベクトルの整列、bias、座標中心化が寄与し得る。`||C W q||/||q||` とbiasを別々に測り、標本共通成分・未来変更差それぞれの増幅率を比較する必要がある。

### 正規化下の更新と、初期学習の競合

[van Laarhoven, 2017](https://arxiv.org/pdf/1706.05350) は、正規化で関数が重みのスケールに不変となる場合でも、最適化の有効な学習率が変わることを示す。仮定したスケール不変な重みでは、方向の学習率はSGDで `eta/||w||²`、Adamで概ね `eta/||w||` に依存する。これは**重みの**スケールの解析であり、今回の**活性値の**normをそのまま代入できない。Adamの状態、上流Jacobian、weight decayまで確認せず、LN前の小さい勾配を「encoderが学習停止した」と解釈しない。[勾配比較の基準](robotwin-release-gradient-comparison-criteria-20261114.md)

[He et al., ICLR 2019](https://arxiv.org/html/1901.05534v2#S3) は、初期に推論器がdecoderの変化へ追い付かないVAEがzを無視する方向へ進む例と、構造・目的関数を保って更新手順を変える改善を示す。また大きなKLだけでは情報の利用を証明できないと指摘する。今回と同一のモデルでもなく、今回の分布は事前分布へ一致していないため、厳密なposterior collapseと同一視はしない。「構造が同じなら同じzが学べるはず」という推論への反例と、初期更新を追う根拠として使う。

今回へ適用する仮説は、(1)ゼロ初期化で初回画像→IDM勾配は0、(2)gとKLが先にIDMを動かす、(3)画像経路が開いた時に未来に応じたμの違いがノイズより小さいと、decoderがその違いを使う利得が弱い、という循環である。(1)はソースから確定、(2)の経路も存在するが、**gの初期優勢が共通方向の肥大を作ったという(3)までの履歴は未確認**。gは有益なzも学習できるので、gを原因と決め付けない。gを外す比較、zをμのみとする比較は別々に行う必要がある。

公開重みとの追加の一致条件での逆伝播では、画像損失のz勾配norm比は約3.14倍で、LN前の約814倍とは異なり、sigmaを掛けた勾配は近い。したがってdecoder全体がzから切れている説明は支持されない。一方、これは損失方向の微分であり、未来情報の利用は保証しない。KL/image比はローカルμで0.00756、log分散で0.42792なので、KLの現在の影響を全枝で一様に小さいとは言えない。[一致条件の測定](robotwin-z-underuse-20261114/released-z-gradients-summary.json)

### 原因を切り分ける測定と強い対立仮説

| 仮説 | 支持する測定 | その解釈を弱める測定 |
|---|---|---|
| Pre-LN内で標本共通の残差が蓄積 | 層ごとの `||E q||²`、`tr Cov(q)`、`2<q,r>`、未来変更のtangent/radialを追うと、共通方向の増大と方向差の減少が特定層から始まる | 初回cross-attention後ですでに同程度に大きい、または深さとともに方向差が十分増える |
| 出力投影が共通方向を選択的に拡大 | Wの大きさだけでなく、共通方向の実効増幅が未来由来の接線方向より大きい | qの時点で既に同じ比率で方向差が小さく、Wが悪化させない |
| gの初期更新が不利な状態を作る | step0/1/10/100の損失別encoderパラメータ勾配・更新とμ変動を記録し、gのみ除去で共通成分の増大とz不使用が改善 | g除去でも同じ時点・程度で発生。単にg損失が画像より大きいだけでは支持しない |
| posteriorノイズがdecoderのz利用を抑える | 他条件を保持してz=μのみで、正しいz対交換zの未来誤差差が増える | μのみでも方向差・交換zとの差が回復しない |
| 小batch・投入標本数・学習期間が不足 | 同じデータから投入量・有効batchを分離して増やすと、未来に応じた方向差とz交換の損失差が育つ | 同一の十分な投入量で特定の損失/更新変更だけが再現性をもって修復する |

最後の仮説は残す必要がある。現在はbatch4×10,000で延べ40,000組であり、保持する約608万フレームを十分学習したことにはならない。論文のbatch1024×100kと差はあるが、それは必要量の証明でも、公式配布重みの履歴の確認でもない。[投入量の記録](../../docs/robotwin-z-underuse-investigation.md#データを増やすべきか構造を変えるべきか)

batch64で同じ更新数を走らせるだけでは、batchの勾配分散と投入量16倍が混ざる。同じ64例を平均して1回更新する条件と、4例ずつ16回更新する条件は、投入例は揃うが更新数が異なる。この二軸とLR scheduleを明示しなければ、batchだけの因果効果とは呼べない。短いbatch4×3000比較は、初期の不具合や条件間の分岐を捉える用途に適する。全条件が不成功でも、データ不足を否定して構造欠陥を確定することはできない。

現段階で最も区別力のある読み取りは、**層ごとの共通成分・未来由来の方向差の追跡**と、**同じDINO座標でのIDM未来入力への画像勾配**である。後者は教師側の未来特徴をdetachし、IDMを通る枝だけを測る。活性値の座標スケールだけで生じる差を減らし、未来入力がどこまで損失へ影響するかを調べられる。ただし、これでも現在の微分であり、過去の学習経路は短期比較のログで確かめる。

追記：8組で実施されたμのみの全経路検査では、未来入力へのSmoothL1勾配normは公式0.0118857、ローカル4.365e-6で約2723倍の差となった。4本のランダムprobeによるμ-Jacobianの推定は約561倍差。未来入力だけを現在入力へ置換した出力の変化MSEも公式0.23834に対しローカル2.596e-9だった。この結果はLN前の座標の差だけでなく、実際の未来入力への依存が弱いことを支持する。一方、学習時のパラメータ更新や原因の履歴は未確定のままである。[全経路の測定](../../docs/robotwin-z-conditioning-proof.md)
