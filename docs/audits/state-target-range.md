# gの教師・出力範囲と差分IDMの区別

2026-11-14確認。公式 `RLinf/LaWAM` main のHEADは `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`。GitHubから直接取得したファイルと、SSD1の改変なしcheckoutの両方で照合した。

## 結論

公開コードのgは状態差分を教師にし、出力はtanhで制限する。この組み合わせには範囲の不整合がある。一方、論文v1/v2の3.2節・式(4)ではgは未来状態そのものを予測する。公開コードと論文はこの点で一致しない。著者が実際の配布重みをどちらで学習したか、なぜ差異が生じたかは未確認。重みだけでは教師やtanhの有無を確定できない。

## 混同してはいけない二つの差分

| 対象 | 計算 | 出所 |
|---|---|---|
| 補助予測gの教師 | 正規化した未来状態 − 現在状態 | 公式公開コードの初期コミットから存在 |
| 新規差分IDM | `z = head((Q(C,F) − Q(C,C)) / scale)` | ローカル研究実装 |

後者のC/Fは現在・未来のDINO特徴、Qは共有encoder。前者は実機状態の教師。両者は別処理であり、同じ「差分」という言葉だけで説明していたことが誤解を招いた。[差分IDMの実験記録](../difference-idm-ablation-plan.md)。

## 一次資料と履歴

- [公式collate.py 58行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/data_loader/collate.py#L58): `delta_t[i, :D] = end - start`。
- [公式g 214行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_decoder.py#L214): `s_pred = torch.tanh(self.state_decoder(fused, emb))`。
- [公式損失への受け渡し](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/lam_lightinng.py#L674-L702): `delta_proprio`をそのまま教師にする。
- [公式L2](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/utils.py#L102-L115): `(s_pred - delta) ** 2` の有効成分平均。出力の2倍化・教師の半分化はない。
- 全公開履歴を取得し、上記g/collate/損失が初期コミット `56a18297ba0833bc6e02af4f5a38328c24f3fba3`（2026-06-14）から変更されていないことを確認。
- ローカルvendorへの導入commit `f9214fe` からHEADまで、g定義とcollateの差分は空。
- [論文v2 3.2節・式(4)](https://arxiv.org/html/2606.15768v2#S3.SS2): 教師は未来状態 `s_T`、補助損失は `||g(s,z) − s_T||²`。v1も同様。tanh・差分教師への変更理由は本文に見当たらない。

## 公式ローダーでの再現

[検証スクリプト](../../research/verify_official_state_target.py)で公式LeRobotDataModule、公式gのforward、公式L2を使用。RoboTwin ID1、1.6秒間隔、16実サンプル。gの隠れ幅8で関数動作のみ確認し、性能評価や学習はしていない。

- episode17665、開始frame23: 右gripper現在−1、未来＋1、実際の教師＋2。
- episode5498、開始frame274: 右gripper現在＋1、未来−1、実際の教師−2。
- gの最終層出力を20に設定したprobe: tanh後はfloat32で1。
- この16例の範囲制約による状態L2下限は0.0325875059。学習済みモデルの誤差ではない。

全6077247フレーム監査の状態L2下限0.02926666は、別の全候補集計。両者を混同しない。

生JSON・全フレーム監査・画像照合はSSD1の `foveated-world-models/research/official-data-contract-audit-20261113/robotwin-training-preflight/` に保存。実行結果は `official-runtime-target-proof.json`。

## 解釈と未確定事項

「未来状態用tanhを残したまま差分教師へ変更した」という説明は整合する仮説だが、公開履歴にその変更過程はなく断定できない。SFT用RoboTwinでの再現は公開コードの不整合を示すが、配布モデルの全事前学習データでの頻度やz学習不良の原因まで証明しない。

公式の正確なStage1混合比は公開予定がないと[著者回答](https://github.com/RLinf/LaWAM/issues/7#issuecomment-5392830464)に記載されている。RoboTwin単独実験を論文の完全再現とは呼ばない。

修正するなら、論文の未来状態教師に戻す案と、差分教師を保ってgの出力制約を変更する案を区別する。教師だけ半分にするとL2の尺度も変わる。いずれも無変更の公開コード対照と別条件として扱う。今回の調査では学習コード・原データを変更していない。

## 修正方針の更新: 差分教師を維持して出力範囲を拡張

当初追加した「グリッパーのみ未来状態を教師にする」実装は、ユーザーの後続指示に従って撤回・削除した。現在はグリッパーを含む全成分で公式の `未来−現在` 教師を保持し、gを `state_output_scale × tanh(...)` に変更できる。[設定overlay](../../configs/stage1_robotwin_delta2.yaml)は `state_output_scale: 2.0`、warmup1000。既定値1.0は無変更の対照用。これは公開コードの範囲不整合への修正であり、未来状態を教師にする論文の式そのものではない。

倍率はLightningのhyperparametersからモデルとgへ渡る。gの重み形状は変更しないが、state_dict単体には倍率が入らないため、読み込み時は必ず設定も復元する。元のg重みで倍率だけ変えて学習済み性能が改善したとは扱わない。本学習の状況は末尾に記載する。

[検証](../../tests/test_state_output_scale.py): 差分教師±2の維持、両方向開閉の予測・勾配、無変更対照との重み互換性、倍率・warmup設定、不正倍率の拒否、半分教師と4倍L2の等価性を確認する。

### ミスとして記録する内容

- 公開コードは正規化済み状態の差分（最大±2）をtanh（±1）で予測し、実データで範囲外教師が存在する。確認された実装上の不整合。公開履歴から著者の変更理由や配布重みの学習定義は確定していない。
- 私は「tanhでは差分を予測できない」と一般化しかねない説明をした。正しくは範囲外の差分が予測できない。
- 私は未来状態への教師変更を先に勧め、`2*tanh`や差分教師の再スケーリングを先に比較しなかった。必須の修正と選択可能な設計変更を混同した。
- グリッパーだけ未来値に変えた構成は全状態が論文準拠ではない。現在はこの混在案を撤回した。
- 範囲修正だけで四元数の符号・幾何の問題が解決するとは扱わない。

### −0.5〜＋0.5への正規化という代案

ユーザー指摘のとおり、現在・未来状態をそれぞれ半分にすれば差分は±1に収まる。ただしgへの現在状態入力も半分になるため、出力だけを2倍にする変更と同じ学習条件ではない。さらに公式RoboTwinでは四元数はmin–max対象外なので、XYZとgripperのmin–maxだけ変更してもq差分の範囲は直らない。

現在状態入力を維持し、**差分教師だけ半分**にする案なら、L2では次が厳密に成り立つ（同じ内部出力aと元の差分d）:

`(2*tanh(a) − d)² = 4*(tanh(a) − d/2)²`

したがって半分教師の補助L2を4倍すれば、2*tanh方式と同じ目的関数・勾配になる。重みを変えずに半分教師へ変更すると、対応する元単位の誤差に対する補助損失は1/4になり、視覚損失・KLとの釣り合いも変わる。これは現在のstate_loss_type=l2についての関係で、L1なら倍率2、一般の損失へ4倍を機械的に適用しない。

採用済み2*tanh版は前処理・入力・教師の単位を変えないため対照を作りやすい。両方式とも端点±2/±1は数学的には漸近値であり、飽和付近の勾配が小さい性質は同じ。性能改善は未検証。別方式へ変更したと偽らず、ユーザーの比較提案として残す。

## XYZと姿勢の調査結果

取得データのmeta/modality.jsonはXYZ・四元数とも `absolute: true`。RoboTwin公式文書[Data Type 2.5.1](https://robotwin-platform.github.io/doc/usage/configurations.html)ではendposeは世界座標系、順序は `x,y,z,qw,qx,qy,qz`。これはRoboTwinの定義と配布metadataの対応であり、元の全エピソードのexport過程を再実行した証明ではない。

| 項目 | 入力の表現 | 公開gの教師 | 出力制約 |
|---|---|---|---|
| XYZ | 絶対位置をmin–max | 正規化後の未来XYZ − 現在XYZ | 各成分tanh |
| 姿勢 | 絶対姿勢の四元数、変換なし | 未来q − 現在qの4成分差 | 各成分tanh |
| グリッパー | min–max後に符号反転 | 未来−現在 | 公開版tanh／採用修正版2*tanh |

四元数成分差は相対回転 `q_future × inverse(q_current)` ではない。gに四元数の単位ノルム制約やq/−q同一視の損失はない。したがって、元の四元数は正常でも成分差教師には表現上の跳びが入る。

全データの隣接qの内積が負の件数は左1733、右6029。映像を照合したep18020/frame158→159とep1268/frame107→108では、符号が反転する一方、実回転角は0.005870°/0.001483°。実機がその角度だけ動いたとの外部計測ではなく、記録されたqから符号不変の回転角を計算し、映像のほぼ同姿勢と照合した結果。

次の修正候補は、姿勢を未来qの予測に変え、単位化とq/−qを同一視する損失を使うこと。ただし単純に教師だけ未来qにしても符号問題は消えない。時系列で符号を揃える方法もあるが、それ単独で軌道間の符号一意性は保証しない。姿勢の教師・損失の変更は今回適用していない。

## 2026-11-14: 絶対値・正規化値・差分の全件検証

全6,077,247フレームの16状態成分を再走査。公式配布stats_gr00tを使ったmin–max（グリッパー符号反転含む）を適用し、四元数は原値のまま。画像や教師は変更していない。

| 成分 | 元の最小〜最大 | 前処理後の最小〜最大 | 1.6秒差分の最大絶対値 | 差分の絶対値>1の個数 |
|---|---|---|---:|---:|
| 左X | −0.718288〜0.278899 | −1〜1 | 1.527280 | 938 |
| 左Y | −0.936240〜0.199289 | −1〜1 | 1.624519 | 1,986 |
| 左Z | 0.521649〜1.494850 | −1〜1 | 1.444637 | 155 |
| 右X | −0.235569〜0.714459 | −1〜1 | 1.478187 | 617 |
| 右Y | −0.848150〜0.197040 | −1〜1 | 1.440835 | 313 |
| 右Z | 0.748042〜1.495603 | −1〜1 | 1.666914 | 2,459 |

XYZの元値は座標系・単位に依存するため、一般に±1の制約はない。このデータでは左Zの1,047,815フレーム、右Zの961,071フレームが1を超える。公式前処理後はXYZ全成分で範囲超過0件。ただし統計範囲外の新しいデータまで保証するものではない。公式min–max処理はここでは教師差分を再正規化しない。

四元数の全成分は元値・前処理後とも±1以内。単位ノルムの最大誤差は約5.17e−8。これは単位四元数の性質であり、各成分をデータmin–maxに通した結果ではない。成分差は別で、右qzの最大絶対差は1.990187、範囲超過177,922候補。四元数には「成分の範囲」と「4成分全体の単位ノルム」の両制約があり、tanhだけでは後者を保証しない。

機械可読結果はSSD1監査ディレクトリの `robotwin-training-preflight/absolute-and-delta-ranges.json`。

## 他の前処理実装5件の比較

調査日2026-11-14。比較対象は論文本文の網羅サーベイではなく、5プロジェクトの該当するデータ処理経路。GitHubのmainを読み、HEADの固定SHAで再取得して内容一致を確認した。GR00Tは現在の公開版であり、LaWAMがvendorした旧系統と同一とは扱わない。観測状態と行動教師の処理を区別する。

| プロジェクト・対象 | 実際にしている処理 | 正規化・教師との関係 | 境界・限界 |
|---|---|---|---|
| Diffusion Policy: RoboMimic absolute-action経路 | 行動のaxis-angleを回転行列経由でrotation_6dへ変換。汎用変換器はquaternion入力にも対応 | 行動XYZは範囲正規化、6D回転はidentity。観測のquatキーもidentityで残す | 行動6D表現はq/−q同一性を保つが、観測quatまで時系列連続化する経路ではない。6Dから回転への復元は直交化が必要 |
| NVIDIA Isaac-GR00T: 相対EEF処理 | quaternion等を回転オブジェクト・同次変換へ変換。相対姿勢は `inverse(T_current) @ T_future`。設定によりXYZ_ROT6D等へ出力 | 相対変換を先に行い、その相対action用統計で正規化。四元数の成分引き算ではない | 相対化は設定で有効化する処理。6D・rotvecなど表現により境界性質が違う。全モデルが一律6Dとは言えない |
| OpenVLA: OXEのFurnitureBench/IAMLab等 | action内quaternionをTensorFlow GraphicsのEuler角へ変換 | 後段にaction/proprioのbounds・q01/q99正規化とclipの処理がある。各データ設定で適用を選択 | Eulerの±π境界や特異点は残る。この関数内で時系列unwrapはしない。すべてのOXEデータが同じ生表現ではない |
| PerAct: RLBench教師生成 | quaternionを単位化、w<0なら符号反転、Euler角にして離散化。360°binは0へ戻す | 回転は3軸の分類教師で、四元数の成分MSEではない | `correct_rotation_instability` は確認版で実質そのまま返す。w≥0だけで全球的な連続性は保証しない。離散化精度・Euler特異点は別問題 |
| LeRobot: LiberoProcessorStep | 観測quat(xyzw)をaxis-angleへ変換。wを[-1,1]にclampし、`2*acos(w)`を使う | これは観測整形であり、action教師やgの損失ではない。関数内に時系列符号整合・quat単位化はない | この関数はqと−qを同じベクトルへ写す保証がない。単にaxis-angleへ変換しても不連続は解消しない。LeRobot全体の一般仕様とは呼ばない |

### 固定版の根拠

1. Diffusion Policy `5ba07ac6661db573af695b419a7947ecb704690f`: [Datasetの設定・normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_image_dataset.py#L148-L180)、[回転変換](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/rotation_transformer.py#L16-L59)。
2. GR00T `3b49067943eb1ab2eb2203b93d461c4390c78da3`: [相対姿勢](https://github.com/NVIDIA/Isaac-GR00T/blob/3b49067943eb1ab2eb2203b93d461c4390c78da3/gr00t/data/state_action/pose.py#L657-L678)、[相対化→正規化](https://github.com/NVIDIA/Isaac-GR00T/blob/3b49067943eb1ab2eb2203b93d461c4390c78da3/gr00t/data/state_action/state_action_processor.py#L363-L417)。
3. OpenVLA `c8f03f48af692657d3060c19588038c7220e9af9`: [FurnitureBench変換](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/oxe/transforms.py#L420-L440)、[後段normalizer](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/utils/data_utils.py#L54-L100)。
4. PerAct `02fb87681c5a47be9dbf20141bedb836ee2d3ef9`: [教師生成](https://github.com/peract/peract/blob/02fb87681c5a47be9dbf20141bedb836ee2d3ef9/agents/peract_bc/launch_utils.py#L116-L120)、[変換本体](https://github.com/peract/peract/blob/02fb87681c5a47be9dbf20141bedb836ee2d3ef9/helpers/utils.py#L43-L82)、[分類損失](https://github.com/peract/peract/blob/02fb87681c5a47be9dbf20141bedb836ee2d3ef9/agents/peract_bc/qattention_peract_bc_agent.py#L466-L468)。
5. LeRobot `fe1c54594f6ba275fccf210b9f7b11f0e2324b89`: [LIBERO観測処理](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/processor/env_processor.py#L62-L75)、[quat変換本体](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/processor/env_processor.py#L110-L149)。

### 同じ実データでの境界probe

[再現スクリプト](../../research/audit_rotation_preprocessing.py)で、固定版DPのRotationTransformerと、固定版LeRobot/PerActの該当関数本体を実行。後二者はASTで関数のみ取り出して不要なframework依存を避けた。GR00T/OpenVLAはソース追跡であり、実行済みとはしない。CPU2threads、学習なし。

| 左episode18020 frame158→159 | 結果 |
|---|---:|
| 記録qから計算したSO(3)角度差 | 0.00586967° |
| 生qベクトルのL2距離 | 1.9999999993 |
| 前frameとの内積で符号を揃えたqのL2距離 | 0.0000512225 |
| DPの6D変換後のL2距離 | 0.000139919 |
| LeRobot LIBERO関数のaxis-angleベクトル差 | 6.2831416 rad |
| PerActの5°刻み分類ラベル | 前後とも[2,53,38] |

これらは表現と尺度が異なるため、数値の大小をモデル性能の比較に使わない。axis-angleベクトル差6.283は実回転角ではない。

さらに、同じ60°Z回転のqと−qを入力すると、DPの6D差は0、PerActの分類は同じ。一方、LeRobot関数ではZ軸ベクトルが+1.0471976と−5.2359877になり、差は約2π。これは該当関数の境界probeであり、LeRobot全タスクで実害があるとまでは示さない。

**今回の全データでは左右ともqwの最小値が正である。それでも隣接内積が負の例が存在する。** 実例でも前後qwは両方正。したがって「wを正に揃えるだけ」の処置は今回のデータを変えず、解決にならない。どのexporterがこの符号を選んだかは未特定。

### 推奨と変更しない範囲

- XYZの範囲不整合は、論文に合わせて正規化済み未来XYZを教師にすれば解消する。ただし未見の統計範囲外には別の検討が必要。
- 四元数を維持する最小案は、未来姿勢予測と、予測を単位化した上でq/−qを同一視する損失。例は `min(||q_pred−q_target||², ||q_pred+q_target||²)`。これは論文の単純成分L2からの追加変更で、ゼロノルム予測の扱いと勾配の検証が必要。入力側のqの不連続まで消すものではない。
- 入力・教師の表現上の符号境界も避ける案は、回転行列経由の6D表現。ただし4→6成分になり、gの入力・出力・mask・既存checkpointとの互換性が変わる。勝手に導入しない。
- 時系列符号整合はepisode内で前frameとの内積が負なら反転する処理。ランダムにサンプルを引いた後ではなく元のepisode順で実施する必要がある。長い軌道・異なるepisode間の符号一意性や、姿勢全体の全球的連続性までは保証しない。
- Euler/axis-angleへの単純変換だけを「連続性の修正」として採用しない。

姿勢の表現・教師を変換する処理は未導入。範囲だけは上記の採用修正によりgの全成分で2*tanhを選択可能。原本データは変更していない。

## 追加確認: wが正でも180°付近で跳ぶケースへの実際の対応

前回の説明では、時系列符号整合という提案に対し、他プロジェクトの具体的な呼び出し元を示さなかった。一般論と実採用例を分けて補足する。

既調査5件について、このケースへの対応は以下の通り。

- Diffusion Policy: absolute-actionの回転を行列経由で6Dにする経路は、wの正負に依存せず回転を表現する。同じ実データで変換後の跳びが小さいことを検証済み。ただし観測quatのidentity処理にまでこの効果を一般化しない。
- GR00T: EEF相対姿勢を行列の積で計算し、XYZ_ROT6D等で表す選択肢がある。成分差を教師にしない。rotvecを選んだ場合まで180°境界が無いと主張しない。
- PerAct: w≥0だけで終わらずEulerへ変換・離散化し分類する。今回の実例の前後は同一ラベル。ただしEulerの別の境界や量子化問題は残る。`correct_rotation_instability`という名前の関数は確認版ではno-op。
- OpenVLA: 調査したOXEの一部変換はEulerへ変換。qの二重表現をEulerへ写すが、Eulerの境界を時系列でunwrapする処理は確認した関数にはない。今回の問題が一般に解決済みとは扱わない。
- LeRobot LIBERO: axis-angle変換だけで、今回の実例では約2πのベクトル差が残る。今回の不連続対策として、そのまま模倣する根拠にはならない。

### 実際に時系列符号整合を使う追加の1件: QuaterNet

ロボット操作データではなく、人間のモーションキャプチャ前処理の実装例であることを明示する。

固定版 `facebookresearch/QuaterNet@9d8485b732b0a44b99b6cf4b12d3915703507ddc` の [qfix](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L119-L136) は、隣接qの内積が負になった箇所から符号を切り替え、累積反転の偶奇で系列全体を整合させる。w≥0を要求しない。

関数の存在だけでなく、[MocapDataset._mirror_sequence](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/mocap_dataset.py#L64-L80) が、鏡映augmentation後の四元数系列へ実際に `qfix(mirrored_rotations)` を適用することを確認した。元の全収録データの生成工程まで同じ処理であるとは断定しない。

この固定版の関数本体をASTで取り出し、今回のRoboTwinの隣接2組に実行した。

| 実例 | 修正前qベクトル距離 | qfix後 | 表す回転 |
|---|---:|---:|---|
| 左ep18020 f158→159 | 1.999999982 | 0.0000512225 | 不変 |
| 右ep1268 f107→108 | 1.999999942 | 0.0000129390 | 不変 |

次フレームのwは負になる。これは姿勢が変わったのではなく、同じ回転の別符号を選んだため。JSONはSSD1監査rootの `robotwin-training-preflight/quaternet-qfix-probe.json`、ソースは `rotation-survey-sources/facebookresearch/QuaterNet/`。

4成分を維持する最小の前処理候補として、QuaterNet型のepisode内qfixには具体的な採用例と今回の境界例での再現根拠がある。ただし現在は2組の診断のみで、全episodeへの適用・学習への導入は未実施。episodeを跨いで累積しないこと、ランダム抽出前に系列順で処理すること、回転不変性・全件の隣接内積・1.6秒教師の分布を検証することが必要。異なるepisode間の符号一意性やgの姿勢損失の妥当性までqfixだけで保証しない。

## 全27,500episodeへのqfix適用と全件監査（実施済み）

ユーザー承認により、QuaterNetの固定版qfix関数をepisodeごと・左右の腕ごとに適用した。[処理・検証スクリプト](../../research/preprocess_robotwin_qfix.py)。出力はSSD1監査rootの `robotwin-qfix-data/robotwin_merged`。元データの `observation.state` 四元数8成分だけを別データとして修正し、XYZ・gripper・action列・timestamp・episode/frame/index列は同一。Stage1用派生データであり、SFT行動教師まで符号処理したデータとは呼ばない。

動画は原本 `sft-subsets/robotwin_merged/videos` へのシンボリックリンクで、動画のコピーは0バイト。派生データの実占有は約439MiB（Parquet約415MiB、meta約23MiB等）。公式ローダーが別途作成したArrow cacheは約1.115GB。合計追加量は約1.6GBであり、元の約79GBの動画等を複製していない。

全6,077,247フレームを変換し、保存した全10 Parquetを再読み込みして監査。episode先頭のqは原値を保持、episodeを跨いだ符号整合はしない。global/episode単位の四元数統計を更新し、それ以外の状態統計は維持。元の全data/metaファイルのhashが処理前後で不変であることも確認した。

| 指標 | 左 | 右 |
|---|---:|---:|
| 符号を反転したフレーム数 | 67,390 | 194,137 |
| 隣接内積が負の組（処理前） | 1,733 | 6,029 |
| 隣接内積が負の組（処理後） | **0** | **0** |
| 隣接qベクトル差の最大（前） | 2.000000058 | 2.000000058 |
| 隣接qベクトル差の最大（後） | 0.421946917 | 0.327528532 |
| 単位ノルムの最大誤差 | 5.14e−8 | 5.16e−8 |
| 全フレームで変更がq→qまたはq→−qのみ | 合格 | 合格 |

別の既存全件監査スクリプトでも、NaN/Inf・timestamp異常なし、隣接内積負0を再確認。元の物理的な大変化は消していない。最大隣接回転角は左48.7176°・右37.7018°で処理前と同じ。従って「全ての動きが小さく連続になった」とは主張しない。

1.6秒=48frame先、同一episode末尾clipで全候補を再評価。全成分の差分で絶対値2を超える候補は0。2*tanhの範囲超過によるL2下限は0（有限出力で端点に数学的に到達できる、あるいは実学習損失が0になるという意味ではない）。旧tanh±1の範囲では依然3,178,365候補に超過があり、全16D平均の下限0.0287463612。範囲拡張は引き続き必要。

公式LeRobotDataModuleからリンク先動画を含めて読み込み、`videos=[8,2,3,256,256]`、`states=[8,2,32]`、ID全て1、有効次元全て16、deltaが未来−現在と一致、最大絶対delta2を確認。既知の符号反転episode18020/1268の読み込みも成功。このローダー確認時点ではtrain27,473/val27の公式splitを使用。本学習のsplitは下記に更新した。

成果物はSSD1監査root内:

- `robotwin-qfix-data/robotwin_merged/qfix/audit.json`: 全件の保存・回転不変性・原本不変性・範囲監査。
- `robotwin-qfix-data/robotwin_merged/qfix/episode-flips.csv`: 全27,500episodeの変更フレーム数。
- `robotwin-training-preflight/qfix-state-audit-independent.json`: 別スクリプトの全件再監査。
- `robotwin-training-preflight/qfix-official-loader-check.json`: 公式ローダーの画像・状態の読み込み確認。

この前処理は学習処理ではなくCPU2threadsのデータ変換・監査。変換・保存再読込・原本hash再確認は18.68秒。別監査とローダー確認はこの時間に含めない。

## 2026-11-14: バッチ4で本学習開始

[本学習runner](../../research/train_robotwin_stage1.py)と[設定](../../configs/stage1_robotwin_qfix.yaml)を追加。SSD1上の公式commit `c12168d` の専用worktreeに、[g出力倍率だけのpatch](../../research/patches/official-state-output-scale.patch)を適用して使う。既存vendorの研究用flow・差分IDMなどはこの実験に含めない。起動時にcommitとpatch全体の一致を検証する。

公式既定の末尾0.1%は27episodeで、metadataの指示文がスイッチ操作に偏っていた。本学習はseed42、episode単位のランダム95%/5%分割を固定。train/valのIDリスト、統計、ソース差分、解決済み設定を出力先に保存する。

| split | episode数 | フレーム数 | バッチ4でのバッチ数 |
|---|---:|---:|---:|
| train | 26,125 | 5,776,302 | 1,444,076 |
| val | 1,375 | 300,945 | 75,237 |

episode重複は0。全27,500episodeのローカル先頭・末尾indexから元frame indexへの対応を検証し、各splitの1,024抽出例がそのsplit内に収まること、動画の読み出し、有効状態16次元、ID1を確認した。

公式の復元抽出を保持しているため、上記の1epochは抽出回数の定義であり全フレームを一度ずつ使う保証ではない。10,000更新×4例=40,000抽出は約0.006925epoch相当（0.69%）。1epoch完了の実験とは呼ばない。

条件はID1、qfix済み状態、全成分の未来−現在教師、g出力2*tanh、バッチ4、勾配蓄積1、10,000更新、warmup1,000。公式のAdamW lr=3e−4・weight decay=.01、bf16、gradient clip1、1.6秒間隔、画像256角、dual-view augmentation、IDM24層・LaWM12層・dim1024・code32を維持。DINOはローカル事前学習重みで凍結し、他は新規初期化。過去の不正な学習checkpointは再開しない。

公式との差分は、上記の範囲修正・qfix・splitに加えて、RoboTwin単独、バッチ4、worker4/prefetch2、学習期間/warmup、単一GPU、検証/保存の運用。公開datasetの正規化統計はそのまま使い、train側だけで再推定していない。これは完全な論文再現でも、無変更モデルとの対照実験完了でもない。

検証・checkpoint保存は2,000更新ごと。検証は固定された512抽出例（128バッチ）で、検証split全300,945フレームを毎回走査する設定ではない。validationのworker順序を固定し、同じ抽出indexの集合を毎回使う。最良1・最新1を保持し、完了時に最終版も保存する。

GPU試運転は6更新＋検証2バッチに合格。初回はGPU初期化でOOMになった。OSのMemAvailable約70GiBに対しCUDAの空きは約7GBだったため、今回の取得データのclean page cacheだけにPOSIX_FADV_DONTNEEDを適用し、MemFreeを約7.4GiBから40.4GiBに回復させて再試行した。原本の変更・削除や他のGPUジョブの停止はしていない。

再試運転の中央値は初回を除き0.652秒/更新、GPU最大allocated13.31GiB・reserved13.74GiB。損失は有限でIDM/LaWM/gに非ゼロの有限勾配、DINOの勾配0を確認。検証損失1.18577は6更新時点の動作確認値で、改善の評価ではない。既存の出力倍率テスト8件も合格。

2026-11-14 01:19:48 JSTに本学習を起動。実体はSSD1の `foveated-world-models/runs/robotwin-qfix-delta2-b4-20261114`。進捗は同ディレクトリの `progress.json`、検証結果は `validation.jsonl`、checkpointは `checkpoints/`。試運転結果は兄弟ディレクトリ `robotwin-qfix-delta2-b4-preflight2-20261114/run_summary.json`。本学習の完了・性能改善はまだ確認していない。
