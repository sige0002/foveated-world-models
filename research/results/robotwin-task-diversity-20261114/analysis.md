# 実投入データのタスク多様性（2026-11-14）

単一タスクだけを繰り返すloaderの形跡はない。元の3000更新×batch4では9096 episode、7856種類の指示文IDを使用し、フレーム一様抽選から期待される7853.54種類と近い。教師のΔXYZは約4.7有効次元で、μの約1次元への集中を教師自体の一次元性だけでは説明できない。

| 対象 | pair投入数 | 異なるpair | episode数 | 指示文ID数 | 最大IDの割合 | ΔXYZ有効次元 | 最大主成分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 元の3000更新 | 12000 | 11988 | 9096 | 7856 | 0.183% | 4.717 | 32.02% |
| 再利用pool | 12000 | 11988 | 9096 | 7856 | 0.183% | 4.717 | 32.02% |
| 新規pool | 12000 | 12000 | 9171 | 7937 | 0.183% | 4.702 | 31.89% |
| 共通検証 | 512 | 512 | 415 | 410 | 0.977% | 4.919 | 29.08% |

全train splitは26125 episode・5776302フレーム・22488指示文ID。元の11988種類のpairは、各trainフレームをbaseとする全候補の約0.208%で、全データを学習済みではない。元のpoolは73.39%がどちらかの腕で5cm超の移動を含み、静止画像だけにも偏っていない。共通検証のΔXYZを実Parquetから再構成すると、保存済み教師との差は完全に0だった。

指示文IDは言い換えも別IDであり、物理的なタスク種類数とは異なる。全27500 episodeのaction_config.skillは空で、固定metadataだけから正確なタスクファミリー数は復元できない。実投入にはハンマーで叩く、引き出しに物を入れる、注ぐ、ブロックを積む等の指示が含まれる。[具体例](instruction-examples.json)。元poolの上位10指示文の合計は1.50%。loaderは指示文一様ではなくフレーム一様なので、長いepisodeほど選ばれやすい。

公開重みの固定YAMLは `data_mix=lam_plus_human` だが、固定commit c12168d9077edea67fe7c1b70d08b80b8e65a9c8 のmixtures.pyにその定義がない。`robotwin_merged`単独mixtureと著者公開datasetは確認できるが、この公開重みが実際に使った全mixtureの構成・混合比・同一episodeの含有は証明できない。「公式と同じtraining dataset全体」とは断定しない。

統合metadataのstats/task_indexは27000 episodeで元データのlocal IDが残っていた。集計ではepisodeの指示文をglobal tasks表へ対応づけ、選択した全48274 endpointの実Parquet task_indexと照合した。Stage1は言語/task IDを入力せず、これをz不調の原因とは判定していない。

[全数値・pin照合](analysis.json)、[全指示文の分布CSV](instruction-distribution.csv)。10個の固定Parquetを5列だけ65536行ずつ読み、必要endpointだけ保持した。動画decode・GPU・学習変更なし。
