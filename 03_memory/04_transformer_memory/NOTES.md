# 研究ノート: Transformer Memory

日付: 2026-08-22。将来の記事の材料であり、記事本文ではない。

## 実装前の疑問と予想

- Transformer memoryは本当にmemoryか、長いinput tensorにすぎないか。
- actionをobservation latentのどこへ置くか。
- future leakを仮定でなくどうtestするか。
- attentionは最初に見えたGoalをretrieveして見せるか。

causal teacher forcingはparallel、rolloutはcontext成長のため逐次・高コスト、6 stepではGRU優位を覆すには短すぎる、と予想した。

## 実装で明確になったこと

- memoryはretained key/value token contextであり、切り捨てたtokenはmodel widthに関わらず忘れる。
- Transformerはrecurrent compressionをなくすが、有限context policyをなくさない。
- `[z_t,a_t]`は「この状態でこのactionをした結果が`z_{t+1}`」という明確なtransition tokenになる。
- positionはcontent/actionとは別の情報源である。

## エラーと修正

- RSSMとTransformerが同じ`test_dataset_contract.py`を持ち、packageでない独立folderではpytest collectionが衝突した。Transformer側を`test_transformer_dataset_contract.py`へrenameし、combined suiteは29 testsになった。
- 初期evaluation JSONはRSSMからコピーした`posterior_autoencoder_reconstruction_mse`という名前だった。Transformerにはposteriorがないため`autoencoder_reconstruction_mse`へ修正した。比較資料の語彙は実装機構と一致させる必要がある。

## 結果と気づき

- seed 29、406,794 parameters、160 steps。validation total 0.035018。
- one-step / six-step rollout Goal accuracyは100%。pixel MSEは0.000964。
- future側attention weightは厳密に0。後のqueryは一つのtokenだけでなく複数の履歴へweightを分配した。
- Goal消失後にも薄いgreen traceが残り、semantic headとimage fidelityを分けて読む必要がある。

## 記事材料と未解決点

- 「GRUはnotebookを持ち運び、attentionは過去ページの棚から必要なページを開く」。ただし棚はcontext長で有限。
- `attention_map.png`はrow=query、column=利用可能key履歴、upper triangle=0と説明する。
- attention可視化はavailability/weightの診断であり、causal importanceの証明ではない。
- 長いdelayed cue、history truncation/shuffle、position/action removal、per-head分析、teacher-forced throughputとrollout latencyを比較する。
