# Step 0: 既存リポジトリ監査

監査日: 2026-08-22（Asia/Tokyo）。

## なぜ監査したか

ユーザーの記憶ではGrid World、autoencoder、latent dynamicsなどが実装済みだった。しかし、存在しないコードを「実装済み」と仮定すると、比較も再現もできない。そこでfilesystemとGit履歴を実際に確認した。

## 監査時に確認できたもの

監査時のrepositoryには`.git/`だけがあり、`main`にもcommitがなかった。source、test、dataset、output、依存関係ファイルは確認できなかった。

したがって、次の機能は「以前作った認識はあるが、このcheckoutでは確認不能」と記録した。

- Grid Worldと状態遷移
- `(state, action) -> next state`
- dataset generationとmulti-step rollout
- 画像観測、Encoder、Decoder、autoencoder
- latent state、latent dynamics、latent rollout

## 重要な判断

存在しないbaselineを勝手に復元済みと呼ばず、`03_memory/01_gru`をself-containedな実験として作った。もし旧コードが別branch・別directory・未commit workspaceから見つかった場合も、現在の実験を壊さず`01_basic_dynamics`または`02_visual_latent`へ取り込み、interfaceと結果を比較する。

## 当時の不足と、その後の対応

| 監査時の不足 | その後の対応 |
|---|---|
| dependency宣言なし | `pyproject.toml`と`uv.lock` |
| test・seed・metricsなし | 各実験のtests、config、outputs |
| 数式とコード対応なし | README / UNDERSTANDING / NOTES |
| baseline比較なし | Phase 90の3-seed memory benchmark |
| 研究の追跡方法なし | `RESEARCH_ROADMAP.md`と`PAPERS.md` |

この監査は「過去の作業を否定する」ためではなく、再現可能な研究記録の出発点を明確にするためのものである。
