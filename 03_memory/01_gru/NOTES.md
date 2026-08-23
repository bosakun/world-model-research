# 研究ノート: GRU潜在ダイナミクス

開始日: 2026-08-22

## 最初の疑問と予想

- 完全観測なら、GRUは何を`z_t`以上に覚えるのか。
- teacher forcing時のhidden更新とautoregressive rolloutは何が違うのか。
- GRUは学習・rolloutできるが、memory自体は原理的に冗長だろうと予想した。
- 予測latentを自分自身へ戻すため、horizonが長いほど画像は悪化すると予想した。

## 監査で分かったこと

checkoutには既存Grid World、autoencoder、latent dynamicsのコードがなかった。既存実装の再利用ではなく、self-containedな実験基盤を作り、`latent_dim=16`などを新規選択として記録する必要があった。

## 実装上の選択

- `hidden_dim=64`、`latent_dim=16`、`GRUCell`を採用した。
- one-hot actionの順序はup/down/left/right。
- 正解latentを使うteacher forcing学習と、別のautoregressive rollout評価を分けた。
- Simple Dynamicsは残したが、random weightとの不公平な比較結果は作らなかった。

## つまずきと修正

1. plain MSEはAgentを消しても低lossになった。小さなmoving objectより背景を復元するshortcutが有利だった。
2. active pixelの重み付けでもagent-cell accuracyは約3.9%（1/25）だった。
3. Goal/Agentの色を強く重み付けすると、Decoderが多くのセルを赤く塗るshortcutを選んだ。
4. 最終的にfull-frame MSEへ戻し、25-way agent-cell cross entropyを追加した。小さいtransposed-convolution decoderも位置を復元できず、教育環境ではMLP decoderの方が意味を追いやすかった。

## 結果と気づき

- one-step agent-cell accuracyは83.59%だったが、8-step平均は55.86%まで下がった。
- pixel MSEと意味的な位置精度は別指標として必ず見るべき。
- one-stepが良くてもimaginationの誤差は蓄積する。
- 完全観測では、GRUが動いても「memoryが効いた」とは言えない。

## 記事に使えそうな材料

- 「GRUが動く」ことと「memoryが必要」なことは別。
- `z_t`は現在の知覚、`h_t`は履歴依存のbelief/context。
- teacher forcingはrollout失敗を隠し得る。
- `outputs/loss_curve.png`、`rollout_comparison.png`、`rollout_error.png`。

## 解釈上の注意

- PlaNet/Dreamerと呼ばない。MDN-RNN、VAE、RSSM、KL、controllerはない。
- optimized GRUとuntrained baselineを比較しない。
- one seedのsmoke runをrobustな優位性と扱わない。
