# 物理世界へ出し、比較して、根拠を持って統合する

## この記事でつなぐもの

世界モデルを実機へつなぐには、prediction精度だけでは足りません。

    controlに必要な表現か
    actionを安全に実行できるか
    複数方式を公平に比較したか
    統合時に何を採用するか

この記事は12 Physical AI、90 Evaluation、99 Integrated World Modelをまとめます。

## 1. 全pixel復元だけが世界モデルではない

Action-Conditioned JEPAは、未来画像の全pixelを描く代わりに、action後の未来表現を予測します。

    context representation + action
    -> future representation

背景の細かな模様より、物体位置や接触のようなcontrolに重要な変化へ集中したい考え方です。

ただし、pixelを復元しないからといって重要情報を捨ててよいわけではありません。障害物や安全に必要な情報が表現に残るかをdownstream taskで確認します。

## 2. modelのactionはrobot commandではない

学習modelは範囲外の数値や危険なactionを出せます。

    model action request
    -> unit / schema validation
    -> speed, position, force limits
    -> emergency stop確認
    -> robot execute
    -> observation, action, resultを記録

Robot Interfaceは単なる配線ではなく、安全装置と再現可能なdata collectionの境界です。simulationで成功しても、実機には摩擦、遅延、sensor noise、予想外の接触があります。

詳細: [12 Physical AI](../12_physical_ai/START_HERE.md)

## 3. 比較しなければ「効いた部品」は分からない

例えばMemoryなら、Partial Observation環境で次を同条件で比べます。

    Memoryなし
    vs GRU
    vs RSSM
    vs Transformer Memory

見るのはone-step errorだけではありません。

- Goalが視界外になった後の予測
- 5step、10stepのrollout error
- parameter数、推論時間、memory使用量
- 複数seedでの安定性
- 部品を外すablation

複雑なmodelが良く見えても、parameter数やtraining量が多いだけかもしれません。比較とablationが、採否判断の根拠になります。

詳細: [90 Evaluation](../90_evaluation/01_memory_benchmark/README.md)

## 4. 統合modelは全部入りではない

統合modelは、これまで有効性または必要性を確認できた部品を選んでつなぎます。

    observation
    -> representation
    -> memory / stochastic state
    -> dynamics / uncertainty
    -> imagination
    -> reward / value
    -> planning or policy
    -> guarded action

RSSMのsampling、ensemble particle、planning candidateを全部増やすと計算量は急増します。したがって各部品について、採用理由、比較結果、cost、採用しなかった方式を記録します。

詳細: [99 Integrated World Model](../99_integrated_world_model/01_evidence_selected/README.md)

## まとめ

- 実機では、predictionより先に安全なaction境界が必要になる。
- 比較とablationなしに、複雑な部品の価値は判断できない。
- 統合modelは、すべてを足す設計ではなく、根拠を持って選ぶ設計である。
