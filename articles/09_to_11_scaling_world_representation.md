# 世界を何として表現するか：物体、動画token、複数感覚

## この記事でつなぐもの

一つのlatent vectorへ画像全体を圧縮する方法は便利です。しかし複数物体、長い動画、複数sensorを扱うと、重要な情報が混ざりやすくなります。

この記事は09 Spatial Representation、10 Video World Model、11 Multimodalをまとめます。

## 1. 物体ごとの表現

C-SWMは、物体ごとのstateと物体間のrelationから未来を予測します。

    object A state + object B state + relation
    -> next object states

Slot Attentionは、画像特徴を複数slotへ集め、教師なしで物体らしいまとまりを作ろうとします。SlotFormerは複数slotの時間変化をTransformerで予測します。

重要なのは、再構成画像が良いことと、slotが本当に同じ物体を追っていることは別だという点です。slotの入れ替わり、背景への偏り、object単位のlong rolloutを確認します。

3D occupancyは、画像でなく「3D空間のどこが物で埋まっているか」を表します。ロボットの通行可能空間を考えるときに直接的ですが、遮蔽された場所は推測になり、memoryも増えます。

詳細: [09 Spatial Representation](../09_spatial_representation/START_HERE.md)

## 2. 動画をtokenとして扱う

高解像度動画の全pixelを予測するのは重い処理です。VQ video tokenizerは、画像をcodebookのtoken ID列へ変えます。

    image -> token IDs
    token IDs -> next token IDs
    token IDs -> image

world modelはpixelでなく視覚tokenを予測します。ただしtokenizerがAgent、物体、動きを失えば、後段のDynamicsは回復できません。

action記録がない動画では、前後状態の差からlatent actionを推論します。これは人間の「右へ動く」のような名前付きactionとは限らず、変化を説明する内部変数です。camera移動と物体操作を混同していないかを評価します。

複数simulatorを混ぜるときは、shapeだけでなく座標系、単位、actionの意味をadapterで記録します。

詳細: [10 Video World Model](../10_video_world_model/START_HERE.md)

## 3. 複数感覚は同じ世界の別の手掛かり

現実のAgentは画像だけでなく、関節角度、速度、接触力、音などを使えます。

    vision encoder
    proprioception encoder
    touch encoder
    + sensorが欠けたことを示すmask
    -> fusion -> shared latent

fusionは数字をただ連結することではありません。各感覚が何を見せ、何が欠けているかを区別します。

全sensorがある場合だけでなく、cameraやtouchを落とした場合も評価します。特定のsensorにだけ依存していないかを確認するためです。

詳細: [11 Multimodal](../11_multimodal/ARTICLE.md)

## まとめ

- object-centric表現は、物体と関係を明示しやすい。
- video tokenは、予測対象をpixelから記号列へ変える。
- latent actionは、action記録のない動画の変化を扱う。
- multimodal fusionは、異なる感覚を同じ世界の手掛かりとして統合する。
- 良い表現とは、きれいに復元できるだけでなく、Dynamicsとcontrolに必要な情報が残る表現である。
