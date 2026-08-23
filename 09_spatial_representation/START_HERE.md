# はじめにここから: 09 Spatial / Object-Centric Representation

これまでのlatent zは、画像全体を一つの数字の並びへ圧縮しました。しかし、複数の物体がある世界では「どの数字がどの物体か」が分かりにくくなります。

このフェーズでは、物体ごとの表現や3D空間の表現を学びます。

## 学ぶ順番

1. [01_cswm](01_cswm/README.md): 物体と物体の関係を別々に表す。
2. [02_slot_attention](02_slot_attention/README.md): 教師なしで画像をslotへ分ける。
3. [03_slotformer](03_slotformer/README.md): 複数slotの時間変化を予測する。
4. [04_occupancy_3d](04_occupancy_3d/README.md): 3D空間のどこが物で埋まっているかを表す。
5. [ARTICLE.md](ARTICLE.md)

ここでの目標は、「一つのlatent vectorでは、複数物体のどんな情報が混ざってしまうか」を説明できることです。
