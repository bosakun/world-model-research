# はじめにここから: 12 Physical AI

ここでは、学習したworld modelをシミュレータの外、実際の機械やロボットへつなぐときに必要な境界を学びます。

## 学ぶ順番

1. [01_action_conditioned_jepa](01_action_conditioned_jepa/README.md): action後に予測可能な部分を表現空間で予測する。
2. [02_robot_interface](02_robot_interface/README.md): 学習modelのaction要求を、安全な物理実行へつなぐ。
3. [ARTICLE.md](ARTICLE.md)

実機では「モデルがactionを出せた」だけで実行してはいけません。制限、停止、記録、監視をinterfaceとして明示する必要があります。
