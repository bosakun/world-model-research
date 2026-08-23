# 異なるsimulatorデータを一つのworld modelへつなぐ

## 何が困るのか

simulatorごとにstate、action、camera、座標の単位が違います。そのまま混ぜると、同じ数字でも意味が違うことがあります。

## interfaceの役割

各データについて「何が観測か」「actionは何を意味するか」「単位は何か」を明示し、共通のDynamics interfaceへ変換します。

    simulator固有の形式
    -> 型と意味を確認するadapter
    -> 共通のworld model入力

## 注意点

形式を揃えるだけでは意味が揃いません。右というactionがカメラ移動なのかロボット移動なのか、座標がmeterなのかpixelなのかを記録する必要があります。

## 自分で説明できるか

- 異なるsimulatorをそのまま混ぜると何が危険か。
- adapterは何を明示するか。
- 数字のshapeが同じでも意味が違う例は何か。
