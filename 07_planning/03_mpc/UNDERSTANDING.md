# MPCはなぜ計画を作り直すのか

## 一度の長い計画の問題

world modelの未来予測には誤差があります。最初に10stepのaction列を決めて全部実行すると、途中で予測が外れても修正できません。

## MPCの流れ

    現在から未来を計画する
    -> 最初のactionだけ実行する
    -> 本物の次の観測を受け取る
    -> その観測からもう一度計画する

これをreceding horizon controlとも呼びます。

## なぜ有効か

本物の観測を毎step取り直すことで、world modelの予測誤差を修正できます。ただし毎stepでplanningするので計算量は増えます。

## 自分で説明できるか

- open-loop計画とは何か。
- MPCで最初のactionだけ実行する理由は何か。
- MPCでもworld modelが悪すぎると困る理由は何か。
