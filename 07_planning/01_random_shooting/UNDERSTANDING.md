# Random Shootingを理解する

## 何をするか

実行前に、ランダムなaction列をたくさん作り、world modelで未来を予測して最も高いscoreの列を選びます。

    候補をランダムに作る
    -> 未来を想像して採点
    -> 一番良い候補の最初のactionを実行

## なぜ必要か

world modelは「このactionなら未来はこうなる」を予測できます。Random Shootingは、その予測を使って行動を選ぶ一番単純な方法です。

## 弱点

候補が少ないと良いaction列を引けません。actionの種類や計画時間が増えるほど、ランダムに当てるのは難しくなります。

## 自分で説明できるか

- Random Shootingは何をrandomに作るか。
- なぜ最初のactionだけ実行することが多いか。
- candidate数を増やすと何が増えるか。
