# CEMで良いaction候補を絞り込む

## Random Shootingの次の問題

ランダム探索では、良いaction列の近くに候補があっても、次の試行でその近くへ集中できません。

## CEMの流れ

    広く候補をsampleする
    -> scoreの高い少数の候補を残す
    -> 残した候補の分布を調べる
    -> その近くから次の候補をsampleする

これを数回繰り返し、良いaction列の周辺を絞ります。

## 何に注意するか

CEMはworld modelが高得点だと予測する場所へ進みます。world modelの誤りを利用するactionを見つける危険もあります。候補数、繰り返し回数、上位何個を残すかが結果に影響します。

## 自分で説明できるか

- CEMはRandom Shootingのどの弱点を補うか。
- elite candidateとは何か。
- なぜmodel誤差があると危険か。
