# 長いrolloutを学ばせるlatent overshooting

## 何が困るのか

1step予測を正しく学べても、長い未来では前の予測を次の入力に使います。

    正解state -> 1step予測
    予測state -> 2step予測
    予測state -> 3step予測

小さな誤差が入力へ戻るため、先へ行くほど誤差が積み重なります。これがcompounding errorです。

## overshootingの考え方

通常の1step lossに加えて、2step先、3step先などのlatentも直接近づけます。

    今のlatentから3step想像した結果
    と
    3step後の画像を見て推論したlatent
    を近づける

これによりDynamics Modelは「次の一歩だけ当てる」のでなく、「数step後も破綻しにくい」方向へ学びます。

## 大切な区別

学習時には将来画像を答え合わせに使います。しかしrollout時には未来画像を入力へ入れません。未来画像を使えば、それは想像ではなく答えを見ながら進む処理になります。

## 外すとどうなるか

- overshootingを外す: 1stepが良くても長期rolloutが崩れやすい。
- 未来画像をrolloutの入力に使う: 評価が不公平になり、実際の想像能力を測れない。
- horizonを大きくしすぎる: 遠い予測のlossが不安定になり、計算も増える。

## 自分で説明できるか

- 1step精度だけで長期精度を保証できない理由は何か。
- overshootingはどの二つのlatentを近づけるか。
- 学習時の答え合わせとrollout時の入力はどう違うか。
