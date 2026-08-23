# action記録のない動画から変化を表す

## 何が困るのか

多くの動画には「誰がどのactionをしたか」の記録がありません。world modelは通常actionを必要とします。

## latent action

前後の状態の差から、変化を説明する内部actionを推測します。

    前の画像 + 後の画像
    -> latent action
    前の状態 + latent action
    -> 後の状態を予測

latent actionは人間が命名した「右へ移動」などと一致するとは限りません。予測に役立つ変化の符号です。

## 注意点

カメラ移動、照明変化、物体の動きを混同する可能性があります。latent actionが本当に原因を表しているかは、介入や生成結果で確認する必要があります。

## 自分で説明できるか

- なぜ動画だけでは通常のactionがないのか。
- latent actionは何から推測するか。
- latent actionを人間のaction名と同一視できない理由は何か。
