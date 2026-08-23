# VQ tokenで動画を記号列として扱う

## 何が困るのか

動画の全pixelを次々に予測するのは重く、少しの見た目のズレでもlossが大きくなります。

## VQ tokenizer

画像の部分を、学習した視覚の単語帳にある番号へ置き換えます。

    画像 -> token番号の列
    次のtoken番号を予測
    token列 -> 画像

world modelはpixel値でなくtokenを予測します。

## 注意点

tokenizerが大事な物体や動きをうまく表せなければ、後のDynamics Modelも予測できません。token化はただの圧縮ではなく、何を世界モデルが見られるかを決める部品です。

## 自分で説明できるか

- VQ tokenは画像を何へ変えるか。
- pixel予測より扱いやすくなる理由は何か。
- tokenizerが悪いと何が起きるか。
