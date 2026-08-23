# latent spaceでPlanningする

## 何が困るのか

毎step、latentを画像へ戻してから未来を採点すると計算が重くなります。また、action選択に不要なpixelまで復元する必要がある場合があります。

## latent planning

    latent state + action列
    -> latent dynamicsで未来latent
    -> reward/value headで採点
    -> 良いactionを選ぶ

画像を復元せず、内部のlatent stateで未来を比べます。

## 大切な条件

latentにcontrolに必要な情報が残っていなければ、速く計画できても間違ったactionを選びます。復元画像がきれいかだけでなく、rewardやvalueを予測できるかを確認する必要があります。

## 自分で説明できるか

- latent planningは何を省くか。
- なぜ速くなる可能性があるか。
- latentが悪いと、planningはどう失敗するか。
