# はじめにここから: 07 Planning

Planningは、world modelの中で複数のaction候補の未来を想像し、良さそうな最初のactionを選ぶことです。

    action候補を作る
    -> それぞれの未来をworld modelで予測
    -> reward / valueで採点
    -> 最初のactionだけ実行

## 学ぶ順番

1. [01_random_shooting](01_random_shooting/README.md): ランダムな候補をたくさん試す。
2. [02_cem](02_cem/README.md): 良い候補の周辺を集中的に試す。
3. [03_mpc](03_mpc/README.md): 一歩実行するたびに計画を立て直す。
4. [04_latent_planning](04_latent_planning/README.md): 画像を復元せずlatent内で採点する。
5. [ARTICLE.md](ARTICLE.md)

重要なのは、Planningは「未来を完璧に当てる」ことではなく、不完全な予測を使って次の一手をより良く選ぶことだ、という点です。
