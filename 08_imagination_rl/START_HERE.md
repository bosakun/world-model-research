# はじめにここから: 08 Imagination RL

Planningでは毎回たくさんのaction列を試しました。Imagination RLでは、world modelの中で想像した未来から、行動方策そのものを学習します。

    world model
    -> imagined trajectory
    -> actor（actionを出す）
    -> critic（将来の良さを評価する）

最初は[01_actor_critic](01_actor_critic/README.md)、次に[ARTICLE.md](ARTICLE.md)を読んでください。
