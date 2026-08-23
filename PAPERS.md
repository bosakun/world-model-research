# 論文台帳

最終更新: 2026-08-23

状態: `Not Read`（未読）、`Overview Read`（概要読了）、`Reading`（読書中）、`Deep Read`（精読）、`Implemented`（主要機構を実装）、`Re-read Needed`（再読が必要）。

ここでの`Implemented`は「論文全体を完全再現した」意味ではない。各実験READMEに、どの機構を採用し、何を簡略化・変更したかを記録している。

## 基礎

| 状態 | 論文 | 著者・年 | 一次資料 | このリポジトリとの関係 |
|---|---|---|---|---|
| Overview Read | World Models / Recurrent World Models Facilitate Policy Evolution | David Ha, Jürgen Schmidhuber, 2018 | https://arxiv.org/abs/1803.10122 / https://arxiv.org/abs/1809.01999 | 画像圧縮、recurrent temporal model、imaginationの出発点 |
| Implemented | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | Kyunghyun Cho et al., 2014 | https://aclanthology.org/D14-1179/ | `03_memory/01_gru`のGRUCellの由来 |

## 潜在変数 / RSSM

| 状態 | 論文 | 著者・年 | 一次資料 | 関係 |
|---|---|---|---|---|
| Implemented | Auto-Encoding Variational Bayes | Diederik P. Kingma, Max Welling, 2013 | https://arxiv.org/abs/1312.6114 | reparameterizationとdiagonal Gaussian KL |
| Implemented | Learning Latent Dynamics for Planning from Pixels (PlaNet) | Danijar Hafner et al., 2018 | https://arxiv.org/abs/1811.04551 | RSSM、overshooting、Phase 99のRSSM planning。すべて教育用簡略実装 |

## 部分観測 / Imagination

| 状態 | 論文 | 著者・年 | 一次資料 | 関係 |
|---|---|---|---|---|
| Overview Read | Planning and Acting in Partially Observable Stochastic Domains | Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra, 1998 | https://doi.org/10.1016/S0004-3702(98)00023-X | stateとobservationの区別、POMDPの直感 |
| Implemented | Dream to Control: Learning Behaviors by Latent Imagination | Danijar Hafner et al., 2019 | https://arxiv.org/abs/1912.01603 | Phase 08のsimplified actor-critic、Phase 99のlatent reward/value imagination |
| Overview Read | Mastering Atari with Discrete World Models (DreamerV2) | Danijar Hafner et al., 2020 | https://arxiv.org/abs/2010.02193 | discrete latentの違いを記録。未実装 |
| Overview Read | Mastering Diverse Domains through World Models (DreamerV3) | Danijar Hafner et al., 2023 | https://arxiv.org/abs/2301.04104 | cross-domain stabilizationを資料化。完全実装なし |

## 不確実性 / Planning

| 状態 | 論文 | 著者・年 | 一次資料 | 関係 |
|---|---|---|---|---|
| Implemented | Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS) | Kurtland Chua et al., 2018 | https://arxiv.org/abs/1805.12114 | probabilistic ensemble、trajectory sampling、Phase 99のrisk-aware MPC。PETS再現ではない |
| Implemented | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? | Alex Kendall, Yarin Gal, 2017 | https://arxiv.org/abs/1703.04977 | aleatoric / epistemicの区別 |
| Reading | MuZero | Julian Schrittwieser et al., 2020 | https://doi.org/10.1038/s41586-020-03051-4 | decoder-free planningの系譜。MCTS未実装 |
| Implemented | TD-MPC2 | Nicklas Hansen, Hao Su, Xiaolong Wang, 2023 | https://arxiv.org/abs/2310.16828 | task-oriented latent planningの簡略実装 |

## Memory / Object / Video / Physical AI

| 状態 | 論文 | 著者・年 | 一次資料 | 関係 |
|---|---|---|---|---|
| Implemented | TransDreamer | Chang Chen et al., 2022 | https://arxiv.org/abs/2202.09481 | causal Transformer memory。TSSM・policyは未実装 |
| Implemented | Attention Is All You Need | Ashish Vaswani et al., 2017 | https://arxiv.org/abs/1706.03762 | multi-head attention、position、causal mask |
| Implemented | Contrastive Learning of Structured World Models (C-SWM) | Thomas Kipf, Elise van der Pol, Max Welling, 2019 | https://arxiv.org/abs/1911.12247 | object slotとrelation dynamicsの簡略実装 |
| Implemented | Object-Centric Learning with Slot Attention | Francesco Locatello et al., 2020 | https://arxiv.org/abs/2006.15055 | Slot Attention。object discovery smoke runは失敗を保持 |
| Implemented | SlotFormer | Ziyi Wu et al., 2022 | https://arxiv.org/abs/2210.05861 | ordered slotのtemporal Transformer |
| Implemented | Neural Discrete Representation Learning (VQ-VAE) | Aaron van den Oord et al., 2017 | https://arxiv.org/abs/1711.00937 | discrete visual tokenizer |
| Implemented | Genie | Jake Bruce et al., 2024 | https://arxiv.org/abs/2402.15391 | VQ token、latent action、interactive dynamics。semantic alignmentは未解決 |
| Implemented | Learning Interactive Real-World Simulators (UniSim) | Sherry Yang et al., 2023 | https://arxiv.org/abs/2310.06114 | typed-adapter heterogeneous simulator |
| Implemented | V-JEPA 2 | Mido Assran et al., 2025 | https://arxiv.org/abs/2506.09985 | action-conditioned JEPAの教育用縮小実装 |
| Implemented | DayDreamer | Philipp Wu et al., 2022 | https://arxiv.org/abs/2206.14176 | physical replay/interfaceとaction boundaryの動機 |
| Implemented | OccWorld | Wenzhao Zheng et al., 2023 | https://arxiv.org/abs/2311.16038 | compact 3D occupancy rollout |

## 読み方

初学者は、GRU → POMDP → VAE → PlaNet → Dreamer/PETS/Transformerの順に進むと、なぜ`h_t`、`z_t`、prior/posterior、KL、planningが必要になるかを段階的に追いやすい。
