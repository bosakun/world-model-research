# Paper Registry

Last updated: 2026-08-22

Statuses: `Not Read`, `Overview Read`, `Reading`, `Deep Read`, `Implemented`, `Re-read Needed`.

The year below follows the first public paper/proceedings record. Metadata and URLs were checked against arXiv, official proceedings, ACL Anthology, or the publisher; status describes this project, not paper quality.

## Foundations

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Overview Read | World Models / Recurrent World Models Facilitate Policy Evolution | David Ha, Jürgen Schmidhuber | 2018 | https://arxiv.org/abs/1803.10122 and https://arxiv.org/abs/1809.01999 | Compressed observation model, recurrent temporal model, imagined environment |
| Implemented | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio | 2014 | https://aclanthology.org/D14-1179/ (DOI: 10.3115/v1/D14-1179) | GRU origin; `03_memory/01_gru/model.py` uses PyTorch's GRUCell equations |

## Latent Variables

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | Auto-Encoding Variational Bayes | Diederik P. Kingma, Max Welling | 2013 | https://arxiv.org/abs/1312.6114 | Variational latent variables, reparameterization, KL |

## Latent Dynamics / RSSM

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Overview Read | Learning Latent Dynamics for Planning from Pixels (PlaNet) | Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson | 2018 | https://arxiv.org/abs/1811.04551 | RSSM, deterministic/stochastic state, prior/posterior, latent overshooting |

## Imagination

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Overview Read | Dream to Control: Learning Behaviors by Latent Imagination | Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi | 2019 | https://arxiv.org/abs/1912.01603 | Recurrent state-space world model and latent imagination |
| Not Read | Mastering Atari with Discrete World Models (DreamerV2) | Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba | 2020 | https://arxiv.org/abs/2010.02193 | Discrete latent world model and imagined behavior learning |
| Not Read | Mastering Diverse Domains through World Models (DreamerV3) | Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap | 2023 | https://arxiv.org/abs/2301.04104 | Robust scaling, normalization, balancing, transformations |

## Uncertainty

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS) | Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine | 2018 | https://arxiv.org/abs/1805.12114 | Probabilistic ensembles and trajectory sampling |

## Planning

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero) | Julian Schrittwieser et al. | 2020 | https://doi.org/10.1038/s41586-020-03051-4 | Planning-relevant representation, dynamics, reward/value/policy, MCTS |
| Not Read | TD-MPC2: Scalable, Robust World Models for Continuous Control | Nicklas Hansen, Hao Su, Xiaolong Wang | 2023 | https://arxiv.org/abs/2310.16828 | Decoder-free latent model, local trajectory optimization, scaling |

## Transformer World Models

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | TransDreamer: Reinforcement Learning with Transformer World Models | Chang Chen, Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn | 2022 | https://arxiv.org/abs/2202.09481 | Transformer State-Space Model and long-range memory |

## Object-Centric World Models

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | Contrastive Learning of Structured World Models | Thomas Kipf, Elise van der Pol, Max Welling | 2019 | https://arxiv.org/abs/1911.12247 | Object-factorized representations and relational dynamics |
| Not Read | Object-Centric Learning with Slot Attention | Francesco Locatello et al. | 2020 | https://arxiv.org/abs/2006.15055 | Competitive iterative object slots |
| Not Read | SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models | Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg | 2022 | https://arxiv.org/abs/2210.05861 | Transformer dynamics over object-centric slots |

## Generative / Video World Models

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | Genie: Generative Interactive Environments | Jake Bruce et al. | 2024 | https://arxiv.org/abs/2402.15391 | Video tokenizer, latent action model, autoregressive interactive dynamics |
| Not Read | Learning Interactive Real-World Simulators (UniSim) | Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, Pieter Abbeel | 2023 | https://arxiv.org/abs/2310.06114 | Generative simulator trained across heterogeneous data |

## Physical World Models

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | Mido Assran et al. | 2025 | https://arxiv.org/abs/2506.09985 | Latent video prediction and action-conditioned robot planning |
| Not Read | DayDreamer: World Models for Physical Robot Learning | Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, Pieter Abbeel | 2022 | https://arxiv.org/abs/2206.14176 | Online Dreamer learning on physical robots |
| Not Read | OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving | Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu | 2023 | https://arxiv.org/abs/2311.16038 | 3D occupancy tokens, future scene and ego trajectory generation |

## Safety

| Status | Paper | Authors | Year | Primary source | Roadmap relevance |
|---|---|---|---:|---|---|
| Not Read | The Safety Challenge of World Models for Embodied AI Agents: A Review | Lorenzo Baraldi et al. | 2025 | https://arxiv.org/abs/2510.05865 | Safety pathologies in scene/control generation |
| Not Read | Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation | Fazhong Liu, Zhuoyan Chen, Haozhen Tan, Yan Meng, Guoxing Chen, Haojin Zhu | 2026 | https://arxiv.org/abs/2607.28226 | Lifecycle threat model, defenses, and evaluation; very recent, re-check revisions before use |

