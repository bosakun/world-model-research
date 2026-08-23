# Understanding the Evidence-selected Integrated World Model

## What problem does this solve?

この実験が解くのは、部分観測画像から行動を選ぶまでのinterface問題である。Goalが視界外に消えた後、現在画像だけでは右Goalと下Goalを区別できない。過去をbelief stateへ保持し、そのstateから未来を想像し、task outcomeを評価して行動を選ぶ必要がある。

また、学習時のposteriorと計画時のpriorの間にあるdistribution shiftも解く。観測があるfeatureだけでtask headを学習しても、未来想像で使う観測なしfeature上で正しいとは限らない。

## Before

Phase 03–12には個別部品が存在したが、同じstate表現とtask上で端から端まで接続されていなかった。Memoryの比較は「hidden Goalを読めるか」までで、実際にGoalへ行動するかは未確認だった。不確実性、task head、planner、physical guardも別環境・別latent上にあった。

## After

一つのRSSM stateが以下を接続する。

```text
partial pixels
  -> posterior belief update
  -> prior ensemble imagination
  -> reward/value/disagreement
  -> receding-horizon action
  -> guarded environment transition
  -> new posterior belief update
```

同じ現在画像でも、過去に見たGoalが異なるためbelief stateが異なり、plannerは右または下を選ぶ。

## Core Idea

統合モデルの核心は「未来に観測できない情報を、過去から作ったbeliefと学習済みpriorで補う」ことである。

- `h_t`: actionと過去latentを再帰的に圧縮した決定論的history。
- `z_t`: 現在時刻の確率的state。学習時は画像で補正し、想像時はpriorで予測する。
- posterior: `h_t`と現在画像の両方を使うstate inference。
- prior: `h_t`だけから観測なし未来を作るdynamics。
- ensemble: 複数priorが候補actionのreturnにどれほど同意するかを見る。
- MPC: 一度に全trajectoryを実行せず、先頭actionだけ実行して観測で再補正する。

## Data Flow

### Filtering

```text
(h_{t-1}, z_{t-1}, a_{t-1})
        │
        └── GRUCell ──> h_t
observation o_t ── CNN ──> e_t
(h_t, e_t) ── posterior ──> z_t
```

### Imagination and control

```text
filtered (h_t,z_t)
  ├─ candidate action sequence 1 ─┬─ prior 1 ─ return R_1
  │                               ├─ prior 2 ─ return R_2
  │                               └─ prior 3 ─ return R_3
  └─ ... 512 candidates

score = mean(R) - beta * std(R)
best candidate -> first action -> guard -> environment
```

## Mathematics

### Deterministic recurrent state

\[
h_t=\operatorname{GRUCell}([z_{t-1},a_{t-1}],h_{t-1})
\]

- `h_t`: `[B,64]`のhistory state。
- `z_{t-1}`: 前時刻のstochastic state。
- `a_{t-1}`: `[B,4]` one-hot action。
- 必要性: 同じ現在画像に至った異なる履歴を区別するため。

### Posterior

\[
q(z_t\mid h_t,o_t)=\mathcal N(\mu^q_t,(\sigma^q_t)^2)
\]

- `o_t`はencoderで`e_t`へ変換される。
- 必要性: dynamicsの予測だけでなく、実際に得た観測でstate beliefを修正するため。
- コードでは安定な小規模planningのため`z_t=mu^q_t`を用いる。分布そのものはKLへ使う。

### Prior ensemble

\[
p_k(z_t\mid h_t)=\mathcal N(\mu^p_{k,t},(\sigma^p_{k,t})^2),\quad k=1,2,3
\]

- 現在画像を使わない。
- 必要性: future imagination中には未来観測がないため。
- head間のreturn差をepistemic proxyとして使う。ただしbackbone共有なので完全な独立ensembleではない。

### KL divergence

対角Gaussianごとに、

\[
D_{KL}(q\|p)=\sum_i\left[
\log\frac{\sigma^p_i}{\sigma^q_i}
+\frac{(\sigma^q_i)^2+(\mu^q_i-\mu^p_i)^2}{2(\sigma^p_i)^2}
-\frac12\right]
\]

- 必要性: 観測で得たposterior stateを、観測なしpriorが予測できるようにする。
- free nats 0.5: 小さすぎるKLをさらに押し下げる圧力を止め、表現容量を確保する。
- 三つのpriorすべてとposteriorを整合する。

### Reconstruction

\[
L_{recon}=\frac{1}{N}\|\hat o_t-o_t\|_2^2,
\qquad \hat o_t=d([h_t,z_t])
\]

- 必要性: stateが画像内容を捨ててtask labelだけを覚えるのを抑え、視覚的監査を可能にする。
- plannerはdecoderを使わない。

### Task heads

\[
(\hat r_t,\hat V_t,\hat c_t,\hat g_t)=f_{task}([h_t,z_t])
\]

- reward: 即時の距離変化とGoal bonus。
- value: `-ManhattanDistance/4`というterminal potential。policy returnではない。
- continuation: terminalで0、それ以外1。
- goal: hidden Goalが右か下かを読む補助分類。
- posterior featureだけでなく、計画に使う全prior featureにも同じtargetを与える。

### Latent overshooting

\[
L_{over}=\mathbb E_{t,d\le3}
\left\|\mu^p_{t+d\mid t}-\operatorname{stopgrad}(z^q_{t+d})\right\|_2^2
\]

- 必要性: 一歩予測を繰り返したときのlatent driftを直接抑える。
- 今回はensemble member 0のmean rolloutを代表として使う。

### Planning objective

\[
R_k=\sum_{j=0}^{H-1}\gamma^j\hat r(s^k_{t+j})
+\gamma^H\hat V(s^k_{t+H})
\]

\[
J=\operatorname{Mean}_k[R_k]-\beta\operatorname{Std}_k[R_k]
\]

- `H=6`, `gamma=0.97`, `beta=0.5`。
- 512本の離散action sequenceをrandom shootingで生成する。
- stdが大きい候補を保守的に減点するが、この課題ではbeta=0でも全成功した。

## Code Mapping

- `dataset.py::IntegratedNavigationDataset`: alias historyと全教師target。
- `model.py::State`: `h_t`と`z_t`。
- `model.py::IntegratedWorldModel.observe`: sequence posterior filtering。
- `model.py::posterior_step`: 実環境の新観測によるbelief更新。
- `model.py::prior_step`: 観測なしimagination。
- `losses.py::diagonal_gaussian_kl`: 上のKL式。
- `losses.py::integrated_loss`: weighted objective、prior task supervision、overshooting。
- `planner.py::RiskAwarePlanner.plan`: return mean/stdと候補選択。
- `planner.py::DiscreteActionGuard.filter`: action境界。
- `evaluate.py::run_episode`: filtering、計画、1 action実行、再filterのloop。

## Important Components

### Why deterministic and stochastic states are both needed

`h_t`は長いhistoryを連続的に運ぶ。`z_t`はその時刻の観測内容と予測不確実性を表す。`h_t`だけでは観測でstateを確率的に補正するinterfaceが弱く、`z_t`だけでは長いhistoryを毎時刻保持しにくい。

### Why train task heads on prior features

planning中のfutureにはposteriorを作る画像がない。posteriorだけでreward/valueを学習すると、同じheadでも入力分布が違い、初回実装では下Goalを認識できなかった。利用時と同じprior featureへlossを与える必要がある。

### Why re-plan after every action

長いopen-loop action列をそのまま実行するとmodel errorが蓄積する。MPCは先頭actionだけ実行し、次の観測でposterior補正してから再計画する。

### What does uncertainty mean here?

同じ候補actionを三つのprior headでrolloutしたreturnのばらつきである。観測noiseを表すaleatoric stdとは違う。ただしheadがbackboneを共有するため、真のepistemic uncertaintyを過小評価する可能性がある。

## What happens if we remove it?

- Memory (`h_t`): Phase 90 ablationでhidden-Goal accuracyが1.0から0.5へ落ちる。
- Posterior: 新観測でbeliefを修正できず、初期model errorを引きずる。
- Prior: 観測なし未来を生成できず、planningできない。
- KL: posteriorとpriorが別々の座標系へ発散し、学習時stateを想像時に再現できない。
- Prior task supervision: 初回実装のように片方のGoalへだけ進む可能性がある。
- Reward head: trajectoryの途中を評価できない。
- Value head: horizonの外側を無視し、短期報酬だけを追う。
- Overshooting: multi-step driftへの直接拘束がなくなる。
- Ensemble: disagreementを測れず、risk scoreはmean-onlyになる。
- Decoder: planningは可能だが、pixel情報保持を監査しにくい。
- MPC replanning: open-loop errorが蓄積する。
- Guard: 型外・範囲外actionがenvironment/device境界へ流れる。

## What I Should Be Able to Explain

- 同じ現在観測なのに右と下の異なる行動を選べる理由は何か。
- `h_t`と`z_t`はそれぞれ何を持つか。
- posteriorが使える時刻とpriorしか使えない時刻を説明できるか。
- KLが小さいとは、posteriorとpriorのどの量が近いことか。
- task headをposteriorだけで学習すると、なぜplanningで失敗しうるか。
- overshootingと通常のone-step KLは何が違うか。
- ensembleのstdはaleatoric uncertaintyとどう違うか。
- risk-aware scoreの`beta`を大きくすると何が起こるか。
- value targetがpolicy returnではなくdistance potentialである影響を説明できるか。
- decoderを外してもplannerが動く理由は何か。
- MPCが候補列の先頭actionだけ実行する理由は何か。
- 40/40成功から「risk-awareが優れている」と結論できない理由は何か。

## Questions

- 独立bootstrap RSSMとshared-prior headsでOOD calibrationはどれだけ違うか。
- latent sampleを使うparticle rolloutにすると多峰性を扱えるか。
- goal補助labelなしでもbeliefにGoal情報が残るか。
- continuation probabilityをreturnへ掛けるべきか。
- learned TD valueへ置き換えたときもclosed-loop successを維持できるか。
- Transformer cacheを同じplanner state interfaceへ接続した場合の長期優位はあるか。
- 実機前に、uncertainty thresholdで停止するguardをどう校正するか。
