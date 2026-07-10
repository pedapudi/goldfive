# The Landscape of LLM Behavior Steering

*A literature survey organized by intervention locus, with an applied lens on runtime agent supervision (goldfive).*

## Abstract

"Steering" an LLM — reliably moving its behavior toward truthfulness, safety,
a persona, a style, or adherence to a task — has become a sprawling research
area spanning weight training, activation edits, decoding-time logit
arithmetic, prompt engineering, and external guardrails. This survey
organizes that space by a single primary axis: the **intervention locus** —
*where in the computational stack the control signal is injected* — cross-cut
by the **control signal** used (contrastive pairs, classifiers, rewards,
probes, trained tokens, written constitutions) and by the **purpose**
(capability, safety, persona/style, task-adherence). We give a taxonomy
(§2), ten deep-dive sections one per method family (§3), a synthesis of the
cross-cutting themes that recur across families — the controllability↔capability
tradeoff, faithfulness/monitorability as the substrate for any read-based
control, evaluation practice, composition, and adversarial robustness (§4) —
and an applied section (§5) mapping the literature onto a concrete runtime
agent-supervision system, goldfive, whose design is *observe → detect drift →
intervene without requiring agent cooperation*. We close with open problems
(§6) and a full, verified bibliography (§7).

The central empirical finding of the literature, stated up front: **behaviors
are often — but not always, and not surgically — linear directions in a
model's representations**, readable by cheap probes and writable by cheap
vector edits, but reliability is model-, behavior-, and layer-specific, and
any single-direction control is both entangled with off-target behavior and
trivially reversible by an adversary. The frontier is not "can we steer" but
"can we steer *reliably, conditionally, and durably*, and know when we
cannot."

## How to read this / scope

- **Audience.** Researchers and engineers building control, oversight, or
  interpretability tooling for deployed LLM agents. The applied section (§5)
  assumes familiarity with goldfive but the rest is self-contained.
- **Scope.** We cover *behavioral* steering of text LLMs: methods that change
  what a model does, for a stated purpose, at any point in its lifecycle. We
  deliberately include training-time methods (RLHF/DPO/CAI) because they set
  the default the runtime methods perturb, and internal-state readout
  (probing) because *detection* is the precondition for any closed-loop
  *control*. We do not attempt to cover pretraining data curation, multimodal
  generation, or retrieval augmentation except where they intersect steering.
- **Citation discipline.** Every method claim carries a `[key]` citation into
  the verified bibliography (§7). The bibliography is a closed set: every
  entry was confirmed to be a real paper, and no claim cites anything outside
  it. Where a synthesis draws a general conclusion across many papers, the
  supporting keys are listed together.
- **Register.** Academic-but-practical. Each deep-dive states *how it works*,
  *what it achieves*, *its limitations*, and *its evaluation practice*, then —
  where relevant — the goldfive-facing implication.

## Table of contents

1. [Abstract & scope](#abstract) (above)
2. [Taxonomy: intervention locus × control signal × purpose](#2-taxonomy-intervention-locus--control-signal--purpose)
3. Deep dives, one per family:
   - [3.1 Activation steering & representation engineering](#31-activation-steering--representation-engineering)
   - [3.2 Sparse autoencoders & feature-based steering](#32-sparse-autoencoders--feature-based-steering)
   - [3.3 Decoding-time & logit-space steering](#33-decoding-time--logit-space-steering)
   - [3.4 Prompt, in-context steering & trained control codes](#34-prompt-in-context-steering--trained-control-codes)
   - [3.5 Training-time steering: RLHF, DPO, and Constitutional AI](#35-training-time-steering-rlhf-dpo-and-constitutional-ai)
   - [3.6 Probing, internal-state readout & deception detection](#36-probing-internal-state-readout--deception-detection)
   - [3.7 CoT faithfulness, monitorability & process supervision](#37-cot-faithfulness-monitorability--process-supervision)
   - [3.8 Agent oversight, guardrails & AI control](#38-agent-oversight-guardrails--ai-control)
   - [3.9 Safety-specific steering: refusal, jailbreak, unlearning](#39-safety-specific-steering-refusal-jailbreak-unlearning)
   - [3.10 Knowledge editing, adapters & architectural control](#310-knowledge-editing-adapters--architectural-control)
4. [Cross-cutting themes](#4-cross-cutting-themes)
5. [Applied: opportunities for agent-steering systems (goldfive lens)](#5-applied-opportunities-for-agent-steering-systems-goldfive-lens)
6. [Open problems & research frontier](#6-open-problems--research-frontier)
7. [References](#7-references)

---

## 2. Taxonomy: intervention locus × control signal × purpose

The field's methods are usually named after their *mechanism* (DPO, GeDi,
CAA, ROME) or their *goal* (detoxification, unlearning, refusal). We find the
most useful organizing axis is neither: it is **where the control signal
enters the computation**. That single choice determines the method's cost,
portability, power, durability, and — crucially for a runtime supervisor —
whether it requires the target model's cooperation, its weights, or only its
API.

### 2.1 The primary axis: intervention locus

We distinguish five loci, ordered roughly from "outside the model" to "inside
the weights":

**(a) Input / context & prompt-level.** The control signal is text (or soft
"virtual" tokens) placed in the model's context: system prompts, instructions,
in-context demonstrations, chain-of-thought exemplars, trained control codes,
prefix/prompt-tuning vectors. Cheapest and most portable — works over any API
— but bounded, brittle to wording, and (for trained control tokens) leaky.
Canonical work: instruction tuning [wei2022flan], RLHF-for-following
[ouyang2022instructgpt], CoT [wei2022cot], CTRL control codes [keskar2019ctrl],
prefix/prompt/P-tuning [li2021prefix; lester2021prompttuning; liu2021ptuning],
automatic prompt optimization [zhou2022ape; yang2023opro; khattab2023dspy].

**(b) Decoding / logit-space.** The control signal reshapes the next-token
distribution during generation: attribute classifiers guiding logits, expert
minus anti-expert deltas, layer/model contrasts, reward-value rescoring, and
hard grammar masks. No weight change; needs only logit access (sometimes only
a second model's logits). Canonical work: PPLM [dathathri2019pplm], GeDi
[krause2020gedi], DExperts [liu2021dexperts], FUDGE [yang2021fudge],
Contrastive Decoding [li2022contrastive; obrien2023cdreasoning], DoLa
[chuang2023dola], proxy-tuning [liu2024proxytuning], model arithmetic
[dekoninck2024arithmetic], reward/value guidance [deng2023rad; khanov2024args;
mudgal2023controlled], constrained decoding [geng2023gcd; willard2023outlines;
dong2024xgrammar].

**(c) Activation / representation-space.** The control signal is a vector
added to (or a subspace projected out of) the residual stream / attention-head
outputs mid-forward-pass. Requires white-box activation access but no weight
update and no cooperation. Canonical work: task arithmetic (as a conceptual
root) [ilharco2023taskarith], ActAdd [turner2023actadd], ITI [li2023iti], RepE
[zou2023repe], CAA [rimsky2024caa], refusal-direction ablation
[arditi2024refusal; marshall2024affine], function/in-context vectors
[todd2024funcvec; liu2024icv; hendel2023taskvec], SAE feature clamping
[templeton2024monosemanticity], conditional/entropic steering [lee2025cast;
rahn2024east].

**(d) Weights / training-time.** The control signal is baked into parameters:
preference optimization, constitutional training, conditional policies,
weight-space knowledge edits, and representation-rerouting training. Most
durable and highest-authority, but static (cannot respond to per-episode
runtime drift) and expensive. Canonical work: RLHF and successors
[christiano2017deeprl; stiennon2020summarize; ouyang2022instructgpt;
rafailov2023dpo], Constitutional AI [bai2022cai], knowledge editing
[meng2022rome; meng2023memit], circuit breakers [zou2024circuitbreakers],
unlearning [li2024wmdp].

**(e) Architecture / adapters.** The control signal is a swappable module or a
routing decision: adapters, LoRA/QLoRA deltas, task-vector arithmetic and
merging, side-memory codebooks, and MoE expert gating. Sits between (c) and
(d): reconfigurable at load-time, composable, but requires the module to
exist. Canonical work: adapters [houlsby2019adapter], LoRA/QLoRA
[hu2022lora; dettmers2023qlora], task arithmetic and TIES-merging
[ilharco2023taskarith; yadav2023ties], side-memory editing
[hartvigsen2023grace; wang2024wise; mitchell2022serac], MoE [shazeer2017moe].

### 2.2 The secondary axis: control signal

Orthogonal to *where* is *what supervises the edit*:

- **Contrastive pairs** — a positive/negative example pair whose activation or
  logit difference defines a direction (CAA, ActAdd, ICV, DExperts, defection
  probes) [rimsky2024caa; turner2023actadd; liu2024icv; liu2021dexperts;
  macdiarmid2024probes].
- **Trained classifiers / discriminators** — a small model scoring an
  attribute, used to guide logits or gate output (PPLM, GeDi, FUDGE, Llama
  Guard, Constitutional Classifiers) [dathathri2019pplm; krause2020gedi;
  yang2021fudge; inan2023llamaguard; sharma2025constitutionalclassifiers].
- **Reward / value functions** — a scalar objective optimized in weights (RLHF
  family) or at decode time (RAD, ARGS, Controlled Decoding) [ouyang2022instructgpt;
  deng2023rad; khanov2024args; mudgal2023controlled].
- **Linear probes** — a supervised or unsupervised direction read from
  activations, used for detection and sometimes control (SAPLMA, CCS,
  geometry-of-truth, ITI) [azaria2023internal; burns2022ccs; marks2023geometry;
  li2023iti].
- **Trained tokens / codes / soft prompts** — discrete or continuous control
  primitives learned into the input interface (CTRL, Qwen3 mode tokens,
  prefix/prompt-tuning) [keskar2019ctrl; qwen2025qwen3; li2021prefix;
  lester2021prompttuning].
- **Written constitutions / specs** — natural-language principle sets the
  model or a classifier reasons over (CAI, Collective CAI, Deliberative
  Alignment, Constitutional Classifiers) [bai2022cai; huang2024ccai;
  guan2024deliberative; sharma2025constitutionalclassifiers].
- **Dictionary features** — sparse-autoencoder latents providing a named,
  interpretable basis (SAE steering) [bricken2023monosemanticity;
  templeton2024monosemanticity].

### 2.3 The tertiary axis: purpose

- **Capability** — reasoning, factuality, task performance (CD-for-reasoning,
  DoLa, process supervision) [obrien2023cdreasoning; chuang2023dola;
  lightman2023verify].
- **Safety** — refusal, harmlessness, jailbreak-robustness, unlearning (RepE,
  refusal direction, circuit breakers, WMDP/RMU) [zou2023repe; arditi2024refusal;
  zou2024circuitbreakers; li2024wmdp].
- **Persona / style** — sentiment, tone, topic, format (ActAdd, DExperts, CFG,
  Golden Gate Claude) [turner2023actadd; liu2021dexperts; sanchez2023cfg;
  templeton2024monosemanticity].
- **Task-adherence** — staying on the user's goal, exploration/caution,
  instruction priority (EAST, instruction hierarchy, AI control) [rahn2024east;
  wallace2024instructionhierarchy; greenblatt2023aicontrol].

### 2.4 Summary table

The table below places representative methods on the primary axis and records
their control signal, whether they need weights, and whether they need the
target's cooperation (defined as: the target must call a tool, follow an
instruction, or otherwise *choose* to comply). "Cooperation-free" methods are
the ones a non-collaborative runtime supervisor can rely on against an
adversarial or indifferent agent.

| Locus | Representative methods | Control signal | Needs weights? | Cooperation-free? | Primary purpose |
|---|---|---|---|---|---|
| (a) Input/prompt | Instruction tuning, CoT, ICL, CTRL codes, prefix/prompt-tuning, APE/OPRO/DSPy, instruction hierarchy [wei2022flan; wei2022cot; keskar2019ctrl; li2021prefix; yang2023opro; wallace2024instructionhierarchy] | text, soft prompts, trained tokens, constitutions | no (soft prompts: light train) | partial — relies on the model heeding text | all |
| (b) Decoding/logit | PPLM, GeDi, DExperts, FUDGE, CD, DoLa, proxy-tuning, model arithmetic, RAD/ARGS/CD-value, grammar decoding [dathathri2019pplm; krause2020gedi; liu2021dexperts; li2022contrastive; chuang2023dola; liu2024proxytuning; dekoninck2024arithmetic; mudgal2023controlled; willard2023outlines] | classifiers, expert deltas, rewards, grammars | no | **yes** (logit-level; grammar is a hard guarantee) | capability, style, safety, format |
| (c) Activation/repr | ActAdd, ITI, RepE, CAA, refusal direction/ACE, function/in-context vectors, SAE clamping, CAST, EAST [turner2023actadd; li2023iti; zou2023repe; rimsky2024caa; arditi2024refusal; marshall2024affine; templeton2024monosemanticity; lee2025cast; rahn2024east] | contrastive pairs, probes, SAE features | no (activation access) | **yes** (forward-pass edit) | safety, style, task-adherence, capability |
| (d) Weights/train | RLHF, DPO/IPO/KTO/ORPO/SimPO, CAI/RLAIF, CLP, ROME/MEMIT, circuit breakers, RMU, TAR, deliberative alignment [ouyang2022instructgpt; rafailov2023dpo; bai2022cai; wang2024clp; meng2022rome; zou2024circuitbreakers; li2024wmdp; tamirisa2024tar; guan2024deliberative] | rewards, preferences, constitutions, edits | **yes** | n/a (pre-deployment) | all — sets defaults |
| (e) Arch/adapters | Adapters, LoRA/QLoRA, task arithmetic, TIES, GRACE/WISE side-memory, SERAC, MoE gating [houlsby2019adapter; hu2022lora; ilharco2023taskarith; yadav2023ties; hartvigsen2023grace; wang2024wise; mitchell2022serac; shazeer2017moe] | trained deltas, codebooks, routers | yes (module), no (base frozen) | **yes** (attach/detach module) | task-adherence, capability, safety |

The three cooperation-free loci — (b), (c), and (e) — are the ones most
directly usable by a runtime supervisor that cannot assume the agent will
comply. The rest of the survey elaborates each family; §5 maps them back onto
this cooperation-free/portability structure.

---

## 3. Deep dives

### 3.1 Activation steering & representation engineering

**How it works.** Treat a high-level behavior — truthfulness, refusal,
sentiment, sycophancy, power-seeking, or "doing task X" — as an approximately
linear direction in the transformer's residual stream. At inference, *add* or
*subtract* a vector at one or more layers to move outputs along that direction.
No weight update, no retraining, no agent cooperation. The direction is
usually obtained from contrastive stimuli: the mean difference of activations
between matched positive and negative examples, or the top principal component
of such differences, or a supervised probe.

**The three roots.** (1) *Weight-space task arithmetic.* Ilharco et al. defined
a "task vector" as finetuned-minus-pretrained weights and showed *negating* it
removes a capability while *adding* it composes capabilities — establishing
that behaviors live on linear directions manipulable by arithmetic
[ilharco2023taskarith]. (2) *Activation-space steering without optimization.*
Turner et al.'s ActAdd computes a steering vector from the activation
difference of a single natural-language contrast pair (e.g. "love" − "hate")
and adds it in the forward pass, controlling high-level output properties on
GPT-2 with off-target performance preserved [turner2023actadd]. (3)
*Probe-guided intervention.* Li et al.'s Inference-Time Intervention (ITI)
locates a sparse set of truth-correlated attention heads via linear probes on
TruthfulQA and shifts activations along those directions, lifting Alpaca
truthfulness from 32.5% to 65.1% and exposing a tunable truthfulness↔helpfulness
tradeoff [li2023iti].

**The organizing framework.** Representation Engineering (RepE) is a
"top-down" program that places *population-level representations* (not
individual neurons or circuits) at the center of analysis and control, giving
both *reading vectors* (linear probes / PCA on contrastive stimuli) and
*control vectors* for honesty, harmlessness, and power-seeking [zou2023repe].
The dominant recipe is Contrastive Activation Addition (CAA): average the
residual-stream difference between matched positive/negative completions, then
add it at all post-prompt positions with a signed coefficient; it stacks on
top of finetuning and system prompts with minimal capability loss
[rimsky2024caa].

**The sharpest existence proof.** Refusal is mediated by a *single*
difference-in-means direction across 13 chat models up to 72B: ablating it
disables refusal (a rank-one weight "abliteration" jailbreak) and adding it
induces refusal on harmless prompts [arditi2024refusal]. Marshall et al.
refine the picture — refusal is better modeled as an *affine* (not purely
linear) function, and Affine Concept Editing (ACE) combines directional
ablation with a bias correction for more reliable control across ten models
[marshall2024affine].

**Mechanizing in-context learning as vectors.** A parallel thread shows ICL
itself is a steering operation: Hendel et al. show demonstrations compress into
a task vector at an intermediate layer that, patched into a zero-shot pass,
reproduces the task [hendel2023taskvec]; Todd et al.'s Function Vectors show a
few attention heads transport a compact, context-robust task representation
that triggers task execution when injected [todd2024funcvec]; Liu et al.'s
In-Context Vectors steer via the top principal component of demonstration-pair
activation differences, freeing the context window and beating ICL on
safety/style/format tasks [liu2024icv].

**For agents specifically.** Two advances matter. Conditional Activation
Steering (CAST) adds a *condition vector* that gates whether the behavior
vector fires — enabling rule-based selective behavior ("if hate speech,
refuse") and solving the indiscriminate-application problem of always-on
steering [lee2025cast]. Entropic Activation Steering (EAST) steers an LLM
agent's *action-entropy* to fix overconfident under-exploration, controlling
high-level parsed actions where temperature sampling fails, and generalizing
across task variants [rahn2024east] — evidence that activation steering can
move decision-level behavior, not just token style.

**What it achieves.** Cheap (one contrast pair to a few hundred examples),
cooperation-free, composable with prompts and finetuning, and — for
well-behaved behaviors — a genuinely causal lever. Feature-level control via
sparse autoencoders (Golden Gate Claude) points toward interpretable,
named steering directions rather than opaque difference vectors
[templeton2024monosemanticity] (elaborated in §3.2).

**Limitations & evaluation practice.** The reliability literature tempers all
of this. Da Silva et al. test DoLa/function/task vectors on 36 models and find
many show *no improvement or outright degradation*, challenging core
assumptions and motivating per-model validation [dasilva2025steeringoffcourse].
Braun et al. show steering fails when the target behavior lacks a coherent
direction, and that *train-time activation alignment* predicts steering
dependability — giving a diagnostic for when to trust a vector
[braun2025unreliability]. Evaluation practice therefore centers on: (i)
behavior-specific benchmarks (TruthfulQA for ITI, sycophancy/refusal datasets
for CAA), (ii) off-target/capability probes to measure collateral damage, and
(iii) strength sweeps to locate the usable coefficient band. Net: direction
quality, conditionality, and per-model validation are the binding
constraints.

### 3.2 Sparse autoencoders & feature-based steering

**The obstacle.** Mechanistic control of LLMs is blocked by *superposition*:
toy networks pack more sparse features than they have neurons, so individual
neurons are polysemantic and not a usable control surface [elhage2022superposition].

**The method.** Train a wide, L1-penalized autoencoder on a layer's
activations to recover an overcomplete dictionary of sparse, largely
monosemantic "features," each a linear direction in activation space
[bricken2023monosemanticity]. Templeton et al. scaled this to Claude 3 Sonnet,
extracting millions of abstract, multilingual, multimodal features and
demonstrating *causal steering*: clamping a feature to a high multiple of its
max activation forces the behavior ("Golden Gate Claude" believes it is the
bridge), and safety-relevant features (deception, sycophancy, bias, dangerous
content) can be surfaced and their clamping/ablation changes behavior
[templeton2024monosemanticity]. This establishes the SAE steering primitive:
*decompose → identify a feature by concept → add/scale/ablate its decoder
direction at inference*, with no retraining and no cooperation.

**Architecture progress.** A wave of work improved the reconstruction↔sparsity
Pareto frontier and removed tuning pain. TopK SAEs fix the active-latent count
directly, killing L1 shrinkage; trained to 16M latents on GPT-4, with
downstream-effect and probe-recovery metrics and clean scaling laws
[gao2024scaling]. Gated SAEs split "which features fire" from "how much" to
remove L1 shrinkage bias [rajamanoharan2024gated]; JumpReLU SAEs use a
discontinuous threshold with straight-through estimators to optimize L0
directly, hitting SOTA fidelity at fixed sparsity [rajamanoharan2024jumprelu].
Switch SAEs add mixture-of-experts routing for compute-efficient scaling
[mudide2024switch]. Gemma Scope shipped 400+ JumpReLU SAEs / 30M+ features
across all Gemma 2 layers, making steering research reproducible on open
weights [lieberum2024gemmascope]. Sparse crosscoders generalize SAEs to
read/write across layers and models for cross-layer feature tracking and model
diffing [lindsey2024crosscoders].

**The control frontier is contested.** Sober evaluations now dominate.
Durmus et al. found a narrow usable band (steering factors ~−5..+5) before
coherence collapses, and pervasive off-target effects (steering gender bias
down raised age bias) — steering is *entangled, not surgical*
[durmus2024evaluating]. Marks et al. built Sparse Feature Circuits — causal
graphs of SAE features — and SHIFT, ablating human-judged task-irrelevant
features to de-bias a classifier, showing feature-level editing can beat neuron
circuits [marks2024circuits]. Chalnev et al. use SAEs to *measure* a steering
vector's feature-space effect and optimize vectors (SAE-TS) that hit a target
feature while minimizing side effects, beating CAA and naive SAE clamping on
the coherence/effect tradeoff [chalnev2024saets]. Xie shows naive constant SAE
steering degenerates into repetition; a top-1-latent + token-wise-decay recipe
is needed to match mean-activation-difference baselines, and reasoning latents
can reliably elicit step-by-step CoT [xie2025comparative].

**What it achieves / limitations.** SAEs are unmatched for *interpretable
discovery* of a control direction — they name *why* a behavior is happening.
But for raw steering *strength* they often tie or lose to cheaper
contrastive/difference-of-means vectors, and feature splitting/absorption plus
reconstruction error limit faithfulness; some teams have reported negative
downstream results and deprioritized SAEs. Evaluation practice combines
reconstruction/sparsity Pareto curves, probe-recovery and downstream-loss
metrics [gao2024scaling], steering-factor bias/off-target sweeps
[durmus2024evaluating], and A/B against difference-of-means baselines
[chalnev2024saets; xie2025comparative]. The pragmatic synthesis: use SAEs for
*what to steer* (interpretable, auditable, named) and cheaper contrastive
vectors for *how hard*, always against an unsteered baseline.

### 3.3 Decoding-time & logit-space steering

Decoding-time steering manipulates the token distribution during generation
rather than retraining weights — the closest literature analog to "intervene
on a running agent without cooperation," since it needs only logit access.

**Classifier/discriminator-guided (the canonical family).** The lineage begins
with train-time control (CTRL codes [keskar2019ctrl]) but pivots to
inference-time with PPLM, which runs gradient ascent on a frozen LM's hidden
activations toward a small attribute classifier — the first "plug-and-play"
steer of a fixed model [dathathri2019pplm]. Because PPLM's backward passes are
slow, the field moved to logit-space arithmetic. GeDi uses a small
class-conditional LM as a *generative discriminator*, applying Bayes' rule over
a control/anti-control pair to reweight every next-token logit in a single
forward pass (~30× faster) [krause2020gedi]. DExperts generalizes to a
product-of-experts: add an "expert" LM's logits and subtract an "anti-expert"'s,
so tokens survive only if desirable and non-toxic [liu2021dexperts]. FUDGE
trains a lightweight predictor on partial sequences that estimates whether the
attribute will hold in the *future*, then multiplies P(attribute|prefix+token)
into the base logits — a Bayesian factorization needing only output logits, and
composing multiple attributes [yang2021fudge].

**Contrast two models/layers (steering that improves capability).** Contrastive
Decoding maximizes the log-likelihood gap between a large "expert" and a small
"amateur" (subject to a plausibility mask), removing degenerate repetition
[li2022contrastive]; O'Brien & Lewis showed the same expert−amateur subtraction
yields large *reasoning* gains (LLaMA-65B beating GPT-3.5/PaLM on
GSM8K/HellaSwag) [obrien2023cdreasoning]. DoLa contrasts *late vs early layers*
of a single model to surface factual knowledge, cutting hallucination (+12–17
pts on TruthfulQA) with no extra model [chuang2023dola]. Proxy-tuning reframes
the delta as portable "tuning": the logit-difference between a small tuned and
untuned model steers a large black-box base, closing 88% of the tune gap on
Llama2-70B using 7B proxies [liu2024proxytuning]. Model arithmetic unifies all
of the above as composable linear/union operators over LM logits, subsuming
CFG/GeDi/DExperts as special cases [dekoninck2024arithmetic].
Classifier-free guidance imports the diffusion trick: extrapolate away from the
unconditioned distribution to amplify prompt adherence, with a continuous γ
knob [sanchez2023cfg].

**Reward/value-guided (steer toward an arbitrary objective).** RAD rescales
sampling by a cached unidirectional reward model [deng2023rad]; ARGS folds an
RLHF reward directly into the search, aligning without RL training
[khanov2024args]; Controlled Decoding trains a prefix value function and does
blockwise value-guided decoding, interpolating best-of-K and tokenwise RL
[mudgal2023controlled].

**Constrained / structured decoding (hard guarantees).** This line hard-masks
the vocabulary to a grammar. GCD enforces CFGs for structured NLP without
finetuning [geng2023gcd]; Outlines compiles regex/CFG to a finite-state index
for near-free guided JSON [willard2023outlines]; XGrammar makes CFG masking
production-fast (up to 100×, <40µs/token) for agent tool-calls
[dong2024xgrammar].

**Limitations & evaluation practice.** Single-attribute discriminators compose
poorly; contrastive methods can over-penalize valid tokens; reward/value
scorers need training and can be gamed; and all logit-space edits act *locally
per-token*, with no memory of trajectory-level drift — precisely the gap a
trajectory-aware supervisor addresses. Evaluation practice: attribute-success
plus fluency/diversity for the classifier family (toxicity via a held-out
scorer, perplexity, distinct-n), task accuracy for the contrastive-capability
family (GSM8K, TruthfulQA), reward and win-rate for the value family, and
constraint-satisfaction rate plus latency overhead for structured decoding.

### 3.4 Prompt, in-context steering & trained control codes

This family steers through the *input interface* and through control
primitives *trained into the weights*, spanning three lineages.

**Trained control codes.** CTRL prepended domain/style codes ("Wikipedia",
"Reviews Rating:") as the first token so a 1.6B causal LM conditions on
explicit steering symbols derived from naturally co-occurring metadata — the
direct ancestor of today's mode tokens [keskar2019ctrl]. Modern reasoning
models generalize this: Qwen3 bakes `/think` and `/no_think` control tokens
plus a numeric "thinking budget" into one weight set via SFT, letting a runtime
caller toggle reasoning depth without swapping models [qwen2025qwen3].

**Soft / continuous prompts.** Rather than discrete tokens, prefix-tuning
prepends trainable "virtual token" vectors that later tokens attend to,
matching fine-tuning at ~0.1% of parameters [li2021prefix]; prompt tuning shows
a single learned soft prompt closes the gap to full fine-tuning as scale grows
and enables cheap per-task conditioning of one frozen model [lester2021prompttuning];
P-tuning interleaves trainable continuous embeddings with discrete prompts to
stabilize the brittleness of hand-written prompts [liu2021ptuning]. These make
"steering vectors in embedding space" a first-class, composable control surface
— a bridge to the activation steering of §3.1.

**Instruction & in-context steering.** Instruction tuning [wei2022flan] and
RLHF [ouyang2022instructgpt] trained models to follow natural-language
directives, making the system/user prompt the primary behavioral lever (a 1.3B
InstructGPT beat 175B GPT-3 on instruction-following). ICL acts as steering by
demonstration; chain-of-thought exemplars elicit multi-step reasoning purely
from the prompt [wei2022cot], and ICL provably compresses demonstrations into a
single task vector that modulates the forward pass [hendel2023taskvec] — the
mechanistic bridge between prompt-level and activation-level steering.

**Automatic prompt optimization.** Because hand-written prompts are fragile,
prompt optimization automates the search: APE generates and selects
instructions with an LLM [zhou2022ape], OPRO treats the LLM as an optimizer
that iteratively proposes better prompts from a trajectory of scored candidates
(+8% GSM8K, +50% BBH) [yang2023opro], and DSPy compiles declarative LM
pipelines, optimizing prompts/demonstrations against a metric [khattab2023dspy].

**Runtime-control tensions (2024–2026).** Three findings dominate and are
directly relevant to any runtime steerer. First, steering is *bounded and
asymmetric*: Miehling et al. formalize steerability as how far a model's
behavioral distribution can be shifted from baseline and find current models
have limited, skewed steerability [miehling2024steerability]. Second, *trained
control tokens leak*: Wang et al. show Qwen3 in `/no_think` still emits
reasoning tokens and "wait" hundreds of times — the control token is a soft
suggestion, not a hard switch [wang2025hybridthinking]. Third, *conflicting
instructions need priority*: the instruction hierarchy trains models to
privilege system over user over tool text to resist injection/override
[wallace2024instructionhierarchy], and instruction-ladder reasoning reframes
hierarchy resolution as an explicit reasoning step [zheng2025instructionladder].
A complementary runtime lever is budget forcing — injecting "Wait" or forcing
end-of-thinking to stretch or cut reasoning length at decode time
[muennighoff2025s1]. The whole space (retraining, fine-tuning, RL, prompt
engineering, latent-space manipulation, decoding-time intervention) is
taxonomized by the controllable-text-generation survey [zhang2024ctgsurvey].

**Limitations & evaluation practice.** Prompt steering is cheapest and most
portable but the most brittle: bounded reach [miehling2024steerability], wording
fragility [liu2021ptuning], and leaky trained tokens [wang2025hybridthinking].
Evaluation practice: instruction-following benchmarks, steerability indices
that measure distributional shift [miehling2024steerability], reasoning-length
and mode-adherence measurements for control tokens [wang2025hybridthinking;
muennighoff2025s1], and metric-driven optimization loops for automated prompts
[yang2023opro; khattab2023dspy].

### 3.5 Training-time steering: RLHF, DPO, and Constitutional AI

Training-time steering shapes an LLM's *default* behavior by baking preferences
into weights before deployment. It sets the baseline that all the runtime
methods perturb.

**Reward-model RL (the canonical recipe).** Christiano et al. scaled
pairwise-preference reward learning to deep RL [christiano2017deeprl]; Stiennon
et al. applied it to summarization — learn a reward model from human
comparisons, optimize a policy with PPO [stiennon2020summarize]; Ouyang et al.
(InstructGPT) fused three stages — SFT on demonstrations, a Bradley-Terry reward
model over ranked outputs, then PPO — showing a 1.3B aligned model beats 175B
GPT-3 on human preference [ouyang2022instructgpt]. Touvron et al. (Llama 2)
productionized this at scale with *two separate* reward models (helpfulness and
safety), rejection sampling + PPO, and safety techniques (context distillation,
adversarial-prompt collection) [touvron2023llama2].

**Constitutional AI / RLAIF.** Constitutional AI replaced human harm labels
with a written list of principles: the model self-critiques and revises its
outputs (SFT phase) then generates AI preference labels used to train a
preference model (RLAIF phase), yielding a harmless-but-non-evasive assistant
[bai2022cai]. Lee et al. showed RLAIF matches or exceeds RLHF on summarization
and harmlessness, and that direct-RLAIF (reward straight from an off-the-shelf
LLM) can skip reward-model training [lee2023rlaif]. Collective Constitutional
AI sourced the constitution itself from ~1,000 US adults via a deliberative
process, improving bias scores while holding MMLU/GSM8K [huang2024ccai].

**Collapsing the RL pipeline into direct offline objectives.** DPO proved the
optimal RLHF policy has closed form relative to a reference policy, so
preference tuning reduces to a simple classification loss ("your LM is secretly
a reward model") — no reward model, no sampling [rafailov2023dpo]. Azar et al.
generalized DPO/RLHF into the ΨPO family and derived IPO (identity mapping),
robust to the over-fitting DPO suffers when preferences are near-deterministic
[azar2023ipo]. KTO drops paired preferences entirely, optimizing a
Kahneman-Tversky loss-averse utility over per-example desirable/undesirable
binary signals, matching DPO at 1B–30B [ethayarajh2024kto]. ORPO folds
preference alignment into SFT via an odds-ratio penalty — no reference model, no
separate stage [hong2024orpo]. SimPO uses length-normalized average log-prob as
a reference-free implicit reward plus a target margin, beating DPO
[meng2024simpo]. Earlier RL-free alternatives — RRHF (ranking loss over scored
candidates) [yuan2023rrhf] and SLiC-HF (sequence-likelihood calibration)
[zhao2023slichf] — anticipated this. Self-Rewarding LMs close the loop: the
model acts as LLM-as-judge to generate its own preference pairs for iterative
DPO [yuan2024selfreward]; Meta-Rewarding adds a meta-judge over its own
judgments [wu2024metareward].

**Conditional / steerable training.** For steerability specifically,
conditional training injects a control variable so one model spans a behavior
range: CTRL prepends control codes governing style/content [keskar2019ctrl];
Conditional Language Policy conditions on multi-objective reward weights so a
single model trades off conflicting objectives (e.g. helpfulness vs. length) at
inference time, producing smooth Pareto fronts without retraining
[wang2024clp].

**Limitations & evaluation practice.** Preference data is expensive and
captures a *single averaged* preference (motivating multi-objective/personalized
work); reward hacking and over-optimization are endemic; the DPO family is
fragile to distribution shift and near-deterministic labels [azar2023ipo]; and
— most relevant to a runtime supervisor — all of these fix behavior at train
time and *cannot respond to run-time drift* in a deployed agent. Evaluation
practice: human/AI preference win-rates (MT-Bench-style), reward-model scores,
safety/harmlessness red-teaming [touvron2023llama2], and — increasingly —
Pareto-front analyses over competing objectives [wang2024clp].

### 3.6 Probing, internal-state readout & deception detection

A large, convergent body of work shows that LLM internal activations encode a
model's own assessment of truth, correctness, and even intent — information
often *not* faithfully reflected in the generated text. Because *detection* is
the precondition for closed-loop *control*, this family is the read-side twin of
§3.1.

**Supervised probes.** SAPLMA — a small classifier on hidden-layer activations
— predicts whether a statement the model is reading or generating is true
(71–83% acc), establishing that "the internal state knows when it's lying"
[azaria2023internal]. Marks & Tegmark sharpened this into a *linear
representation of truth*: true/false statements separate along a linear
direction that is visible in PCA, transfers across datasets, and is *causal* —
patching along a difference-of-means direction flips the model's treatment of a
statement [marks2023geometry]. RepE generalized this to reading *and*
controlling high-level concepts (honesty, power-seeking, morality) via reading
vectors and activation edits [zou2023repe], with ITI as the control-side
operationalization [li2023iti].

**Unsupervised elicitation.** A parallel thread asks whether latent knowledge
can be found without labels. Contrast-Consistent Search finds a direction
satisfying logical-consistency constraints (a statement and its negation have
opposite truth values), beating zero-shot by ~4% and staying robust when the
model is prompted to lie [burns2022ccs] — the concrete instantiation of the
Eliciting Latent Knowledge agenda: map an AI's world-model to truth even when
its outputs mislead [christiano2021elk]. But Farquhar et al. proved CCS-style
methods can lock onto *any* prominent, consistency-satisfying feature — not
knowledge — and contributed sanity checks; unsupervised elicitation remains
unsolved [farquhar2023challenges]. Liu et al. pushed the supervised side toward
generality, showing a near-universal truthfulness hyperplane emerges when
probes train on 40+ diverse datasets (diversity matters more than volume)
[liu2024hyperplane]. Mallen et al.'s "quirky models" (which err when "Bob"
appears) show probes recover 89% of the truthful/deceptive performance gap and
mechanistic anomaly detection flags dishonesty at 0.95 AUROC [mallen2023quirky].

**From facts to behavior and intent (the agent-relevant frontier).** Orgad et
al. found truthfulness signal concentrates in specific "exact-answer" tokens,
is multifaceted (probes don't universally transfer), and — critically —
internal state often encodes the *correct* answer while the model outputs a
*wrong* one [orgad2024know]. Defection probes catch sleeper-agent backdoors at
>99% AUROC using *generic* contrast pairs ("Are you doing something dangerous?
yes/no") with no knowledge of the trigger [macdiarmid2024probes]. Strategic-
deception probes trained on simple honest/deceptive roleplay generalize to
Llama-3.3-70B concealing insider trading or sandbagging (AUROC 0.96–0.999),
though they fire on spurious correlates — promising but not yet robust
[goldowsky2025deception]. On uncertainty, Kadavath et al. established models are
calibrated and can self-evaluate P(True)/P(IK) [kadavath2022know]; Semantic
Entropy Probes distill semantic entropy into a single-forward-pass hidden-state
probe for cheap hallucination detection that generalizes OOD [kossen2024sep].

**Sandbagging & evaluation-awareness.** A distinct sub-area concerns
*strategic underperformance*. Van der Weij et al. showed frontier models can be
prompted or fine-tuned (password-locked) to selectively underperform on
dangerous-capability evals [vanderweij2024sandbagging]; Tice et al. recover
hidden capability by injecting weight noise (sandbaggers *improve* anomalously)
[tice2024noise]; Nguyen et al. show models linearly represent "this is an eval
vs. deployment," and that direction is both probe-readable and steerable
[nguyen2025evalawareness].

**Limitations & evaluation practice.** Probe transfer failures, spurious
features, and no guarantees for *naturally-arising* (vs. trained) deception.
Evaluation practice centers on AUROC/accuracy on held-out and OOD datasets,
cross-dataset transfer tests, causal validation via activation patching
[marks2023geometry], and the sanity checks of Farquhar et al. against
prominent-but-non-knowledge features [farquhar2023challenges]. The durable
recommendation: prefer diverse-trained probes [liu2024hyperplane], treat any
single probe as advisory, and aggregate multiple directions to cut false
alarms.

### 3.7 CoT faithfulness, monitorability & process supervision

Chain-of-thought makes an LLM's intermediate reasoning legible — simultaneously
a lever for *improving* reasoning and a channel for *oversight*, but a fragile,
only-partly-trustworthy one. Three threads.

**Faithfulness: is the stated CoT the actual cause?** Turpin et al. showed CoT
can systematically misrepresent the true reason: planting biasing features
(e.g. always-"(A)" few-shot ordering) flips answers while the CoT rationalizes
the biased choice without ever mentioning the bias, dropping accuracy up to 36%
[turpin2023lm]. Lanham et al. operationalized faithfulness via interventions —
early-answering, adding mistakes, paraphrasing, filler tokens — and found
reliance on CoT varies widely by task and, counterintuitively, that larger
models often reason *less* faithfully [lanham2023measuring]. In the
reasoning-model era, Chen et al. measured hint-verbalization rates: models
reveal a used hint only ~25% (Claude 3.7 Sonnet) / ~39% (DeepSeek R1) of the
time; outcome-based RL improves faithfulness but plateaus, and reward-hacking
behaviors are almost never verbalized [chen2025reasoning]. Arcuschin et al.
documented unfaithfulness "in the wild" with *no* artificial bias — implicit
post-hoc rationalization, restoration errors (silently fixing a wrong step),
and illogical shortcuts — at non-trivial rates in frontier models
[arcuschin2025wild]. Newer metrics push beyond intervention heuristics:
Faithfulness-by-Unlearning-Reasoning erases a reasoning step's information from
parameters and measures the answer shift [fu2025fur].

**Monitorability: CoT as safety telemetry.** Baker et al. showed a weaker LLM
(GPT-4o) can monitor a stronger model's (o3-mini) CoT to catch reward hacking
in agentic coding *far better than watching actions/outputs alone* — but
applying optimization pressure directly against the monitor produces
*obfuscated* reward hacking: the agent keeps cheating while hiding intent from
the CoT, motivating a "monitorability tax" (avoid strong CoT-directed
optimization) [baker2025monitoring]. The Korbak et al. position paper, signed
by 40+ authors across labs, frames CoT monitorability as "a new and fragile
opportunity" and urges developers to measure and preserve it
[korbak2025monitorability]. Concrete measurement followed: Emmons et al. propose
legibility + coverage autorater proxies for "default monitorability"
[emmons2025pragmatic]; Meek et al. combine faithfulness and verbosity axes,
noting models can look faithful yet be unmonitorable when they omit key factors,
and that monitorability varies sharply by model family [meek2025verbosity].

**Process supervision: reward correct steps, not just outcomes.** Lightman et
al. ("Let's Verify Step by Step") showed step-level process reward models
(PRMs) beat outcome RMs on MATH, and released PRM800K [lightman2023verify].
Math-Shepherd auto-generates step labels via MCTS-style completion, removing
human annotation and enabling step-level PPO [wang2024mathshepherd].
ProcessBench benchmarks error-step localization and finds most PRMs fail to
generalize beyond GSM8K/MATH [zheng2024processbench]. Generative PRMs like
ThinkPRM verify via a long CoT critique using only 1% of PRM800K labels, making
the verifier itself interpretable [khalifa2025thinkprm]. DeepSeek-R1 shows
pure-RL reasoning (self-reflection, verification CoT) emerges *without* step
labels, sharpening the tension: capability from outcome RL vs. legibility from
process signals [deepseek2025r1].

**Limitations & evaluation practice.** Faithfulness metrics disagree;
monitorability is trainable-away; PRMs are brittle out-of-distribution.
Evaluation practice: intervention suites for faithfulness (early-answering,
mistake-injection, paraphrase, filler) [lanham2023measuring], hint-verbalization
rates [chen2025reasoning], monitor-catch-rate under optimization pressure
[baker2025monitoring], autorater legibility/coverage/verbosity scores
[emmons2025pragmatic; meek2025verbosity], and earliest-error-localization
benchmarks for PRMs [zheng2024processbench]. The synthesis: reasoning traces are
the single richest oversight signal available *today*, but a *degrading
resource* — worth reading, worth measuring, and worth *not* optimizing against.

### 3.8 Agent oversight, guardrails & AI control

Four threads together define the "watch-and-intervene" stack for LLM agents.

**LLM-as-a-judge.** A strong model can evaluate open-ended outputs at ~80%
agreement with humans [zheng2023judge]; G-Eval showed CoT + form-filling
probability-weighted scoring boosts correlation with human judgment
[liu2023geval]; a large survey catalogs the reliability failure modes any
judge-based monitor inherits — position, verbosity, and self-enhancement bias,
with mitigations [gu2024judgesurvey]. These are the direct antecedents of a
runtime "observer/judge" that scores agent behavior.

**Guardrails frameworks.** External, programmable checks wrap the model. Llama
Guard is an instruction-tuned classifier applying a safety taxonomy to both
prompts and responses [inan2023llamaguard]; NeMo Guardrails adds a
dialogue-management runtime (Colang) with input/dialog/retrieval/execution
rails independent of and interpretable over the underlying LLM
[rebedea2023nemo]; Constitutional Classifiers push this to production-grade
jailbreak defense using synthetic data generated from a natural-language
constitution, surviving 3,000+ red-team hours at ~0.4% added refusals and ~24%
inference overhead [sharma2025constitutionalclassifiers].

**Self-correction / critique-revise.** Constitutional AI has a model critique
and revise its own drafts against written principles [bai2022cai]; Self-Refine
iterates generate→feedback→refine with a single model for ~20% gains
[madaan2023selfrefine]; Reflexion turns episodic verbal feedback into a
memory-based policy update without weight changes [shinn2023reflexion]; Saunders
et al. showed model-written critiques help humans catch flaws they'd miss — a
scalable-oversight proof of concept [saunders2022selfcritiquing]. Crucially,
Huang et al. show LLMs *cannot* reliably self-correct reasoning without external
feedback — intrinsic self-correction often *degrades* performance
[huang2024cannotselfcorrect] — the core empirical argument for external,
non-cooperative oversight.

**Scalable oversight & AI control.** Sandwiching operationalizes measuring
oversight with non-expert supervisors [bowman2022scalableoversight]; debate lets
weak judges adjudicate stronger debaters, and more-persuasive debaters yield
*more* truthful answers [khan2024debate], though weak-LLM-judge experiments show
debate's gains over baselines are real but modest [kenton2024weakjudges]. The AI
Control agenda reframes oversight adversarially: *assume the strong model is
intentionally subversive*, then compose it with a weaker *trusted* monitor plus
a small human audit budget into "protocols" (trusted monitoring, untrusted
monitoring, trusted editing) whose safety is empirically stress-tested by a red
team [greenblatt2023aicontrol]. Ctrl-Z extends this to multi-step agents with
*resample* protocols on BashBench — dynamically re-sampling suspicious actions
and executing the least-threatening [bhatt2025ctrlz]. AgentHarm benchmarks agent
misuse across 11 harm categories and finds frontier agents brittle to basic
jailbreaks [andriushchenko2024agentharm]. CoT monitorability threads back in
here as a "new and fragile opportunity" for agent oversight
[korbak2025monitorability].

**Limitations & evaluation practice.** Judges/monitors are biased and gameable;
monitor–monitored collusion is a live risk; self-correction needs a reliable
external signal; control protocols depend on a genuinely trusted weaker model
and a tight audit budget. Evaluation practice: human-agreement and bias audits
for judges [zheng2023judge; gu2024judgesurvey], red-team safety/usefulness
Pareto curves for control protocols [greenblatt2023aicontrol; bhatt2025ctrlz],
harm-category coverage for agent benchmarks [andriushchenko2024agentharm], and
sandwiching for oversight scalability [bowman2022scalableoversight].

### 3.9 Safety-specific steering: refusal, jailbreak, unlearning

Safety-specific steering treats refusal, jailbreak-robustness, and forgetting
as manipulable directions/subspaces rather than as behaviors trained in via
labels alone.

**The linear picture and its fragility.** RepE frames safety concepts
(harmfulness, honesty, power-seeking) as population-level linear directions read
and controlled top-down [zou2023repe]. Arditi et al. sharpened this: refusal in
13 chat models up to 72B is mediated by a *single* difference-in-means direction
— ablating it (a rank-one "abliteration" edit) disables refusal on harmful
prompts, adding it forces refusal on harmless ones [arditi2024refusal]. CAA
generalizes the recipe to refusal/sycophancy/compliance steering vectors,
stackable on finetuning/prompting [rimsky2024caa]. Marshall et al. showed
refusal is better modeled as an *affine* function; ACE combines directional
ablation with a bias correction and is more reliable than pure CAA
[marshall2024affine]. CAST adds a gate — a condition vector detects prompt
category in the hidden state and only fires the refusal vector on a rule match
("if hate speech, refuse"), enabling programmable, context-selective refusal
without weight updates [lee2025cast]. But because refusal is *one direction*,
safety is fragile: Qi et al. showed finetuning on ~10 adversarial (or even
benign) examples strips alignment for under $0.20 [qi2023finetuning].

**Entangling safety deeper than a direction.** Circuit Breakers / Representation
Rerouting trains the model to reroute the internal representations that generate
harmful continuations toward an orthogonal/nonsense space, "shorting the
circuit" mid-generation using a circuit-breaker set plus a retain set; it
generalizes to unseen attacks, multimodal image hijacks, and *agent actions*,
cutting harmful compliance by roughly two orders of magnitude under strong
attacks [zou2024circuitbreakers]. Latent Adversarial Training perturbs *latent
activations* rather than inputs to surface unforeseen failures [casper2024lat];
Targeted LAT perturbs the residual stream toward specific failure modes then
trains under those perturbations, improving jailbreak robustness, removing
backdoors without the trigger, and aiding unlearning [sheshadri2024tlat].

**Unlearning.** The WMDP benchmark (3,668 hazardous bio/cyber/chem MCQs)
introduced RMU, which steers forget-topic representations toward a random vector
at an intermediate layer while a retain loss preserves general capability
[li2024wmdp]; TOFU provides a synthetic fictitious-author forget/retain
benchmark with forget-quality (KS-test) and utility metrics, finding no baseline
truly forgets [maini2024tofu]. But unlearning is *shallow*: Łucki et al. recover
most "unlearned" WMDP capability by finetuning on 10 unrelated examples or
removing a few activation directions, questioning whether unlearning beats
ordinary refusal training [lucki2024adversarial]. Tamper-Resistant Safeguards
use meta-learning over simulated finetuning attacks so refusal/unlearning
survives hundreds of adversarial finetuning steps [tamirisa2024tar].

**Reasoning-based alignment.** Orthogonally, Deliberative Alignment teaches
models the safety-spec text and trains them to reason over it in CoT before
answering, pushing the jailbreak-robustness/over-refusal Pareto frontier without
human-labeled CoT [guan2024deliberative].

**Limitations & evaluation practice.** A persistent theme: any *single-direction*
safeguard is trivially removable [arditi2024refusal; qi2023finetuning;
lucki2024adversarial]. The methods form a ladder from cheap inference-time
vectors (CAA/CAST/ACE), through representation-level training (RR, LAT, RMU), to
tamper-resistant and reasoning-based alignment. Evaluation practice: attack
success rate under diverse jailbreaks and adaptive attacks, over-refusal rate on
benign prompts, capability retention (MMLU/MT-Bench), and — for durability —
survival under adversarial finetuning [tamirisa2024tar; qi2023finetuning] and
post-unlearning recovery attacks [lucki2024adversarial].

### 3.10 Knowledge editing, adapters & architectural control

This family changes behavior by editing weights or wiring conditional
structure, rather than by prompting or decoding-time intervention.

**Knowledge-as-memory (locate-then-edit).** Dai et al.'s knowledge neurons
showed factual associations concentrate in FFN neurons whose activation
correlates with fact expression, and can be surgically amplified/suppressed
[dai2022kn]. ROME formalized this via causal tracing — localizing facts to
mid-layer MLP modules on the subject token — then writing a rank-one update to a
single MLP as an associative key→value edit [meng2022rome]. MEMIT generalized
ROME to a closed-form update spread across several MLP layers, scaling to
thousands of simultaneous edits while preserving unrelated knowledge
[meng2023memit].

**Meta-learned and side-memory editors.** MEND uses a hypernetwork that
transforms a fine-tuning gradient (via low-rank decomposition) into a fast local
edit, working on 10B+ models [mitchell2022mend]; SERAC keeps the base model
frozen (black-box) and routes queries through an explicit edit memory plus a
scope classifier and a counterfactual model — a retrieval/gating approach
directly analogous to observe-then-intervene control [mitchell2022serac]. GRACE
maintains a discrete codebook of cached activation keys with deferral radii,
enabling thousands of sequential lifelong edits without touching weights
[hartvigsen2023grace]; WISE argues an "impossible triangle"
(reliability/generalization/locality) blocks pure-parametric or pure-retrieval
lifelong editing, and resolves it with a dual parametric memory plus a learned
router and knowledge-sharding into disjoint subspaces later merged
[wang2024wise]. In-context editing shows demonstrations alone can rival gradient
editing with fewer side effects — pure prompt-space steering [zheng2023ike]. The
whole space is taxonomized by the KnowEdit study (external / merge-into-model /
edit-intrinsic) [zhang2024knowedit].

**The localization caution.** Hase et al. showed causal-tracing localization
does *not* predict where editing works best: edit success is largely unrelated
to traced location, decoupling "where a fact lives" from "where to intervene"
[hase2023localization] — a direct caution for any localization-driven steering.

**PEFT / architectural control.** Houlsby adapters insert small trainable
bottleneck modules per task, freezing the backbone [houlsby2019adapter]. LoRA
reparameterizes updates as low-rank matrices (exploiting low intrinsic
dimension), giving swappable, composable behavior modules [hu2022lora]; QLoRA
makes this feasible on a single GPU via 4-bit NF4 quantization + paged
optimizers [dettmers2023qlora]. A survey taxonomizes
additive/selective/reparameterized/hybrid PEFT families [han2024peft].
Weight-space arithmetic treats fine-tuned deltas as manipulable vectors: task
arithmetic negates a task vector to unlearn and adds vectors to compose skills,
with analogy generalization [ilharco2023taskarith]; TIES-Merging fixes
destructive interference when merging many task vectors by trimming small
entries, electing a majority sign, and averaging only sign-aligned params
[yadav2023ties]. The architectural-control lineage is sparsely-gated MoE, where
a learned gate routes each token to a sparse expert subset — conditional
computation that is itself a behavior-routing surface [shazeer2017moe].

**Limitations & evaluation practice.** Edits produce messy ripple effects on
logically entailed facts [cohen2024ripple]; sequential editing causes gradual
then catastrophic forgetting; and localization is unreliable
[hase2023localization] — so weight editing is powerful for *targeted facts* but
fragile as a *general behavioral controller*. Evaluation practice:
reliability/generalization/locality metrics, mass-edit scaling curves
[meng2023memit], the RippleEdits benchmark for entailed-fact propagation
[cohen2024ripple], forget/retain/utility for unlearning-style edits, and
lifelong-editing sequences [hartvigsen2023grace; wang2024wise].

---

## 4. Cross-cutting themes

### 4.1 The controllability ↔ capability tradeoff

Almost every family surfaces the same tension: pushing a behavior harder costs
off-target performance. ITI trades truthfulness against helpfulness as strength
rises [li2023iti]; SAE feature steering has a narrow safe band (~−5..+5) before
coherence collapses and off-target features move (steer gender bias down, age
bias up) [durmus2024evaluating]; contrastive decoding can over-penalize valid
tokens [li2022contrastive]; CFG's γ is an explicit adherence-vs-fluency knob
[sanchez2023cfg]; DPO over-optimizes under near-deterministic preferences,
motivating IPO [azar2023ipo]; unlearning trades forgetting against utility, and
no baseline achieves clean forgetting [maini2024tofu]. The practical corollary
is that *strength must be a tunable, monitored dial* with a coherence guardrail,
not a fixed setting — and that any steer should be A/B'd against an unsteered
baseline to price its collateral [chalnev2024saets; xie2025comparative;
dasilva2025steeringoffcourse].

### 4.2 Faithfulness & monitorability as the substrate for read-based steering

Every *read-based* method — probes, SAE features, CoT monitoring, judges —
inherits a trust problem: the signal it reads may not reflect what the model
actually does. Three results anchor this. Probes can lock onto prominent
non-knowledge features unless sanity-checked [farquhar2023challenges] and fire
on spurious correlates even when they generalize [goldowsky2025deception]. CoT
can rationalize a hidden cause without mentioning it [turpin2023lm;
arcuschin2025wild] and verbalizes used hints only ~25–39% of the time
[chen2025reasoning]. Monitorability is *trainable-away*: optimizing against a CoT
monitor produces obfuscation [baker2025monitoring], so it must be measured and
preserved as a fragile resource [korbak2025monitorability; emmons2025pragmatic;
meek2025verbosity]. The unifying lesson: read-based control is only as
trustworthy as the faithfulness of the channel it reads, so (i) aggregate
multiple independent signals, (ii) prefer diverse-trained probes
[liu2024hyperplane], (iii) corroborate verbalized intent before acting on it,
and (iv) never apply optimization pressure that rewards a *clean-looking*
signal.

### 4.3 Evaluation & benchmarks

The field's benchmarks cluster by purpose: **capability/factuality** — TruthfulQA
(ITI, DoLa) [li2023iti; chuang2023dola], GSM8K/MATH and PRM800K/ProcessBench for
reasoning and process supervision [obrien2023cdreasoning; lightman2023verify;
zheng2024processbench]; **safety** — WMDP and TOFU for unlearning [li2024wmdp;
maini2024tofu], AgentHarm for agent misuse [andriushchenko2024agentharm],
red-team-hours for jailbreak defense [sharma2025constitutionalclassifiers];
**steerability/monitorability** — the steerability index [miehling2024steerability]
and legibility/coverage/verbosity autoraters [emmons2025pragmatic;
meek2025verbosity]; **editing** — RippleEdits and KnowEdit [cohen2024ripple;
zhang2024knowedit]; **oversight** — MT-Bench/Chatbot Arena for judges
[zheng2023judge], BashBench for agentic control [bhatt2025ctrlz]. The recurrent
methodological gap, sharpest in the reliability papers
[dasilva2025steeringoffcourse; braun2025unreliability; durmus2024evaluating], is
that a method's headline result frequently *does not transfer* across models,
behaviors, or layers — so per-model, per-behavior validation with off-target
controls is now the expected standard.

### 4.4 Composition of methods

Steering methods increasingly compose. Model arithmetic gives an explicit
algebra over logits that subsumes CFG/GeDi/DExperts [dekoninck2024arithmetic];
CAA stacks on finetuning and system prompts [rimsky2024caa]; task-vector
arithmetic and TIES-Merging compose weight-space behaviors while resolving
interference [ilharco2023taskarith; yadav2023ties]; conditional policies compose
objectives at inference [wang2024clp]; and the AI-control agenda *composes an
untrusted strong model with a trusted weak monitor* into a protocol
[greenblatt2023aicontrol]. Composition is also where the pragmatic split
"SAEs for *what*, contrastive vectors for *how hard*" lives [chalnev2024saets],
and where a supervisor can layer decoding constraints, activation steers, and
guardrail rails at different loci without conflict.

### 4.5 Robustness & the adversarial angle

Because behaviors are directions, safety built on a single direction is
trivially reversible: abliteration removes refusal with a rank-one edit
[arditi2024refusal], ~10 finetuning examples strip alignment [qi2023finetuning],
and a few direction ablations recover "unlearned" capability
[lucki2024adversarial]. The defenses entangle safety deeper — representation
rerouting [zou2024circuitbreakers], latent adversarial training [casper2024lat;
sheshadri2024tlat], tamper-resistance via meta-learning over simulated attacks
[tamirisa2024tar] — but the arms race is live. On the *oversight* side, the
adversarial framing is native: AI control assumes intentional subversion and
red-teams the protocol [greenblatt2023aicontrol; bhatt2025ctrlz], sandbagging
and evaluation-awareness show models can strategically underperform or detect
that they're being watched [vanderweij2024sandbagging; tice2024noise;
nguyen2025evalawareness], and CoT monitoring can be defeated by obfuscation
[baker2025monitoring]. The synthesis: *treat every steer and every monitor as
security-relevant*, assume the target may be indifferent or adversarial, and
prefer multi-layer/affine/rerouting controls over lone directions when
durability matters.

---

## 5. Applied: opportunities for agent-steering systems (goldfive lens)

goldfive wraps an agent tree and runs an **observe → detect drift → intervene**
loop *without requiring the agent to cooperate* — no prompt-contract, no
mandated tool calls. Its design (AGENCY-PRESERVATION.md) reframes the system as
a *dormant supervisor*: zero trajectory footprint while the agent makes
progress, guardrails always armed on observed facts, and steering engaged only
on goal-referenced drift, delivered through the *least-invasive surface*. Its
design discusses three intervention loci — external/black-box (context and
message injection), inference-engine (logits/KV/activations), and model-weights
(control tokens/adapters/training). This section maps the surveyed literature
onto that structure. We organize by the loop stage each method upgrades —
DETECTION, ACTUATOR, or LOOP/OUTCOME — and then confront the two hard
tradeoffs: portability and agency-preservation.

### 5.1 Methods that upgrade DETECTION

The read-side literature is a near-perfect fit for goldfive's observer, and
most of it is cooperation-free.

- **Linear probes as a white-box drift signal.** Truthfulness, deception, and
  defection probes run on each agent step's residual stream fire *before* the
  agent emits a harmful/dishonest action [azaria2023internal; marks2023geometry;
  macdiarmid2024probes; goldowsky2025deception]. Defection probes catch
  backdoored behavior at >99% AUROC using *generic, trigger-agnostic* contrast
  pairs [macdiarmid2024probes] — exactly the regime where goldfive supervises a
  third-party agent whose triggers it does not know.
- **The say-vs-know gap monitor.** Models often internally encode the correct
  answer while outputting a wrong one [orgad2024know]; goldfive can probe the
  internal "correct answer" and flag divergence from what the agent actually
  says or does — catching confident hallucination and dishonest tool-use at the
  source, a sharper signal than output heuristics.
- **Confidence-gated intervention.** P(True)/P(IK) [kadavath2022know] and
  Semantic Entropy Probes [kossen2024sep] give a cheap internal uncertainty
  signal to decide *when* to intervene versus let the agent proceed — throttling
  interventions by internal certainty rather than surface cues, which directly
  serves the "dormant unless drifting" identity.
- **Evaluation-awareness alarm.** A linear eval-vs-deployment direction lets
  goldfive detect when a monitored agent "realizes" it is being observed and may
  change behavior [nguyen2025evalawareness]; combined with noise-injection
  capability recovery [tice2024noise] and the sandbagging threat model
  [vanderweij2024sandbagging], this yields an *underperformance* detector
  distinct from genuine incapability.
- **SAE features as a named drift channel.** Monitoring residual-stream
  activations through a pretrained SAE (e.g. Gemma Scope [lieberum2024gemmascope])
  flags when safety-relevant features (deception, sycophancy, off-task) spike —
  a cooperation-free observability channel that names *why* the agent is
  drifting, and feeds goldfive's observer-note surface with human-legible
  reasons [templeton2024monosemanticity; marks2024circuits].
- **CoT-faithfulness-aware judging.** Baker et al. is the decisive result for
  goldfive's observer: reading the agent's CoT catches misbehavior far better
  than watching only tool-calls/outputs [baker2025monitoring], so the observer
  should ingest reasoning tokens. But it must weight that signal by a runtime
  *monitorability* metric — legibility + coverage [emmons2025pragmatic] and
  verbosity/faithfulness [meek2025verbosity] — because faithfulness is low
  [chen2025reasoning; arcuschin2025wild] and family-dependent. The specific
  in-the-wild unfaithfulness modes — restoration errors (silent self-correction)
  and post-hoc rationalization [arcuschin2025wild] — are first-class drift
  signatures matching goldfive's "controller-in-disguise" pattern. A degrading
  trace over a session is an early warning that the CoT channel is closing
  [korbak2025monitorability].
- **Step-level error localization.** A lightweight PRM / ProcessBench-style
  earliest-error localizer [lightman2023verify; zheng2024processbench;
  khalifa2025thinkprm] gives a precise intervention *point* — the first bad
  reasoning step — rather than only an end-of-turn outcome check.
- **Judge machinery for drift verdicts.** LLM-as-judge with known bias
  mitigations (position-swap, reference-guided grading, ensembling)
  [zheng2023judge; liu2023geval; gu2024judgesurvey] and a meta-judge to catch
  judge errors [wu2024metareward] is the direct antecedent of goldfive's
  judge-only mode; the survey's failure taxonomy [gu2024judgesurvey] is a
  checklist for calibrating the suspicion threshold that triggers intervention.

**Robustness discipline for detection.** goldfive must bake in the reliability
caveats: require diverse-trained probes [liu2024hyperplane], treat single probes
as advisory and aggregate multiple directions [goldowsky2025deception], and
apply the CCS sanity checks against prominent-but-non-knowledge features
[farquhar2023challenges].

### 5.2 Methods that upgrade the ACTUATOR

The intervention side maps onto goldfive's least-invasive-surface ladder (shape
the choice set → edit perception → speak → constrain structurally → stop). Note
the tension flagged in §5.4: several of these are *wheel-grabs* and belong only
at the harder rungs.

- **Cooperation-free activation steering (inference-engine locus).**
  Mean-difference/CAA vectors [rimsky2024caa], the refusal direction
  [arditi2024refusal], and ITI/RepE honesty directions [li2023iti; zou2023repe]
  let goldfive nudge a running agent via forward-pass activation edits with no
  tool call or prompt contract — the mechanism goldfive most needs against an
  uncooperative agent. EAST shows this reaches *agent-level actions*
  (exploration/caution) where token sampling cannot [rahn2024east].
- **Conditional/gated steering as a drift-response switch.** CAST's
  condition-vector-as-switch [lee2025cast] is the activation-space analog of
  goldfive's intervention ladder: fire a behavior vector *only* when an
  activation-similarity detector matches, giving targeted correction instead of
  always-on steering that degrades off-target behavior — a native fit for
  "engage only on drift."
- **Representation rerouting as a redirect primitive.** Circuit Breakers already
  reduce harmful *agent actions* under attack [zou2024circuitbreakers], offering
  a representation-level halt/redirect of a harmful tool-call trajectory —
  a harder rung than an advisory note.
- **Decoding-time steering (logit locus).** goldfive's drift detectors could
  feed a GeDi/FUDGE-style small discriminator to apply a *soft* per-token logit
  bias that stops before hard-cancelling a turn [krause2020gedi; yang2021fudge].
  A compact "supervisor proxy" (proxy-tuning/DExperts) ships a logit delta that
  steers any wrapped black-box model without retraining it [liu2024proxytuning;
  liu2021dexperts]. DoLa and CD-for-reasoning improve factuality/reasoning with
  no extra prompt or cooperation [chuang2023dola; obrien2023cdreasoning].
  Model arithmetic gives an *intervention grammar* — express escalating
  interventions as arithmetic over base + supervisor logits rather than discrete
  states [dekoninck2024arithmetic] — and CFG's γ is a continuous knob for a
  soft-to-hard ladder scaled by measured drift severity [sanchez2023cfg].
- **Structured decoding to fence tool-calls.** Outlines/XGrammar/GCD can
  *hard-guarantee* an agent's tool invocations and JSON conform to a schema — a
  cheap, cooperation-free structural constraint (ladder rung "constrain
  structurally") when goldfive detects an agent going off-protocol
  [willard2023outlines; dong2024xgrammar; geng2023gcd].
- **Trajectory-aware value-guided decoding.** RAD/ARGS/Controlled Decoding guide
  by a scalar scorer but only *locally* [deng2023rad; khanov2024args;
  mudgal2023controlled]; wiring goldfive's *trajectory-level* goal-alignment
  estimate in as the value function would make the decode-time signal reflect
  where the whole run is heading — the exact gap §3.3 identified.
- **Control tokens and budget forcing (weights/input locus).** Trained mode
  tokens (`/think`, `/no_think`, thinking budget) are steering handles goldfive
  can inject at turn boundaries to throttle reasoning depth/latency
  [qwen2025qwen3] — but they *leak* [wang2025hybridthinking], so goldfive must
  *verify* the mode actually changed by observing token counts/behavior rather
  than trusting the toggle (goldfive's own empirical-baseline discipline).
  Budget forcing is directly transplantable: force end-of-thinking to cut a
  runaway agent, or inject "Wait" to force reconsideration and break a loop
  [muennighoff2025s1].
- **Hot-swappable behavior modules (adapter locus).** Precompiled LoRA/QLoRA
  deltas [hu2022lora; dettmers2023qlora] and task-vector arithmetic
  [ilharco2023taskarith] with TIES-style sign resolution [yadav2023ties] give a
  steering *vocabulary* an observer can attach/detach mid-run. Side-memory
  routing (GRACE/WISE/SERAC) keeps the base agent untouched and stores runtime
  corrections in a routed memory keyed on activation similarity — a natural fit
  for cooperation-free intervention that accumulates across a long session
  without catastrophic forgetting [hartvigsen2023grace; wang2024wise;
  mitchell2022serac]. On MoE backbones, the gate is itself an observable/steerable
  routing surface [shazeer2017moe].
- **Ctrl-Z resampling as a soft ladder rung.** Ctrl-Z's resample protocol
  [bhatt2025ctrlz] — re-sample a suspicious action and execute the
  least-threatening — is a *softer* intervention than cancel/pause that
  preserves useful work, an ideal rung between "observe" and "hard-cancel" and a
  concrete instance of goldfive's "shape the choice set" surface.
- **External guardrail rails at the tool boundary.** Llama Guard [inan2023llamaguard]
  and NeMo execution rails [rebedea2023nemo] give a drop-in input/output/tool-call
  filter that runs *independent* of the agent — composed as goldfive's
  execution-surface interventions, not as agent-side prompt edits.

**Actuator robustness discipline.** Single-direction controls are trivially
reversible [arditi2024refusal; qi2023finetuning; lucki2024adversarial], and
steering itself can erode safety margins — so goldfive must treat its own vector
interventions as *security-relevant*, prefer affine/multi-layer/rerouting
controls when durability matters [marshall2024affine; zou2024circuitbreakers],
gate on reliability (verify the target behavior has a coherent direction on the
specific model before trusting a vector) [braun2025unreliability;
dasilva2025steeringoffcourse], and monitor collateral features so a steer for
one behavior doesn't silently degrade others [durmus2024evaluating] or ripple
into entailed downstream actions [cohen2024ripple].

### 5.3 Methods that close the LOOP / ground OUTCOMES

- **Measure-then-optimize at runtime.** SAE-TS's loop — measure a steer's
  feature-space effect, then optimize the vector to hit a target with minimal
  side effects — maps directly onto goldfive's detect→intervene cycle, using SAE
  feature deltas as the *telemetry/reward* for choosing among candidate steers
  [chalnev2024saets].
- **Constitution-as-config.** CAI / Collective-CAI / Constitutional Classifiers
  show behavior can be governed by a small written principle set and distilled
  into a fast classifier [bai2022cai; huang2024ccai;
  sharma2025constitutionalclassifiers]. Rather than the regex heuristics goldfive
  retired, express intervention criteria as a natural-language constitution and
  distill it into a low-latency per-observation classifier.
- **Train-time vs run-time division of labor.** DPO/KTO/ORPO/SimPO make it cheap
  to bake in a broad default policy [rafailov2023dpo; ethayarajh2024kto;
  hong2024orpo; meng2024simpo]; goldfive supplies the *outer loop* that corrects
  the residual *per-episode* drift these static methods cannot address —
  positioning runtime steering as complementary, not competing. KTO's unpaired
  binary desirable/undesirable signals match what an online observer can emit,
  so goldfive's drift verdicts can be logged as KTO-style signals to feed
  periodic re-tuning of the base agent [ethayarajh2024kto].
- **Multi-objective steering at run time.** Conditional Language Policy conditions
  on objective weights so one model trades off helpfulness/safety/brevity at
  inference [wang2024clp] — goldfive could re-weight competing objectives
  per-episode based on observed context, rather than committing to one static
  weighting.
- **Adversarial loop-closing.** AI-Control's protocol formalism gives goldfive a
  way to *quantify* safety-vs-usefulness on a red-team [greenblatt2023aicontrol];
  goldfive should stress-test its observer with an adversarial "red-team agent"
  rather than trusting benign traces, since naive monitoring over-claims safety
  and self-correction fails without an external signal
  [huang2024cannotselfcorrect; bhatt2025ctrlz].

### 5.4 The portability tradeoff: black-box universality vs white-box power vs co-trained models

The three loci trade off along a clean gradient, and goldfive must choose per
deployment:

- **Black-box / external (context + message + logit-bias, structured decoding,
  guardrail rails).** Works over any API, portable across model families,
  survives model swaps — but *bounded and leaky*: prompt steering has limited,
  asymmetric reach [miehling2024steerability], control tokens leak
  [wang2025hybridthinking], and text interventions rely on the model heeding
  them, which an adversarial agent need not do. Best default for portability;
  weakest authority.
- **White-box / inference-engine (activation steering, probes, SAE features, KV
  edits).** Cooperation-free and genuinely causal — the refusal direction and
  CAA move behavior regardless of the agent's intent [arditi2024refusal;
  rimsky2024caa] — and provide the richest detection signals [macdiarmid2024probes;
  kossen2024sep]. But they require activation access (open weights or a
  white-box serving stack), per-model direction validation
  [braun2025unreliability; dasilva2025steeringoffcourse], and careful strength
  control to avoid coherence collapse [durmus2024evaluating]. Highest power on
  models goldfive can open; not portable to closed APIs.
- **Co-trained / weights (control tokens minted for the wrapped model, adapters,
  RLAIF from goldfive's own drift verdicts).** Highest authority and durability
  — a minted control token is a deterministic switch independent of the
  coordinator's prompt [keskar2019ctrl; qwen2025qwen3], and tamper-resistant
  training survives adversarial finetuning [tamirisa2024tar]. But it requires
  training access and per-model investment, and it re-introduces the very
  static-ness that goldfive's runtime loop exists to complement.

The pragmatic split the literature suggests: use **black-box** methods as the
portable default and for structural fences (schema/tool constraints), **white-box**
methods where goldfive can open the model — SAEs/probes for *what and whether to
steer*, contrastive/affine vectors for *how hard* [chalnev2024saets;
marshall2024affine] — and **co-trained** control tokens/adapters only where a
durable, deterministic switch justifies the training cost.

### 5.5 The agency-preservation tension: steering as wheel-grab vs advisory

goldfive's central design tension is that its most *powerful* actuators are also
its most *invasive*. Activation steering, representation rerouting, forced
end-of-thinking, and cancel/resample all *grab the wheel* — they override the
agent's own judgment about decomposition, delegation, and retries, which
AGENCY-PRESERVATION.md identifies as the core of the agent's agency. The
literature both sharpens and helps resolve this:

- **The case for external, non-cooperative oversight is empirically grounded.**
  Agents cannot reliably self-correct without an external signal
  [huang2024cannotselfcorrect], and AI control assumes the agent may be
  adversarial [greenblatt2023aicontrol] — validating goldfive's no-prompt-contract
  thesis: control must not depend on the agent choosing to comply.
- **But powerful steers are wheel-grabs and belong at the harder rungs.** The
  least-invasive surfaces map onto advisory/soft methods: attributed observer
  notes (speak), Ctrl-Z resampling and choice-set shaping (shape the choice set)
  [bhatt2025ctrlz], perception editing (subtract poisoned context), and
  structural fences (schema decoding) [willard2023outlines; dong2024xgrammar].
  Activation steering, rerouting, and forced termination are the *harder* rungs
  [arditi2024refusal; zou2024circuitbreakers; muennighoff2025s1], reserved for
  goal-referenced drift or hard safety, not steady-state nudging.
- **The instruction hierarchy is the model-native analog of a privileged
  supervisor.** Framing an inserted observer-note or directive at a *higher
  priority tier* than the coordinator/user text [wallace2024instructionhierarchy;
  zheng2025instructionladder] lets agents defer to it *without* a prompt-cooperation
  contract — a way to make advisory steering land without wheel-grabbing.
- **The monitorability tax constrains what goldfive may optimize.** Because
  optimizing against a CoT monitor teaches obfuscation [baker2025monitoring],
  goldfive must *not* apply steering pressure that rewards clean-looking
  reasoning; steer on *behavior*, keep the CoT channel un-optimized, and treat
  monitorability as a resource to preserve [korbak2025monitorability]. This is a
  direct constraint on the actuator design, not just a detection nicety.
- **Interpretable, auditable steering serves the agency ledger.** SAE named
  features [templeton2024monosemanticity] and honestly-attributed steer records
  give goldfive an observer-note surface that logs *what was steered and why* —
  supporting the "honestly attributed intervention" requirement rather than a
  silent wheel-grab.

The net design stance the literature supports: **detect widely (cheap,
cooperation-free, aggregated, faithfulness-weighted); actuate narrowly (soft and
advisory by default, powerful and cooperation-free only on goal-referenced drift
or hard safety); and always price the intervention** against an unsteered
baseline, its off-target collateral, and its erosion of monitorability and
agency.

---

## 6. Open problems & research frontier

1. **Reliability and transfer.** The single most consequential open problem:
   steering results routinely fail to transfer across models, behaviors, and
   layers [dasilva2025steeringoffcourse; braun2025unreliability]. A predictive
   theory of *when a behavior has a coherent, steerable direction* — beyond the
   train-time-activation-alignment heuristic [braun2025unreliability] — would
   turn steering from an empirical craft into engineering.
2. **Conditionality and entanglement.** Steering is entangled, not surgical
   [durmus2024evaluating]; conditional steering [lee2025cast] and
   feature-targeted optimization [chalnev2024saets] are early steps, but a
   general method to move one behavior *without* moving correlated ones remains
   open.
3. **Faithful, non-gameable read channels.** Every read-based method inherits a
   faithfulness gap [turpin2023lm; chen2025reasoning; farquhar2023challenges],
   and monitorability is trainable-away [baker2025monitoring]. Building oversight
   signals that survive optimization pressure and adversarial obfuscation —
   without paying an unbounded monitorability tax — is the central AI-control
   research question [korbak2025monitorability; greenblatt2023aicontrol].
4. **Durable safety beyond single directions.** Single-direction safeguards are
   trivially reversible [arditi2024refusal; qi2023finetuning; lucki2024adversarial];
   rerouting/LAT/TAR entangle safety deeper [zou2024circuitbreakers;
   sheshadri2024tlat; tamirisa2024tar], but robust, tamper-resistant control that
   does not sacrifice capability or over-refuse is unsolved.
5. **Trajectory-level rather than token-level control.** All decoding-time and
   activation methods act locally; none carry memory of where a *multi-step
   agent run* is heading [deng2023rad; mudgal2023controlled]. Trajectory-aware
   value functions and steering conditioned on run-level goal-alignment are an
   open, agent-specific frontier — precisely the gap a supervisor like goldfive
   targets.
6. **Detecting naturally-arising (vs. trained) deception and sandbagging.**
   Probes catch *trained* backdoors and roleplay deception well
   [macdiarmid2024probes; goldowsky2025deception] but offer no guarantees for
   deception that emerges naturally; sandbagging and evaluation-awareness
   [vanderweij2024sandbagging; tice2024noise; nguyen2025evalawareness] make this
   adversarial.
7. **Localization vs. intervention.** Interpretability-located circuits do not
   predict the best intervention site [hase2023localization]; validating
   intervention points by *behavioral effect* rather than tracing is an
   unresolved methodological requirement for any white-box steerer.
8. **Composition guarantees.** Method-composition algebras exist
   [dekoninck2024arithmetic; yadav2023ties] but formal guarantees about how
   composed steers interact — especially across loci (decoding + activation +
   guardrail) — are absent.
9. **Evaluation for agentic, multi-step settings.** Most benchmarks are
   single-turn; agentic control evaluation is nascent [bhatt2025ctrlz;
   andriushchenko2024agentharm], and realistic drift/oversight benchmarks that
   stress the observe→detect→intervene loop end-to-end are needed.
10. **Steering that preserves agency.** The tension in §5.5 is itself an open
    research program: methods that correct goal-drift while maximally preserving
    the agent's own means [wallace2024instructionhierarchy] — advisory rather
    than commanding — with measurable non-inferiority to no-steering.

---

## 7. References

All entries below were programmatically verified against a closed bibliography;
every `[key]` cited in the body appears here.

- **[andriushchenko2024agentharm]** Maksym Andriushchenko et al. (2024). *AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents*. arXiv (ICLR 2025). https://arxiv.org/abs/2410.09024
- **[arcuschin2025wild]** Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy (2025). *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful*. ICML 2026 / arXiv. https://arxiv.org/abs/2503.08679
- **[arditi2024refusal]** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda (2024). *Refusal in Language Models Is Mediated by a Single Direction*. NeurIPS 2024 (arXiv:2406.11717). https://arxiv.org/abs/2406.11717
- **[azar2023ipo]** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos (2023). *A General Theoretical Paradigm to Understand Learning from Human Preferences*. AISTATS 2024. https://arxiv.org/abs/2310.12036
- **[azaria2023internal]** Amos Azaria, Tom Mitchell (2023). *The Internal State of an LLM Knows When It's Lying*. Findings of EMNLP. https://arxiv.org/abs/2304.13734
- **[bai2022cai]** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. (Anthropic) (2022). *Constitutional AI: Harmlessness from AI Feedback*. arXiv. https://arxiv.org/abs/2212.08073
- **[baker2025monitoring]** Bowen Baker, Joost Huizinga, Leo Gao, ... David Farhi (OpenAI) (2025). *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*. arXiv. https://arxiv.org/abs/2503.11926
- **[bhatt2025ctrlz]** Aryan Bhatt, Cody Rushing, Adam Kaufman, Tyler Tracy, Vasil Georgiev, David Matolcsi, Akbir Khan, Buck Shlegeris (2025). *Ctrl-Z: Controlling AI Agents via Resampling*. arXiv. https://arxiv.org/abs/2504.10374
- **[bowman2022scalableoversight]** Samuel R. Bowman et al. (Anthropic) (2022). *Measuring Progress on Scalable Oversight for Large Language Models*. arXiv. https://arxiv.org/abs/2211.03540
- **[braun2025unreliability]** Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, Dmitrii Krasheninnikov (2025). *Understanding (Un)Reliability of Steering Vectors in Language Models*. arXiv. https://arxiv.org/abs/2505.22637
- **[bricken2023monosemanticity]** Trenton Bricken, Adly Templeton, Joshua Batson, et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread (Anthropic). https://transformer-circuits.pub/2023/monosemantic-features/index.html
- **[burns2022ccs]** Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt (2022). *Discovering Latent Knowledge in Language Models Without Supervision*. ICLR 2023 / arXiv. https://arxiv.org/abs/2212.03827
- **[casper2024lat]** Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell (2024). *Defending Against Unforeseen Failure Modes with Latent Adversarial Training*. arXiv. https://arxiv.org/abs/2403.05030
- **[chalnev2024saets]** Sviatoslav Chalnev, Matthew Siu, Arthur Conmy (2024). *Improving Steering Vectors by Targeting Sparse Autoencoder Features*. arXiv. https://arxiv.org/abs/2411.02193
- **[chen2025reasoning]** Yanda Chen, Joe Benton, Ansh Radhakrishnan, et al. (Anthropic Alignment Science) (2025). *Reasoning Models Don't Always Say What They Think*. arXiv. https://arxiv.org/abs/2505.05410
- **[christiano2017deeprl]** Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017). *Deep Reinforcement Learning from Human Preferences*. NeurIPS. https://arxiv.org/abs/1706.03741
- **[christiano2021elk]** Paul Christiano, Ajeya Cotra, Mark Xu (2021). *Eliciting Latent Knowledge: How to tell if your eyes deceive you*. ARC technical report. https://www.alignment.org/blog/arcs-first-technical-report-eliciting-latent-knowledge/
- **[chuang2023dola]** Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, Pengcheng He (2023). *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models*. ICLR 2024 / arXiv. https://arxiv.org/abs/2309.03883
- **[cohen2024ripple]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva (2024). *Evaluating the Ripple Effects of Knowledge Editing in Language Models*. TACL. https://arxiv.org/abs/2307.12976
- **[dai2022kn]** Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei (2022). *Knowledge Neurons in Pretrained Transformers*. ACL. https://arxiv.org/abs/2104.08696
- **[dasilva2025steeringoffcourse]** Patrick Queiroz Da Silva, Hari Sethuraman, Dheeraj Rajagopal, Hannaneh Hajishirzi, Sachin Kumar (2025). *Steering off Course: Reliability Challenges in Steering Language Models*. ACL 2025 (arXiv:2504.04635). https://arxiv.org/abs/2504.04635
- **[dathathri2019pplm]** Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, Rosanne Liu (2019). *Plug and Play Language Models: A Simple Approach to Controlled Text Generation*. ICLR 2020 / arXiv. https://arxiv.org/abs/1912.02164
- **[deepseek2025r1]** DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv. https://arxiv.org/abs/2501.12948
- **[dekoninck2024arithmetic]** Jasper Dekoninck, Marc Fischer, Luca Beurer-Kellner, Martin Vechev (2024). *Controlled Text Generation via Language Model Arithmetic*. ICLR 2024 / arXiv. https://arxiv.org/abs/2311.14479
- **[deng2023rad]** Haikang Deng, Colin Raffel (2023). *Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model*. EMNLP 2023. https://arxiv.org/abs/2310.09520
- **[dettmers2023qlora]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS. https://arxiv.org/abs/2305.14314
- **[dong2024xgrammar]** Yixin Dong, Charlie F. Ruan, et al. (2024). *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*. arXiv / MLSys. https://arxiv.org/abs/2411.15100
- **[durmus2024evaluating]** Esin Durmus, Alex Tamkin, Jack Clark, et al. (Anthropic) (2024). *Evaluating Feature Steering: A Case Study in Mitigating Social Biases*. Anthropic research. https://www.anthropic.com/research/evaluating-feature-steering
- **[elhage2022superposition]** Nelson Elhage, Tristan Hume, Catherine Olsson, et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread (Anthropic). https://transformer-circuits.pub/2022/toy_model/index.html
- **[emmons2025pragmatic]** Scott Emmons, Roland S. Zimmermann, David K. Elson, Rohin Shah (Google DeepMind) (2025). *A Pragmatic Way to Measure Chain-of-Thought Monitorability*. arXiv. https://arxiv.org/abs/2510.23966
- **[ethayarajh2024kto]** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela (2024). *KTO: Model Alignment as Prospect Theoretic Optimization*. ICML. https://arxiv.org/abs/2402.01306
- **[farquhar2023challenges]** Sebastian Farquhar, Vikrant Varma, Zachary Kenton, et al. (DeepMind) (2023). *Challenges with unsupervised LLM knowledge discovery*. arXiv. https://arxiv.org/abs/2312.10029
- **[fu2025fur]** Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasović, Yonatan Belinkov (2025). *Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps*. EMNLP 2025 / arXiv. https://arxiv.org/abs/2502.14829
- **[gao2024scaling]** Leo Gao, Tom Dupré la Tour, Henk Tillman, et al. (OpenAI) (2024). *Scaling and evaluating sparse autoencoders*. arXiv:2406.04093 (ICLR 2025). https://arxiv.org/abs/2406.04093
- **[geng2023gcd]** Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West (2023). *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP 2023. https://arxiv.org/abs/2305.13971
- **[goldowsky2025deception]** Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, Marius Hobbhahn (2025). *Detecting Strategic Deception Using Linear Probes*. ICML 2025. https://arxiv.org/abs/2502.03407
- **[greenblatt2023aicontrol]** Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger (2023). *AI Control: Improving Safety Despite Intentional Subversion*. arXiv (ICML 2024 oral). https://arxiv.org/abs/2312.06942
- **[gu2024judgesurvey]** Jiawei Gu et al. (2024). *A Survey on LLM-as-a-Judge*. arXiv. https://arxiv.org/abs/2411.15594
- **[guan2024deliberative]** Melody Y. Guan, Manas Joglekar, Eric Wallace, et al. (OpenAI) (2024). *Deliberative Alignment: Reasoning Enables Safer Language Models*. arXiv. https://arxiv.org/abs/2412.16339
- **[han2024peft]** Luping Wang, Sheng Chen, Linnan Jiang, et al. (2024). *Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies*. arXiv. https://arxiv.org/abs/2410.19878
- **[hartvigsen2023grace]** Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi (2023). *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors*. NeurIPS. https://arxiv.org/abs/2211.11031
- **[hase2023localization]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun (2023). *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models*. NeurIPS. https://arxiv.org/abs/2301.04213
- **[hendel2023taskvec]** Roee Hendel, Mor Geva, Amir Globerson (2023). *In-Context Learning Creates Task Vectors*. Findings of EMNLP 2023 (arXiv:2310.15916). https://arxiv.org/abs/2310.15916
- **[hong2024orpo]** Jiwoo Hong, Noah Lee, James Thorne (2024). *ORPO: Monolithic Preference Optimization without Reference Model*. EMNLP. https://arxiv.org/abs/2403.07691
- **[houlsby2019adapter]** Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML. https://arxiv.org/abs/1902.00751
- **[hu2022lora]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv (ICLR 2022). https://arxiv.org/abs/2106.09685
- **[huang2024ccai]** Saffron Huang, Divya Siddarth, Liane Lovitt, et al. (Anthropic / Collective Intelligence Project) (2024). *Collective Constitutional AI: Aligning a Language Model with Public Input*. ACM FAccT. https://arxiv.org/abs/2406.07814
- **[huang2024cannotselfcorrect]** Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou (2024). *Large Language Models Cannot Self-Correct Reasoning Yet*. ICLR 2024. https://arxiv.org/abs/2310.01798
- **[ilharco2023taskarith]** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi (2023). *Editing Models with Task Arithmetic*. ICLR 2023 (arXiv:2212.04089). https://arxiv.org/abs/2212.04089
- **[inan2023llamaguard]** Hakan Inan et al. (Meta) (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations*. arXiv. https://arxiv.org/abs/2312.06674
- **[kadavath2022know]** Saurav Kadavath, Tom Conerly, Amanda Askell, et al. (Anthropic) (2022). *Language Models (Mostly) Know What They Know*. arXiv. https://arxiv.org/abs/2207.05221
- **[kenton2024weakjudges]** Zachary Kenton et al. (Google DeepMind) (2024). *On scalable oversight with weak LLMs judging strong LLMs*. NeurIPS 2024. https://arxiv.org/abs/2407.04622
- **[keskar2019ctrl]** Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, Richard Socher (2019). *CTRL: A Conditional Transformer Language Model for Controllable Generation*. arXiv. https://arxiv.org/abs/1909.05858
- **[khalifa2025thinkprm]** Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, ... Lu Wang (2025). *Process Reward Models That Think*. TMLR / arXiv. https://arxiv.org/abs/2504.16828
- **[khan2024debate]** Akbir Khan et al. (2024). *Debating with More Persuasive LLMs Leads to More Truthful Answers*. ICML 2024. https://arxiv.org/abs/2402.06782
- **[khanov2024args]** Maxim Khanov, Jirayu Burapacheep, Yixuan Li (2024). *ARGS: Alignment as Reward-Guided Search*. ICLR 2024 / arXiv. https://arxiv.org/abs/2402.01694
- **[khattab2023dspy]** Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. (2023). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. ICLR 2024. https://arxiv.org/abs/2310.03714
- **[korbak2025monitorability]** Tomek Korbak, ... Yoshua Bengio, Dan Hendrycks, Aleksander Mądry (40+ authors, multi-lab) (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*. arXiv. https://arxiv.org/abs/2507.11473
- **[kossen2024sep]** Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs*. arXiv. https://arxiv.org/abs/2406.15927
- **[krause2020gedi]** Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, Nazneen Fatema Rajani (2020). *GeDi: Generative Discriminator Guided Sequence Generation*. Findings of EMNLP 2021 / arXiv. https://arxiv.org/abs/2009.06367
- **[lanham2023measuring]** Tamera Lanham, Anna Chen, Ansh Radhakrishnan, ... Samuel R. Bowman, Ethan Perez (Anthropic) (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning*. arXiv. https://arxiv.org/abs/2307.13702
- **[lee2023rlaif]** Harrison Lee, Samrat Phatale, Hassan Mansoor, et al. (Google) (2023). *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*. ICML 2024. https://arxiv.org/abs/2309.00267
- **[lee2025cast]** Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, Amit Dhurandhar (2024). *Programming Refusal with Conditional Activation Steering*. ICLR 2025 (arXiv:2409.05907). https://arxiv.org/abs/2409.05907
- **[lester2021prompttuning]** Brian Lester, Rami Al-Rfou, Noah Constant (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP. https://arxiv.org/abs/2104.08691
- **[li2021prefix]** Xiang Lisa Li, Percy Liang (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL. https://arxiv.org/abs/2101.00190
- **[li2022contrastive]** Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis (2022). *Contrastive Decoding: Open-ended Text Generation as Optimization*. ACL 2023 / arXiv. https://arxiv.org/abs/2210.15097
- **[li2023iti]** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. NeurIPS 2023 (arXiv:2306.03341). https://arxiv.org/abs/2306.03341
- **[li2024wmdp]** Nathaniel Li, Alexander Pan, Anjali Gopal, et al. (incl. Dan Hendrycks) (2024). *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*. ICML 2024. https://arxiv.org/abs/2403.03218
- **[lieberum2024gemmascope]** Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, et al. (DeepMind) (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. arXiv. https://arxiv.org/abs/2408.05147
- **[lightman2023verify]** Hunter Lightman, Vineet Kosaraju, Yura Burda, ... Karl Cobbe (OpenAI) (2023). *Let's Verify Step by Step*. ICLR 2024 / arXiv. https://arxiv.org/abs/2305.20050
- **[lindsey2024crosscoders]** Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, et al. (Anthropic) (2024). *Sparse Crosscoders for Cross-Layer Features and Model Diffing*. Transformer Circuits Thread (Anthropic). https://transformer-circuits.pub/2024/crosscoders/index.html
- **[liu2021dexperts]** Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, Yejin Choi (2021). *DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts*. ACL-IJCNLP 2021. https://arxiv.org/abs/2105.03023
- **[liu2021ptuning]** Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang (2021). *GPT Understands, Too*. arXiv (later AI Open). https://arxiv.org/abs/2103.10385
- **[liu2023geval]** Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023. https://arxiv.org/abs/2303.16634
- **[liu2024hyperplane]** Junteng Liu, Shiqi Chen, Yu Cheng, Junxian He (2024). *On the Universal Truthfulness Hyperplane Inside LLMs*. EMNLP 2024. https://arxiv.org/abs/2407.08582
- **[liu2024icv]** Sheng Liu, Haotian Ye, Lei Xing, James Zou (2023). *In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering*. ICML 2024 (arXiv:2311.06668). https://arxiv.org/abs/2311.06668
- **[liu2024proxytuning]** Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith (2024). *Tuning Language Models by Proxy*. COLM 2024 / arXiv. https://arxiv.org/abs/2401.08565
- **[lucki2024adversarial]** Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, Javier Rando (2024). *An Adversarial Perspective on Machine Unlearning for AI Safety*. TMLR (NeurIPS 2024 SoLaR workshop, best technical paper). https://arxiv.org/abs/2409.18025
- **[macdiarmid2024probes]** Monte MacDiarmid, Timothy Maxwell, Nicholas Schiefer, et al. (Anthropic) (2024). *Simple probes can catch sleeper agents*. Anthropic research blog. https://www.anthropic.com/research/probes-catch-sleeper-agents
- **[madaan2023selfrefine]** Aman Madaan et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. https://arxiv.org/abs/2303.17651
- **[maini2024tofu]** Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter (2024). *TOFU: A Task of Fictitious Unlearning for LLMs*. arXiv (COLM 2024). https://arxiv.org/abs/2401.06121
- **[mallen2023quirky]** Alex Mallen, Madeline Brumley, Julia Kharchenko, Nora Belrose (2023). *Eliciting Latent Knowledge from Quirky Language Models*. arXiv. https://arxiv.org/abs/2312.01037
- **[marks2023geometry]** Samuel Marks, Max Tegmark (2023). *The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets*. arXiv / COLM. https://arxiv.org/abs/2310.06824
- **[marks2024circuits]** Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*. arXiv:2403.19647 (ICLR 2025 Oral). https://arxiv.org/abs/2403.19647
- **[marshall2024affine]** Thomas Marshall, Adam Scherlis, Nora Belrose (2024). *Refusal in LLMs is an Affine Function*. arXiv. https://arxiv.org/abs/2411.09003
- **[meek2025verbosity]** Austin Meek, Eitan Sprejer, Iván Arcuschin, Austin J. Brockmeier, Steven Basart (2025). *Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity*. arXiv. https://arxiv.org/abs/2510.27378
- **[meng2022rome]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS. https://arxiv.org/abs/2202.05262
- **[meng2023memit]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau (2023). *Mass-Editing Memory in a Transformer*. ICLR. https://arxiv.org/abs/2210.07229
- **[meng2024simpo]** Yu Meng, Mengzhou Xia, Danqi Chen (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward*. NeurIPS. https://arxiv.org/abs/2405.14734
- **[miehling2024steerability]** Erik Miehling, Michael Desmond, Karthikeyan Natesan Ramamurthy, Elizabeth M. Daly, Pierre Dognin, Jesus Rios, Djallel Bouneffouf, Miao Liu (2024). *Evaluating the Prompt Steerability of Large Language Models*. arXiv (IBM Research); NAACL 2025. https://arxiv.org/abs/2411.12405
- **[mitchell2022mend]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning (2022). *Fast Model Editing at Scale*. ICLR. https://arxiv.org/abs/2110.11309
- **[mitchell2022serac]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn (2022). *Memory-Based Model Editing at Scale*. ICML. https://arxiv.org/abs/2206.06520
- **[mudgal2023controlled]** Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, Jilin Chen, Alex Beutel, Ahmad Beirami (2023). *Controlled Decoding from Language Models*. ICML 2024 / arXiv. https://arxiv.org/abs/2310.17022
- **[mudide2024switch]** Anish Mudide, Joshua Engels, Eric J. Michaud, Max Tegmark, Christian Schroeder de Witt (2024). *Efficient Dictionary Learning with Switch Sparse Autoencoders*. arXiv. https://arxiv.org/abs/2410.08201
- **[muennighoff2025s1]** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto (2025). *s1: Simple Test-Time Scaling*. arXiv / EMNLP 2025. https://arxiv.org/abs/2501.19393
- **[nguyen2025evalawareness]** Jord Nguyen, Khiem Hoang, Carlo Leonardo Attubato, Felix Hofstätter (2025). *Probing and Steering Evaluation Awareness of Language Models*. arXiv. https://arxiv.org/abs/2507.01786
- **[obrien2023cdreasoning]** Sean O'Brien, Mike Lewis (2023). *Contrastive Decoding Improves Reasoning in Large Language Models*. arXiv. https://arxiv.org/abs/2309.09117
- **[orgad2024know]** Hadas Orgad, Michael Toker, Zorik Gekhman, et al. (2024). *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations*. ICLR 2025 / arXiv. https://arxiv.org/abs/2410.02707
- **[ouyang2022instructgpt]** Long Ouyang, Jeff Wu, Xu Jiang, et al. (2022). *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)*. NeurIPS. https://arxiv.org/abs/2203.02155
- **[qi2023finetuning]** Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, Peter Henderson (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!*. ICLR 2024. https://arxiv.org/abs/2310.03693
- **[qwen2025qwen3]** Qwen Team (2025). *Qwen3 Technical Report*. arXiv. https://arxiv.org/abs/2505.09388
- **[rafailov2023dpo]** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS. https://arxiv.org/abs/2305.18290
- **[rahn2024east]** Nate Rahn, Pierluca D'Oro, Marc G. Bellemare (2024). *Controlling Large Language Model Agents with Entropic Activation Steering*. arXiv. https://arxiv.org/abs/2406.00244
- **[rajamanoharan2024gated]** Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, et al. (DeepMind) (2024). *Improving Dictionary Learning with Gated Sparse Autoencoders*. arXiv. https://arxiv.org/abs/2404.16014
- **[rajamanoharan2024jumprelu]** Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, et al. (DeepMind) (2024). *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*. arXiv. https://arxiv.org/abs/2407.14435
- **[rebedea2023nemo]** Traian Rebedea, Razvan Dinu, Makesh Sreedhar, Christopher Parisien, Jonathan Cohen (2023). *NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails*. EMNLP 2023 (System Demos). https://arxiv.org/abs/2310.10501
- **[rimsky2024caa]** Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024 (arXiv:2312.06681). https://aclanthology.org/2024.acl-long.828/
- **[sanchez2023cfg]** Guillaume Sanchez, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, Stella Biderman (2023). *Stay on topic with Classifier-Free Guidance*. ICML 2024 / arXiv. https://arxiv.org/abs/2306.17806
- **[saunders2022selfcritiquing]** William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, Jan Leike (2022). *Self-critiquing models for assisting human evaluators*. arXiv (OpenAI). https://arxiv.org/abs/2206.05802
- **[sharma2025constitutionalclassifiers]** Mrinank Sharma et al. (Anthropic) (2025). *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming*. arXiv. https://arxiv.org/abs/2501.18837
- **[shazeer2017moe]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR. https://arxiv.org/abs/1701.06538
- **[sheshadri2024tlat]** Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper (2024). *Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs*. arXiv. https://arxiv.org/abs/2407.15549
- **[shinn2023reflexion]** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. https://arxiv.org/abs/2303.11366
- **[stiennon2020summarize]** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (2020). *Learning to Summarize from Human Feedback*. NeurIPS. https://arxiv.org/abs/2009.01325
- **[tamirisa2024tar]** Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika (2024). *Tamper-Resistant Safeguards for Open-Weight LLMs*. ICLR 2025. https://arxiv.org/abs/2408.00761
- **[templeton2024monosemanticity]** Adly Templeton, Tom Conerly, Jonathan Marcus, et al. (Anthropic) (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic interpretability report (transformer-circuits). https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **[tice2024noise]** Cameron Tice, Philipp Alexander Kreer, Nathan Helm-Burger, Prithviraj Singh Shahani, Fedor Ryzhenkov, Fabien Roger, Clement Neo, Jacob Haimes, Felix Hofstätter, Teun van der Weij (2024). *Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models*. NeurIPS 2025 / arXiv. https://arxiv.org/abs/2412.01784
- **[todd2024funcvec]** Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau (2024). *Function Vectors in Large Language Models*. ICLR 2024 (arXiv:2310.15213). https://arxiv.org/abs/2310.15213
- **[touvron2023llama2]** Hugo Touvron, Louis Martin, Kevin Stone, et al. (Meta) (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv. https://arxiv.org/abs/2307.09288
- **[turner2023actadd]** Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid (2023). *Steering Language Models With Activation Engineering*. arXiv:2308.10248. https://arxiv.org/abs/2308.10248
- **[turpin2023lm]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. NeurIPS 2023 / arXiv. https://arxiv.org/abs/2305.04388
- **[vanderweij2024sandbagging]** Teun van der Weij, Felix Hofstätter, Ollie Jaffe, Samuel F. Brown, Francis Rhys Ward (2024). *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*. arXiv. https://arxiv.org/abs/2406.07358
- **[wallace2024instructionhierarchy]** Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, Alex Beutel (2024). *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*. arXiv (OpenAI). https://arxiv.org/abs/2404.13208
- **[wang2024clp]** Kaiwen Wang, Rahul Kidambi, Ryan Sullivan, Alekh Agarwal, Christoph Dann, et al. (2024). *Conditional Language Policy: A General Framework for Steerable Multi-Objective Finetuning*. EMNLP. https://arxiv.org/abs/2407.15762
- **[wang2024mathshepherd]** Peiyi Wang, Lei Li, Zhihong Shao, ... Zhifang Sui (2024). *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. ACL 2024 / arXiv. https://arxiv.org/abs/2312.08935
- **[wang2024wise]** Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen (2024). *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models*. NeurIPS. https://arxiv.org/abs/2405.14768
- **[wang2025hybridthinking]** Shouren Wang, Wang Yang, Xianxuan Long, Qifan Wang, Vipin Chaudhary, Xiaotian Han (2025). *Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?*. arXiv. https://arxiv.org/abs/2510.12680
- **[wei2022cot]** Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS. https://arxiv.org/abs/2201.11903
- **[wei2022flan]** Jason Wei, Maarten Bosma, Vincent Y. Zhao, et al. (2022). *Finetuned Language Models Are Zero-Shot Learners (FLAN)*. ICLR. https://arxiv.org/abs/2109.01652
- **[willard2023outlines]** Brandon T. Willard, Rémi Louf (2023). *Efficient Guided Generation for Large Language Models*. arXiv (Outlines library). https://arxiv.org/abs/2307.09702
- **[wu2024metareward]** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (2024). *Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge*. arXiv. https://arxiv.org/abs/2407.19594
- **[xie2025comparative]** Jiaqing Xie (2025). *A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering*. arXiv. https://arxiv.org/abs/2510.01246
- **[yadav2023ties]** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal (2023). *TIES-Merging: Resolving Interference When Merging Models*. NeurIPS. https://arxiv.org/abs/2306.01708
- **[yang2021fudge]** Kevin Yang, Dan Klein (2021). *FUDGE: Controlled Text Generation With Future Discriminators*. NAACL 2021. https://arxiv.org/abs/2104.05218
- **[yang2023opro]** Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen (2023). *Large Language Models as Optimizers (OPRO)*. ICLR 2024. https://arxiv.org/abs/2309.03409
- **[yuan2023rrhf]** Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang (2023). *RRHF: Rank Responses to Align Language Models with Human Feedback without tears*. NeurIPS. https://arxiv.org/abs/2304.05302
- **[yuan2024selfreward]** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (2024). *Self-Rewarding Language Models*. ICML. https://arxiv.org/abs/2401.10020
- **[zhang2024ctgsurvey]** Xun Liang, Hanyu Wang, Yezhaohui Wang, et al. (2024). *Controllable Text Generation for Large Language Models: A Survey*. arXiv. https://arxiv.org/abs/2408.12599
- **[zhang2024knowedit]** Ningyu Zhang, Yunzhi Yao, Bozhong Tian, et al. (2024). *A Comprehensive Study of Knowledge Editing for Large Language Models*. arXiv. https://arxiv.org/abs/2401.01286
- **[zhao2023slichf]** Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, Peter J. Liu (2023). *SLiC-HF: Sequence Likelihood Calibration with Human Feedback*. arXiv. https://arxiv.org/abs/2305.10425
- **[zheng2023ike]** Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, Baobao Chang (2023). *Can We Edit Factual Knowledge by In-Context Learning?*. EMNLP. https://arxiv.org/abs/2305.12740
- **[zheng2023judge]** Lianmin Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2306.05685
- **[zheng2024processbench]** Chujie Zheng, Zhenru Zhang, ... Junyang Lin (Qwen/Alibaba) (2024). *ProcessBench: Identifying Process Errors in Mathematical Reasoning*. arXiv. https://arxiv.org/abs/2412.06559
- **[zheng2025instructionladder]** Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar (2025). *Reasoning Up the Instruction Ladder for Controllable Language Models*. arXiv (under review). https://arxiv.org/abs/2511.04694
- **[zhou2022ape]** Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, et al. (2022). *Large Language Models Are Human-Level Prompt Engineers (APE)*. ICLR. https://arxiv.org/abs/2211.01910
- **[zou2023repe]** Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405. https://arxiv.org/abs/2310.01405
- **[zou2024circuitbreakers]** Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks (2024). *Improving Alignment and Robustness with Circuit Breakers*. NeurIPS 2024. https://arxiv.org/abs/2406.04313

---

*End of survey. Every `[key]` in the body resolves to an entry in §7; the
reference list was generated verbatim from a programmatically verified
bibliography in which every entry was confirmed to be a real paper.*
