# MGT-B v3: Theory and Implementation Reference

This document is the reference for the MGT-B v3 MVP. It explains the scientific motivation, the limits of the older martingale framing, the window-level design, the calibration strategy, and the software architecture.

## 1. Objective of MGT-B v3

MGT-B v3 is an inference-time decoding controller for small reasoning models, especially quantized models such as INT4 variants. It does not fine-tune the LLM and does not modify model weights. It wraps decoding with an external controller:

```text
monitor the reasoning trajectory
-> detect a drift or degeneration regime
-> backtrack
-> restore decoding and monitor state
-> re-decode with targeted anti-degeneration constraints
```

The core target is not single-token oddness. A reasoning failure is often a trajectory event: repetition, confidence collapse, confidence inflation around a loop, or a phase change after many healthy tokens. MGT-B v3 therefore works at window level.

## 2. Autoregressive Setup

At step \(t\), the model has generated:

```latex
W_{1:t-1} = (W_1, W_2, \dots, W_{t-1})
```

It produces:

```latex
p_t(v) = \mathbb{P}(W_t = v \mid \mathcal{F}_{t-1})
```

\(\mathcal{F}_{t-1}\) contains the prompt, previous tokens, logits, KV cache, decoding parameters, prior scores, alerts, and backtracking decisions. A response is a trajectory:

```latex
W_1, W_2, \dots, W_T
```

MGT-B v3 monitors this trajectory while preserving enough state to roll it back coherently.

## 3. Reminder: MGT-B v2

The v2 statistic used token-level entropy:

```latex
H_t = - \sum_{v \in \mathcal{V}} p_t(v)\log p_t(v)
```

and increment:

```latex
d_t = \log p_t(W_t) + H_t
```

with cumulative martingale:

```latex
M_n = \sum_{t=1}^{n} d_t
```

The martingale identity is clean:

```latex
\mathbb{E}[\log p_t(W_t) \mid \mathcal{F}_{t-1}]
=
\sum_{v \in \mathcal{V}}p_t(v)\log p_t(v)
=
-H_t
```

so:

```latex
\mathbb{E}[d_t \mid \mathcal{F}_{t-1}] = 0
```

and:

```latex
\mathbb{E}[M_t \mid \mathcal{F}_{t-1}] = M_{t-1}
```

The problem is semantic: v2 observes surprise under the model's own distribution, not validity of reasoning. A confident loop can be invisible. If:

```latex
p_t(W_t) \approx 1
```

then:

```latex
\log p_t(W_t) \approx 0,\quad H_t \approx 0,\quad d_t \approx 0
```

The model can repeat confidently and the v2 signal may stay calm.

## 4. Why a Naive Ville Guarantee Is Fragile

A tempting construction is:

```latex
E_t = \prod_{i=1}^{t} e_i
```

and Ville-style control:

```latex
\mathbb{P}_{H_0}\left(\exists t : E_t \geq \frac{1}{\delta}\right) \leq \delta
```

This would require:

```latex
\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \leq 1
```

In LLM decoding, this condition is not guaranteed. Scores are dependent: overlapping windows reuse tokens, n-grams reappear, entropy has phase structure, and decoding choices alter future distributions.

### Problem A: Marginal Is Not Conditional

An ECDF or conformal-like p-value may give a marginal statement:

```latex
\mathbb{P}_{H_0}(p_t \leq u) \leq u
```

The sequential product needs conditional validity. A product of marginally valid but dependent e-values is not automatically a valid e-process.

### Problem B: Positional Non-Exchangeability

Scores vary with position. Early reasoning, middle reasoning, endings, short traces, and long traces have different distributions. Repetition also increases mechanically with length. A global calibration pool can be biased, so MGT-B v3 uses positional buckets:

```text
[0,512), [512,1024), [1024,2048), [2048,4096), [4096,+inf)
```

### Problem C: The Problem Is a Changepoint

The operational event is usually:

```text
healthy trace prefix -> regime change -> degeneration
```

A classic product can be driven very low by a long healthy prefix. MGT-B v3 therefore uses a reset-like CUSUM-e shape.

## 5. Theoretical Design Decision

MGT-B v3 uses:

```latex
E_t = \max(1, E_{t-1}) \cdot e_t
```

or in log-space:

```latex
\log E_t = \max(0, \log E_{t-1}) + \log e_t
```

This is a detector shape motivated by sequential testing and changepoint detection. It is not presented as an exact finite-sample Ville guarantee for LLM decoding. The operational threshold is calibrated empirically on held-out healthy traces.

Correct claim:

```text
The theory inspires the detector form; empirical calibration controls the observed false-alert rate on healthy held-out traces.
```

## 6. Token-Level to Window-Level

MGT-B v3 does not trigger token by token. The pipeline is:

```text
log token-level signals
-> aggregate into windows
-> compute one score per window
-> update the detector per window
```

Default hyperparameters:

```text
window_size = 64
stride = 32
ngram_min = 6
ngram_max = 8
```

Window-level monitoring is more robust because adjacent tokens are strongly dependent, reasoning drift usually spans multiple tokens, block features are more comparable, and backtracking can target a suspicious segment.

## 7. Window Features

### 7.1 Mean Entropy

Entropy must be computed from logits before top-p, min-p, or other sampling truncation. Otherwise it measures the sampler rather than the model:

```latex
H_j = \frac{1}{|B_j|}\sum_{t \in B_j} H_t
```

### 7.2 Mean Chosen Log-Probability

```latex
\ell_t = \log p_t(W_t),\quad
C_j = \frac{1}{|B_j|}\sum_{t \in B_j}\ell_t
```

This captures the model's confidence in the path actually taken.

### 7.3 Continuous Repetition

For \(n \in \{6,7,8\}\), excluding n-grams from the prompt:

```latex
R_j =
\frac{\#\{\text{window n-grams already seen before}\}}
{\#\{\text{window n-grams}\}}
```

This is more stable than binary 2-5 gram triggers.

### 7.4 Confident Loop Score

When an n-gram \(g\) recurs, compute the mean log-probability of the current occurrence and compare it with prior occurrences:

```latex
D_j = \max(0, \overline{\ell}_{g,current} - \overline{\ell}_{g,past})
```

The window aggregates this by max in the MVP. This is central: repeated text with increasing confidence is much more suspicious than repeated text alone.

### 7.5 Local/Global Entropy Ratio

```latex
L_j = \log \frac{H_j^{local} + \epsilon}{H_j^{global} + \epsilon}
```

MGT-B v3 keeps:

```latex
L_j^+ = \max(0,L_j),\quad L_j^- = \max(0,-L_j)
```

A low local entropy is not bad by itself. It becomes suspicious when paired with repetition or confident-loop evidence.

## 8. ECDF Instead of Z-Score

Sparse signals such as confident-loop deltas and rare repetition are poor z-score candidates. If the calibration variance is tiny, z-scores can explode. MGT-B v3 uses upper-tail ECDF:

```latex
p_X(x)=\frac{1+\#\{x_i^{cal}\geq x\}}{N+1}
```

and optionally:

```latex
a_X(x)=-\log p_X(x)
```

The current MVP calibrates the final linear score by ECDF, and the architecture leaves room to calibrate individual feature scores before scoring.

## 9. Positional Calibration

Each positional bucket has its own calibration distribution. For score \(s_j\):

```latex
p_j =
\frac{1+\#\{s^{cal}_{b(j)} \geq s_j\}}{N_{b(j)}+1}
```

where \(b(j)\) is the bucket for the window end position.

## 10. Final Score

The MVP linear score is:

```latex
s_j =
w_H a_H +
w_C a_C +
w_R a_R +
w_D a_D +
w_{L+} a_{L+} +
w_{L-} a_{L-}
```

In code, the MVP uses raw window features with configurable weights. A later logistic mode should use a strict problem-level split:

```text
train -> learn weights
calib -> ECDF + p-values + threshold
test -> final evaluation
```

Do not split train/calib/test by window.

## 11. Betting Function and Detector

The betting function is:

```python
gammas = [0.1, 0.3, 0.5, 0.7]
b(p) = mean(gamma * p ** (gamma - 1) for gamma in gammas)
```

with clipping:

```python
p = max(p, 1e-6)
```

The detector updates in log-space:

```python
logE = max(0.0, logE) + loge
```

## 12. Empirical Threshold

Do not set \(A = 1/\delta\) automatically. The threshold is selected from healthy held-out traces:

1. Generate or replay healthy INT4 traces.
2. Compute window features.
3. Compute raw scores.
4. Convert scores to positional p-values.
5. Apply the detector over each trace.
6. Pick the smallest searched threshold whose false-alert rate per trace is at or below the target.

Default target:

```text
false_alert_rate_per_trace <= 0.05
```

## 13. Backtracking

On alert at window \(\tau\), estimate:

```text
last window where logE <= 0
```

Then:

```text
cp_token = cp_window * stride
rollback_token = max(prompt_len, cp_token - margin_tokens)
```

Rollback must restore all coupled state: token prefix, KV cache, monitor buffers, n-gram tables, detector state, and refractory period. Backtracking without cache crop and monitor truncate is incoherent.

## 14. Re-Decoding

Re-decoding is curative and targeted. Defaults:

```text
temperature = 0.6
light repetition penalty = 1.1
optional no-bad-ngrams from the suspect region
```

Do not apply a permanent repetition penalty from the beginning by default.

## 15. Refractory Period

After backtracking, the detector is disarmed for a small number of windows:

```text
refractory_windows = 2
```

This avoids alert/backtrack/immediate-alert loops.

## 16. Guarantees and Non-Guarantees

MGT-B v3 does not guarantee truth. It does not guarantee that every alert is a real error. It does not claim an exact Ville guarantee in the dependent LLM setting. It empirically calibrates the alert threshold on healthy traces and reports observed behavior.

## 17. Mandatory Ablations

Required comparisons:

```text
vanilla INT4
mgtb_v2_baseline
MGT-B v3
entropy threshold
direct score threshold
e-detector
random-trigger at the same mean trigger rate
fixed-k backtracking
adaptive changepoint backtracking
continue + penalty without backtracking
repetition penalty always-on
knockout without entropy
knockout without logprob
knockout without repetition
knockout without confident-loop
knockout without local entropy
global vs positional calibration
token-level vs window-level
```

Random-trigger is critical: if it matches MGT-B v3, the detector is not adding useful information.

## 18. Implementation Map

`features/` computes entropy, log-probability, n-gram repetition, confident-loop deltas, and window features.

`calibration/` implements ECDF, positional buckets, and empirical threshold search.

`detector/` implements the mixture betting function and CUSUM-e style detector.

`control/` implements cache cropping, no-bad-ngram helpers, and backtracking orchestration.

`generation/` contains the HuggingFace MVP loop.

`logging/` writes JSONL token, window, and backtrack events.

`eval/` and `baselines/` provide minimal comparison scaffolding.

## 19. Development Order

The intended implementation order is:

```text
scaffolding
offline feature extraction
calibration
offline detector
online HF generation without backtracking
backtracking
ablations
```

The MVP in this repository keeps every stage executable while leaving room for richer learned scoring and larger-scale evaluation.
