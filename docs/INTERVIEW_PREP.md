# Interview Preparation

Everything here maps to code you can open. If an interviewer asks "show me", you
should be able to point at the file.

**One rule above all others: never claim CIC-IDS2017 results you did not
produce.** The metrics currently in the README come from synthetic data. If you
have not retrained on the real dataset, say so before you're asked — an
interviewer who catches you inflating a number stops trusting everything else
you said. Volunteering it has the opposite effect.

---

## Part 1 — 20 technical questions about the project

### 1. Walk me through the architecture.

Four layers. **Data**: `data_loader.py` finds and concatenates CSVs and
normalises headers; `preprocessing.py` cleans and writes a parquet artifact.
**Training**: a stratified three-way split, a preprocessing pipeline fitted on
training rows only, three candidate classifiers compared on validation, the
winner evaluated once on test, and everything serialised into a single
`ModelBundle`. **Inference**: `ThreatPredictor` loads that bundle and is the only
path from a flow record to a prediction. **Consumers**: FastAPI, Streamlit and
pytest all call that same predictor.

The load-bearing decision is that last one — one inference path, three
consumers. It means the API and the dashboard cannot disagree, because there's
no second implementation to drift.

### 2. Why is the preprocessor saved inside the model file rather than separately?

Version skew. If they're two files, nothing stops someone loading v2 of the
model with v1 of the scaler. The predictions don't crash — they're just quietly
wrong, and it's one of the hardest production bugs to trace because every
component looks healthy. One artifact means one version and one consistent
transformation. `save_model_bundle` in `train.py`.

### 3. How do you prevent data leakage?

Two separate mechanisms, and it's worth naming both because they're different
problems.

**Column leakage**: I drop flow ID, source/destination IP, source port and
timestamp. In CIC-IDS2017 the attacker hosts are fixed, so a model given the
source IP learns "traffic from 172.16.0.1 is an attack." That scores near-perfect
on this dataset and detects nothing on any other network. It's the most common
flaw in published NIDS results.

**Fitting leakage**: the preprocessor is fitted on the training split only. The
scaler's median and IQR and the encoder's vocabulary are learned from train and
then applied to validation and test. Fitting on the full dataset first leaks the
held-out distribution into training.

### 4. Why macro-F1 for model selection instead of accuracy?

Because ~80% of flows are benign, so predicting "BENIGN" for everything scores
80% accuracy while detecting zero attacks. Accuracy is dominated by the majority
class.

Macro-F1 averages per-class F1 with equal weight, so a rare class contributes as
much as the common one. A model that ignores Infiltration gets punished for it.
I also report per-class recall so a silently-missed class is visible rather than
averaged away.

### 5. Why is the test set opened only once?

Because every time you look at test performance and change something in
response, you've used the test labels to make a decision — they've influenced
the model, so they're no longer held out and the score becomes optimistically
biased. All three models are compared on validation, the winner is chosen there,
and the test split is scored exactly once at the end. In `train_models()` the
test transform happens after the selection line, deliberately.

### 6. Why not SMOTE for the class imbalance?

SMOTE interpolates between existing minority samples. For network flows that
means fabricating records that can't physically exist — 2.5 packets, a
fractional SYN flag count, a byte total inconsistent with the packet count. You
teach the model a decision boundary shaped partly by impossible traffic.

I use balanced class weights instead, which adjusts the loss so minority errors
cost more without inventing data, plus majority downsampling in the training
split only. Validation and test keep the real distribution — otherwise the
false-alarm rate you measure isn't the one you'd get in production.

### 7. Explain your risk score.

A weighted blend of four signals: class severity (0.45), model confidence
(0.25), anomaly score (0.20), and model-independent traffic heuristics (0.10),
mapped onto 0–100 with four bands.

Two rules override the arithmetic. A confidently-benign flow the detector also
considers normal gets halved, keeping it firmly in LOW. And any flow the
detector flags gets a floor of 45 even if the classifier says benign — that's
the unknown-attack path the Isolation Forest exists for.

Be upfront about what it isn't: the weights are reasoned, not fitted to labelled
triage decisions. It's auditable — every component is returned in the response —
but it's a heuristic, not a calibrated probability. I'd want to learn those
weights from real analyst dispositions.

### 8. Why add an Isolation Forest when you already have a classifier?

They fail differently. The supervised model can only output classes it was
trained on, so a novel attack *must* be mapped onto a known label — usually
BENIGN. It has no way to say "I've never seen this."

The Isolation Forest is fitted on benign traffic only and never sees an attack
label. It answers a different question: "does this resemble normal traffic?" So
a zero-day shows up as a benign classification with a high anomaly score. That
disagreement is the signal, which is why I surface both rather than blending
them into one number.

### 9. How does an Isolation Forest actually work?

It builds trees by picking a random feature and a random split point,
recursively. Anomalies get isolated in fewer splits, because a point far from
the bulk of the data is easy to separate. The score is the average path length
across trees — shorter means more anomalous.

Two things worth knowing: `score_samples` returns values where *lower* means
more anomalous, which is unintuitive to expose in an API, so I invert and min-max
normalise against calibration bounds captured at training time. And the result
is not a probability — 0.94 means "top few percent of unusual relative to
training traffic."

### 10. Why is a false negative worse than a false positive here? Is it always?

A false negative is an intrusion nobody notices. A false positive costs an
analyst a few minutes. Orders of magnitude apart, so I optimise for recall.

But it flips at the extreme, and this is the part people miss. An IDS with a
high false-alarm rate gets muted by the team operating it — and a detector
nobody looks at has a real-world recall of zero. My own results show this: the
Logistic Regression baseline had the *highest* detection rate of the three
models, but at six times XGBoost's false alarm rate. It caught marginally more
attacks by flagging far more benign traffic. That's why I report detection rate
and false alarm rate side by side rather than optimising one.

### 11. Why bucket `destination_port` instead of scaling it?

Port numbers are nominal, not ordinal. Port 443 isn't "more" than port 80, and
the gap between 22 and 23 has no magnitude. Feeding the integer to a linear
model is meaningless, and tree models waste splits rediscovering boundaries I
already know.

So I map ports to service categories — `http`, `https`, `ssh`, `dns`, `smb`,
`database`, `ephemeral`. It also generalises: SSH brute-force on a non-standard
high port still lands in a sensible bucket.

### 12. Walk me through your derived features.

Nine, each targeting a named behaviour. `packets_per_second` and
`bytes_per_second` because a flood is defined by *rate* — a raw packet count
can't distinguish a DDoS from a long legitimate download. `fwd_bwd_packet_ratio`
because normal TCP is roughly balanced while a port scan sends many SYNs and
gets nothing back; it's the single cleanest scan indicator. `tcp_flag_density` —
flag counts over packet count — because well-behaved flows use a few flags
across their lifetime while SYN floods push it toward 1, as nearly every packet
is control-only with no payload.

Every ratio has a guarded denominator, because zero-duration and zero-reply
flows are *normal* in attack traffic, not edge cases. SYN floods routinely report
zero microseconds. There are tests for exactly that.

### 13. Why only about 29 features when the dataset has 78?

A deliberate trade-off. The compact set keeps the API contract fillable by hand,
keeps SHAP fast enough for interactive use, and reduces the chance of leaning on
a capture artefact. `--all-features` switches to every numeric column if you want
to compare.

I'd want to verify the trade-off empirically before defending it hard — run both
and compare macro-F1. That's a one-flag experiment I've built for but haven't
run on real data.

### 14. What happens if the API receives a flow with missing features?

It's scored. Only `destination_port` is required; everything else is imputed
with the training-set medians via the pipeline's `SimpleImputer`. Real collectors
export different subsets, so demanding all 29 features would make the API
unusable.

`align_features` reindexes onto the expected columns — missing ones become NaN
and get imputed, extra ones are dropped. That's also how a full 78-column
CICFlowMeter record can be posted without trimming.

### 15. What's the difference between your cleaning and your inference paths?

They're deliberately asymmetric, which surprised me when I first wrote it.

`clean_dataset` deduplicates, drops rare classes and drops rows — all fine for
training. `prepare_for_inference` does none of that, because at inference you
must return exactly one prediction per input row. If someone uploads a CSV with
500 rows, they get 500 predictions, duplicates included. There's a test that
feeds in deliberate duplicates and asserts the row count survives.

### 16. How do you know the model behaves the same after serialisation?

There's a test for it. It saves the bundle, reloads from disk, runs the same
flow through both, and asserts identical class and confidence to 1e-9. Round-trip
equivalence is exactly the kind of thing that silently breaks between "works on
my machine" and "works in the container."

### 17. Why do the API and dashboard both call the predictor directly instead of the dashboard calling the API?

Two reasons. They're independent consumers of one inference layer, so neither
can drift from the other — there's no second implementation to get out of sync.
And a dashboard outage can't take detection offline; they fail independently.

The cost is that the dashboard doesn't exercise the API path, so an API
regression wouldn't show up there. I cover that with API tests instead, and the
dashboard displays the API's configured URL so the coupling is at least visible.

### 18. What happens if the model file is missing when the API starts?

It starts in degraded mode rather than crash-looping. `/health` returns 200 with
`"status": "degraded"`, and the prediction endpoints return 503 with a message
telling you to run the training script.

That distinction matters operationally: an orchestrator should be able to tell
"the container is broken" from "no model has been deployed yet." Killing the
container for the second case gives you a restart loop and a useless error.

### 19. What security measures does the project itself implement?

It parses untrusted CSV, so: extension allowlist, size cap enforced before
parsing (so a large upload can't exhaust memory during read), row cap after
parsing, and every numeric field bounded in Pydantic rather than merely typed —
a negative packet count or a port above 65535 is a 422, not a quietly imputed
value.

Two subtler ones. Validation errors report the field and reason but *not* the
submitted value — FastAPI's default echoes input back, which both breaks on
non-serialisable values like `Infinity` and is a bad habit in a service handling
hostile input. And logs record request shape — path, status, duration, request
ID — never payload contents, because flow records are traffic metadata that may
be sensitive.

The container also runs as an unprivileged UID. And the project is deliberately
detection-only; there's no traffic-generation or attack tooling in it.

### 20. If you had two more weeks, what would you do?

Three things, in order.

**Retrain on real CIC-IDS2017 and publish honest numbers.** Everything currently
reported is synthetic. That's the biggest gap and I'd close it first.

**Calibrate the confidence output.** Right now it's a raw softmax value that I
call "confidence," which is not the same as a probability. Platt scaling or
isotonic regression against a held-out set would make it one — and the risk score
consumes that value, so calibration would improve the score too.

**Per-class thresholds instead of argmax**, tuned against an explicit
false-negative-to-false-positive cost ratio. Right now every class implicitly
gets the same threshold, which doesn't reflect that missing an Infiltration is
far worse than missing a PortScan.

---

## Part 2 — 10 Python questions relevant to this project

### 1. What does `@dataclass` give you, and why use it here?

Auto-generated `__init__`, `__repr__` and `__eq__` from type-annotated fields.
I use it for `ModelBundle`, `PredictionResult` and `RiskAssessment` — structured
records where a dict would lose type information and typo-safety. `frozen=True`
on the config classes makes them immutable and hashable, so config can't be
mutated at runtime.

Gotcha: mutable defaults need `field(default_factory=list)`, not `= []`, for the
same reason mutable default arguments are a bug.

### 2. Explain the generator in the CSV chunk reader.

`_read_one` yields DataFrames instead of returning a list. With `chunksize`,
pandas returns an iterator, and yielding each chunk means only one is in memory
at a time. CIC-IDS2017 is ~2.8M rows — materialising every chunk before
concatenating would defeat the point of chunking entirely.

### 3. Why `if_version`-style locking on the predictor singleton?

`get_predictor` uses double-checked locking: check for None, acquire the lock,
check again inside. Uvicorn serves requests across threads, so without it two
concurrent first-requests could both deserialise the model — wasted memory and a
race. The second check matters because another thread may have finished while
you were waiting for the lock.

### 4. What's the difference between `@st.cache_data` and `@st.cache_resource`?

`cache_data` is for serialisable return values and returns a copy per caller, so
mutation can't corrupt the cache — I use it for scored DataFrames. `cache_resource`
is for global singletons returned by reference — the model. You want one model in
memory, not a copy per session, and it isn't cheaply serialisable anyway.

### 5. Why does `add_derived_features` copy the input?

Because silently mutating a caller's DataFrame is a nasty surprise, especially
inside a scikit-learn pipeline where the same frame may be reused. It does
`out = frame.copy()` and returns the new object. There's a test asserting the
input's columns are unchanged after the call.

### 6. How does `FunctionTransformer` fit into the sklearn pipeline?

It wraps a plain function as a pipeline step so feature engineering happens
*inside* the fitted pipeline rather than as a separate call. That's what makes it
impossible to forget at inference — it's serialised with everything else.

The key argument is `validate=False`, which keeps the DataFrame intact instead of
coercing to a numpy array. The downstream `ColumnTransformer` selects columns by
name, so it needs the names to survive.

### 7. What does `handle_unknown="ignore"` do on the OneHotEncoder and why does it matter?

At inference the system will meet categories absent from training — a protocol
like GRE, or a port bucket that never appeared. The default raises. `ignore`
produces an all-zero block for that feature instead.

For a service that must not crash on unusual input, failing soft is right here.
There's a test that transforms a row with protocol 47 after fitting on a fixture
that never saw it.

### 8. Why `RobustScaler` rather than `StandardScaler`?

`StandardScaler` centres on the mean and scales by standard deviation, both of
which are dragged around by outliers. Flow features are extremely right-skewed —
a DDoS flow can have a packet rate four orders of magnitude above normal, which
would inflate the standard deviation and compress every benign flow into a
near-zero band.

`RobustScaler` uses the median and IQR, which barely move. Trees don't care about
scaling, but Logistic Regression and the Isolation Forest do, and sharing one
preprocessor across all models keeps the comparison honest.

### 9. How do the pytest fixtures avoid polluting the repository?

The `trained_bundle` fixture is session-scoped and swaps the `PATHS` object on
each module for one pointing into a `tmp_path_factory` directory before training.
Artifacts land in a temp dir that pytest cleans up, so running the tests never
overwrites `models/nids_model.joblib`.

Session scope matters because it trains a real model — doing that per-test would
make the suite unusably slow.

### 10. Why do you catch broad exceptions around the explainer and anomaly scorer?

Because neither is load-bearing. If SHAP fails, the correct behaviour is to
return the prediction without attributions, not to fail the request — the
prediction is the product, the explanation is a bonus. Same for anomaly scoring:
it degrades to zero scores and the supervised path still works.

Broad `except` is usually a smell, and I wouldn't use it around the prediction
itself. Here it's a deliberate isolation boundary, and every one of them logs the
exception with a traceback so failures are visible rather than swallowed.

---

## Part 3 — 10 machine learning questions

### 1. Precision vs recall vs F1 — define them in this domain.

Precision: of the flows I flagged as attacks, what fraction really were? Low
precision means analyst noise. Recall: of the actual attacks, what fraction did I
catch? Low recall means missed intrusions. F1 is their harmonic mean, which
punishes a bad score on either — you can't get a good F1 by sacrificing one
entirely.

### 2. Macro vs weighted F1 — when does the difference matter?

Macro averages per-class F1 with equal weight. Weighted averages by class
support. They diverge exactly when performance correlates with class frequency —
which is the normal case here. A model that nails BENIGN and fails on
Infiltration gets a good weighted F1 and a poor macro F1. I select on macro
precisely because I don't want the majority class hiding failures.

### 3. What's ROC-AUC and why is it less useful here?

The probability that a random positive is ranked above a random negative,
threshold-independent. It's reported, but with heavy imbalance it's optimistic —
the false positive rate has a huge denominator, so even many false positives
barely move it. Precision-recall AUC is more informative for rare classes. I use
it as a sanity check, not a selection metric.

### 4. Bagging vs boosting.

Bagging (Random Forest) trains trees in parallel on bootstrap samples and
averages them — it reduces variance and is hard to overfit by adding trees.
Boosting (XGBoost) trains sequentially, each tree fitting the previous
ensemble's residuals — it reduces bias, usually scores higher on tabular data,
but *can* overfit with too many rounds and is more sensitive to hyperparameters.

### 5. How would you tune hyperparameters without touching the test set?

Randomised search with stratified k-fold cross-validation on the *training*
split, scoring macro-F1. Randomised over grid because with a large space it
finds good configurations in far fewer fits. The validation split stays for
final model comparison and the test split is untouched throughout.

The rigorous version is nested CV — an inner loop for tuning, an outer for
estimating generalisation — but it multiplies cost and I'd only reach for it if
the dataset were small enough to make a single split unreliable.

### 6. What is SHAP and why use TreeSHAP?

SHAP assigns each feature a contribution to a specific prediction, derived from
Shapley values in cooperative game theory — the feature's average marginal
contribution across all orderings. Computing that exactly is exponential.
TreeSHAP exploits tree structure to do it in polynomial time, which is what makes
per-prediction explanation feasible interactively.

Caveat worth volunteering: with correlated features, attribution spreads across
the correlated group, so an individually important feature can look weak.

### 7. Global feature importance vs SHAP — when do you want each?

Global importance answers "what does this model rely on overall" — useful for
sanity-checking that it learned rate and asymmetry rather than something
spurious. SHAP answers "why *this* flow" — which is the only one that helps an
analyst challenging a specific alert. The dashboard shows both, in different
places, for those different jobs.

### 8. Your confidence is 0.99 — what does that actually mean?

Strictly, that the softmax output for the winning class was 0.99. It is *not* a
calibrated probability, and I shouldn't call it one. Tree ensembles are typically
overconfident, so 0.99 does not mean "99 times out of 100 this is right."

Making it a real probability needs calibration — Platt scaling or isotonic
regression on a held-out set — checked with a reliability diagram. That's on the
improvements list, and it matters here because the risk score consumes this
value.

### 9. How would you detect that the model has gone stale in production?

Traffic distributions shift as networks and applications change. I'd monitor
input feature distributions against the training reference — a PSI or KS test per
feature — and alert on divergence. I'd also watch the *predicted* class mix:
a sudden change with no corresponding infrastructure change usually means drift,
not an attack wave.

Ground-truth labels arrive late in security, if at all, so distribution
monitoring is the practical signal rather than live accuracy.

### 10. Your model scores 98% — why should I be sceptical?

You should be, and I'd tell you why before you asked: that number came from
synthetic data I generated, where classes are drawn from deliberately separable
distributions. It demonstrates the pipeline runs; it says nothing about
detection performance.

Even on real CIC-IDS2017, high numbers deserve suspicion. The dataset is a
scripted testbed with clean class separation and known labelling errors, and the
usual cause of a suspiciously high score is leakage through identifier columns —
which is why I drop IPs, ports and timestamps explicitly. The honest test is
performance on traffic from a different network entirely, which I haven't run.

---

## Part 4 — 10 networking / cybersecurity questions

### 1. What is a network flow, and why use flows instead of packets?

A flow is one bidirectional conversation identified by the 5-tuple: source IP,
source port, destination IP, destination port, protocol. Instead of raw bytes,
it's described by statistics — duration, packet counts each way, size
distributions, inter-arrival times, TCP flag counts.

Two advantages. Scale: a million packets become a handful of rows. Privacy: flow
records carry no payload, so you analyse behaviour without reading content.
The trade-off is that anything only visible in the payload is invisible to you.

### 2. Explain the TCP three-way handshake and how a SYN flood abuses it.

Client sends SYN, server replies SYN-ACK and allocates state in a backlog queue,
client sends ACK and the connection is established.

A SYN flood sends many SYNs and never sends the final ACK — often with spoofed
sources so the SYN-ACK goes nowhere. Each half-open connection holds server
state until it times out, and the backlog fills, so legitimate connections are
refused. In flow features this is unmistakable: very high SYN count, near-zero
backward packets, tiny or zero duration, `tcp_flag_density` near 1.

### 3. DoS vs DDoS?

DoS comes from one source; DDoS from many coordinated hosts, usually a botnet.
DDoS is harder to mitigate because you can't just block one address, and the
aggregate volume can be far larger. At the individual flow level they look
similar, which is partly why my model sometimes confuses the two — the
distinguishing feature is the *number of distinct sources*, which is a property
of the traffic aggregate, not any single flow. A per-flow classifier
structurally can't see it.

### 4. How does a port scan appear in flow data?

Many very short flows from one source to many destination ports, each with one
or two forward packets and almost no reply traffic. `fwd_bwd_packet_ratio` goes
extreme, duration is tiny, payload is near zero.

Variants differ in flags: SYN/half-open scans send SYN and abandon; FIN, NULL and
XMAS scans use unusual flag combinations to evade simple filters. That's why I
include individual flag counts rather than just a total.

### 5. What's a brute-force attack and how would you distinguish it from normal logins?

Repeated authentication attempts against a service — SSH on 22, FTP on 21 — with
different credentials each time. In flow terms: many short, similarly-sized
flows to the same destination port, at machine-like regularity.

The discriminator is the *pattern*: humans mistype a password twice, not four
hundred times at even intervals. Flow-level features capture the size and timing
regularity; the count itself needs aggregation across flows, which my per-flow
model doesn't do — a real deployment would pair this with a rate rule.

### 6. What is botnet C2 traffic and why is it hard to detect?

Command-and-control is a compromised host checking in with its operator for
instructions. It's hard precisely because it's designed to look boring — low
volume, often over HTTPS on port 443, blending with normal web traffic.

The exploitable signal is *periodicity*: automated beaconing has regular
intervals a human browsing session doesn't. My model gets partial signal from
packet-size regularity and port category, but genuine beacon detection needs
temporal analysis across multiple flows from one host, which is on my
improvements list.

### 7. Signature-based vs anomaly-based detection.

Signature-based matches known-bad patterns — precise, fast, explainable, and
structurally blind to anything new. Anomaly-based models normal and flags
deviation — can catch novel attacks, but produces more false positives and
"unusual" isn't the same as "malicious."

This project uses both deliberately: supervised classification is the
signature-like component, and the Isolation Forest is the anomaly component, kept
visible as a separate signal rather than blended.

### 8. What's a zero-day, and how does your system handle one?

An exploit for a vulnerability with no available patch or signature. The
supervised model can't classify it correctly by definition — it can only output
classes it was trained on, so a novel attack gets mapped to the nearest known
label, usually BENIGN.

That's the whole reason for the Isolation Forest. It's trained on benign traffic
only, so a zero-day surfaces as a benign classification with a high anomaly
score. My risk scorer enforces a floor of 45 on any detector-flagged flow, so it
can't be dismissed even when the classifier is confident it's benign.

I'd be honest about the limit: it catches zero-days that *behave* unusually. One
that mimics normal traffic shape evades both models.

### 9. What is defence in depth, and where does this system sit?

Layered controls, so no single failure is fatal — firewall, IDS, endpoint
protection, segmentation, monitoring, access control.

This is one detection layer, and it's *detection*, not prevention: it tells you
something happened, it doesn't stop it. It would sit alongside signature IDS like
Snort or Suricata rather than replacing it, feeding a SIEM where its alerts are
correlated with host logs. Anyone deploying only this would be badly exposed.

### 10. Why does your dashboard mask IP addresses?

Because it's a demo surface that gets screenshotted and shared, and endpoint
addresses are identifying information about internal infrastructure. I truncate
to the /16 prefix, which keeps the subnet — the thing an analyst actually triages
on — while dropping the specific host.

It's the same reason my logs record request shape but never payload contents. A
security tool that leaks the data it's protecting through its own telemetry has
undermined its purpose.

---

## Part 5 — 60-second explanation

> I built a network intrusion detection system that classifies network traffic
> as benign or as one of seven attack types — DDoS, port scans, brute force,
> botnet traffic and so on.
>
> It works on network *flows*, which are statistical summaries of a conversation
> between two hosts rather than raw packets, so it detects attacks by behaviour
> instead of signatures. That means it can catch variants a signature would miss.
>
> Three models are compared — logistic regression, random forest and XGBoost —
> and the winner is chosen on macro-F1, not accuracy, because 80% of traffic is
> benign and a model that predicts "benign" for everything would score 80%
> accuracy while detecting nothing. There's also an Isolation Forest trained on
> benign traffic only, which handles the case the classifier structurally can't:
> an attack type it was never trained on.
>
> Everything sits behind a FastAPI service and a Streamlit dashboard, with SHAP
> explanations so an analyst can see why a flow was flagged. It's containerised
> and has 98 tests.
>
> One thing I'd flag up front: the metrics currently in the README come from
> synthetic data I generated, not the real CIC-IDS2017 dataset — I built the
> pipeline to run on the real thing but haven't published those numbers yet, and
> I didn't want to report results I hadn't actually produced.

---

## Part 6 — 2-minute detailed explanation

> **The problem.** Signature-based intrusion detection matches traffic against
> known-bad patterns. It's precise, but blind to anything without a signature —
> which includes every attack on its first day. Behavioural detection asks
> instead whether traffic *behaves* like an attack. A SYN flood is defined by
> its rate and directional asymmetry, not by any byte sequence, so it's learnable
> from statistical features and generalises to variants.
>
> **The data.** CIC-IDS2017 — about 2.8 million network flow records from a
> testbed where researchers ran scripted benign traffic and known attacks. Each
> flow has around 78 statistical features. I collapse the fifteen fine-grained
> labels into eight families, because several original labels have fewer than
> twenty rows and a class with eleven examples can't be learned or meaningfully
> validated.
>
> **The pipeline.** Cleaning handles the things flow data actually throws at you
> — CICFlowMeter writes literal "Infinity" for zero-duration flows, there are
> duplicate records, and some columns are constant. Then feature engineering adds
> nine derived features, each targeting a specific behaviour: packets per second
> for floods, forward-to-backward ratio for scans, TCP flag density for
> control-only traffic. Destination port gets bucketed into service categories
> rather than scaled, because port numbers are nominal — 443 isn't "more" than
> 80.
>
> **Leakage prevention** is where I spent the most care. I drop IPs, ports,
> timestamps and flow IDs, because the attacker hosts in this dataset are fixed —
> a model given the source IP learns "traffic from this address is an attack,"
> scores near-perfect, and detects nothing on any other network. That's the most
> common flaw in published NIDS results. I also fit the preprocessor on training
> rows only, so the held-out distribution never leaks in.
>
> **Model selection.** Three-way stratified split. Three models compared on
> validation macro-F1; the test set opened exactly once, at the end, for the
> winner. Something interesting came out of that: logistic regression had the
> *highest* detection rate of all three models, but at six times XGBoost's false
> alarm rate. It caught marginally more attacks by flagging far more benign
> traffic — which is exactly why I report both numbers, because an IDS that cries
> wolf gets muted, and a detector nobody looks at has a real-world recall of zero.
>
> **Serving.** The preprocessor and model are serialised into one artifact,
> deliberately, so you can't load version two of the model with version one of
> the scaler — silent train/serve skew that's very hard to debug. One inference
> class is shared by the API, the dashboard and the tests, so no two consumers
> can drift apart. The API validates every numeric field with bounds rather than
> just types, caps upload size before parsing, and returns validation errors
> without echoing the submitted value back. If no model is present it starts
> degraded and returns 503 rather than crash-looping.
>
> **What I'd fix.** The confidence score is a raw softmax output that I shouldn't
> really call a probability — it needs calibration. The risk-score weights are
> reasoned, not fitted to real triage decisions. And per-flow classification
> structurally can't detect campaigns: distinguishing DoS from DDoS needs the
> number of distinct sources, and botnet beaconing needs periodicity across
> flows. Both need temporal aggregation I haven't built.

---

## Part 7 — Resume bullet points

Pick three or four. **Only use the metrics version once you have retrained on
real CIC-IDS2017 and can defend those numbers.** Until then use the neutral
variants, which describe engineering work you genuinely did.

**Neutral (safe to use today):**

- Built an end-to-end network intrusion detection system in Python classifying
  network flows into eight attack families, combining supervised classification
  (XGBoost, Random Forest, Logistic Regression) with unsupervised Isolation
  Forest anomaly detection to cover attack types absent from training data.
- Designed a leakage-resistant ML pipeline dropping host-identifier columns and
  fitting all transformations on the training split only, addressing the target
  leakage that inflates commonly published intrusion-detection results.
- Engineered nine domain-specific flow features (packet-rate, directional
  asymmetry, TCP flag density) targeting named attack behaviours, with guarded
  denominators for the zero-duration and zero-reply flows characteristic of
  flood and scan traffic.
- Shipped the model behind a FastAPI service and Streamlit SOC dashboard sharing
  a single inference layer, with SHAP explanations, bounded and validated file
  uploads, graceful degradation when artifacts are missing, and 98 automated
  tests covering preprocessing, inference, risk scoring and API error paths.
- Containerised the system with a multi-stage Docker build running as an
  unprivileged user, orchestrating API and dashboard services via Docker Compose.

**With metrics (only after retraining on real data — substitute your own):**

- Trained and compared three classifiers on CIC-IDS2017 (~2.8M flows), selecting
  on validation macro-F1 rather than accuracy to prevent the 80%-benign majority
  class from masking undetected attacks; achieved [X] macro-F1 and [Y] attack
  detection rate at a [Z] false-alarm rate on a held-out test split opened once.

---

## Part 8 — Questions to ask them

Having good questions matters as much as good answers:

- How do you currently balance detection coverage against analyst alert fatigue?
  Is there an explicit false-positive budget?
- When a model flags something in production, what does the analyst see? How much
  explanation do they get?
- How do you know when a deployed model has gone stale, given that ground-truth
  labels in security arrive late or never?
- What does the path from a trained model to production look like here?

---

## Final reminders

1. **Lead with the synthetic-data caveat.** Volunteering a limitation builds
   more credibility than any metric.
2. **"I don't know, but here's how I'd find out" is a strong answer.** Fabricating
   one is fatal.
3. **Know your trade-offs, not just your choices.** Every decision in this
   project cost something — be ready to name what.
4. **Have the repository open.** Every claim above maps to a file you can show.
