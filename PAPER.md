# Fitted degrees of freedom, not in-sample tuning, is what fails to transfer

### A measured combination curve from 465 wells

    Jungyun Han
    Independent researcher
    hjyact@gmail.com

*Keywords: forecast combination; combination puzzle; stacking; cross-fitting; model selection bias;
held-out validation*

*Code, figures and every intermediate table: https://github.com/hjyact/fitted-degrees-of-freedom*

*Every number in this manuscript is produced by a script in that repository and stored as a
table beside the figure it feeds; `code/reproduce.py` rebuilds them all and reports a digest
for each. The controlled study in §4.6 (`code/synthetic_dof.py`) needs no external data.*

---

## Abstract

A simple average of forecasts often beats a combination whose weights are estimated from data. We
turn this binary comparison into a curve. On a spatial prediction problem with 465 independent units
and 2.3 million observations, we hold the candidate pool, the aggregation and the tuning location
fixed and vary only the number of fitted weights k, measuring held-out risk on eight geographically
disjoint blocks. Choosing a single scalar weight on the rows it is scored on costs essentially
nothing. Fitting one weight per candidate does not transfer even when cross-fitted: on two
independent candidate families the held-out gain degrades monotonically in k along parallel curves,
one crossing from helping to hurting at k ≈ 33. The same 37 candidates are worth −0.033 averaged at
one weight and +0.024 with 37 fitted weights. A controlled simulation reproduces the crossing and
places it linearly in the number of units.

---

## 1. The problem, and why the external check cannot settle it

A horizontal well is drilled through a layered formation, and what an operator needs to know is
where the bit sits *within* the layer — the true vertical thickness from a reference horizon, dTVT —
at every point along a lateral that runs for a mile or more. The measurements available are the
well's own gamma-ray log, its survey geometry, a vertical "typewell" nearby with the same log
against known depth, and whatever neighbouring laterals have already been drilled. The score is
pooled RMSE of dTVT in feet over the rows of the hidden section, across 465 wells and 2.29 million
rows; throughout, a *gain* is a change in that RMSE, so **negative is better**.

Two properties of this problem shape everything that follows.

**The metric is dominated by a per-well line** (Figure 1, right). Fitting an oracle constant and
slope per well to the held-out rows takes the ensemble's error from 5.24 ft to 2.87 — 70.0% of the
squared error lives in two numbers per well, and a constant alone accounts for 50.8%. **The error is
also concentrated** (Figure 1, left): the worst 20 of 465 wells hold 43.1% of it and the worst 40
hold 55.0%. In an earlier measurement on the same frame, within the worst 20 a per-well constant
explained 65.7% of their error against 37.7% for the other 425 — the tail is not merely larger, it
is a different kind of error. So the problem is not "predict a curve well"; it is "get one or two
numbers per well right, and do not blow up on the tail".

Both panels are oracle measurements: the per-well line is fitted on the very rows it is scored on,
so it bounds what any method could recover rather than reporting a result.

**The external check is smaller than the effects being chosen between.** The competition's public
leaderboard covers 46 wells. Resampling wells with replacement puts its sampling standard deviation
at 0.93 ft, so under a normal approximation it ranks a design that is genuinely 0.235 ft worse as
better 27% of the time. Every effect this paper adjudicates is between 0.003 and 0.09 ft. The
external check cannot see any of them, and a difference of under about half a foot on it carries no
information about which design is better.

That combination — a metric with very few effective degrees of freedom per well, and an external
signal too small to adjudicate — is what makes this a good testbed for the question the paper is
actually about: when you cannot appeal to a held-out leaderboard, how do you decide what to keep?

**Where this sits.** The phenomenon we measure is not new. In the forecasting literature it is
known as the *forecast combination puzzle*: a simple average of forecasts repeatedly beats
combinations using weights estimated from the data (Bates and Granger, 1969; Timmermann, 2006;
Smith and Wallis, 2009), and the accepted explanation is that estimating the weights costs more
variance than the optimal weights buy (Claeskens et al., 2016; Wang et al., 2023). What is new here
is not the direction of the effect but its *shape*: we hold the aggregation, the tuning location and
the candidate pool fixed and vary only the number of fitted weights k, which turns the puzzle from a
binary comparison — simple average versus estimated weights — into a curve with a measurable
crossing point (k ≈ 33 on 465 wells).

Three further connections. **Selection bias.** That optimising a selection criterion on the same
data that evaluates it inflates the result is well established, with nested cross-validation as the
standard remedy (Ambroise and McLachlan, 2002; Varma and Simon, 2006; Cawley and Talbot, 2010).
Section 4.5 measures that cost here, and Section 4.2 finds it *negligible* for a single scalar
weight, which is a caveat to the usual advice rather than a contradiction of it: the bias scales
with the freedom being exercised, and one weight is very little freedom.

**Stacking.** Our k-weight estimator is exactly Breiman's stacked regression — a non-negative
least-squares combination of candidate predictors fitted out-of-fold (Wolpert, 1992; Breiman, 1996)
— and the Super Learner literature extends it with cross-validated risk minimisation and asymptotic
optimality guarantees (van der Laan et al., 2007). Those guarantees are asymptotic in the number of
independent units. We have 465 wells, and our result is a finite-sample statement about where that
regime begins: cross-fitting by well does *not* make forty fitted weights safe here.

**Geostatistics.** One of our four likelihood channels is an ordinary-kriging interpolation of
horizon picks across the field, in the standard formulation (Matheron, 1963; Chilès and Delfiner,
2012). We use it as an input and make no methodological claim about it.

We claim novelty in none of these literatures. What we contribute is a measurement that, as far as
we know, has not been made in this form: the held-out gain as a function of the number of fitted
weights, on a real problem, with everything else held fixed so that the curve means what it appears
to mean.

---

## 2. Method, only as much as the argument needs

Each *member* is a forward-backward smoother over a two-dimensional latent state: the deviation of
the well from a reference line, and the deviation of the local dip from that line's slope. The state
evolves along measured depth with a process noise on the dip; the observation model is a sum of
independent likelihood channels, and members differ in which channels are enabled and how much each
is trusted:

* the well's own gamma ray matched against its typewell's log at candidate stratigraphic positions;
* a kriged structural surface interpolated from the training wells' horizon picks at this well's
  coordinates;
* the gamma-ray logs of *neighbouring laterals* at their known stratigraphic position, pooled within
  a radius on an absolute stratigraphic coordinate;
* a finished baseline predictor ("carrier") entered as a third emission rather than as a prior.

The answer is not a member. It is a gated move from the carrier toward a mixture of family means,

    P = B + w₀ · exp(γ · g) · (T − B),   clipped to ±c · p,

where T mixes the family means, g is a per-row gate built from the smoother's own posterior spread
and from the disagreement between two families, and p is the row's fractional distance from the
anchor, so the permitted excursion grows with distance from the last certain point. Throughout the
paper *k* denotes a number of fitted weights and never appears as a model parameter; the gate
exponent is γ.

Figure 2 shows what this amounts to on three real wells. On a typical well (left) the carrier
already has the shape and the ensemble corrects its level: 5.2 → 3.0 ft. On a tail well where the
carrier drifts the wrong way (centre) the ensemble moves it 29.5 ft on average and recovers most of
the error: 36.9 → 6.9. And on a well where the carrier was almost exactly right (right) the ensemble
moves anyway — 10.6 ft, confidently, in the wrong direction: 1.2 → 11.6.

That third panel is the whole wager. A member is not an independent prediction; it is a *move* away
from a baseline, and what an ensemble weight buys is the size of that move. The design earns −2.9 ft
on average and pays for it on wells where the baseline needed no help. Section 4 counts degrees of
freedom over exactly this quantity.

Nothing in this section is offered as novel. It exists so that "a member", "a family" and "a weight"
are concrete objects when Section 4 counts degrees of freedom over them.

---

## 3. The protocol

Every adoption decision in this paper is made the same way.

**Regions, not folds.** Wells are partitioned into eight held-out blocks: quartiles of the formation
isopach, and blocks of the X/Y plane. Random folds would leave a well's immediate neighbours in the
training set, and several of our channels read neighbours directly, so a random split measures
interpolation and not transfer.

**Both sides tuned outside.** For a comparison of designs A and B, *both* are tuned on the wells
outside the held-out block, on a fixed grid, and read on it. Comparing "B chosen elsewhere" against
"the best possible A here" charges B for a cost both designs pay.

**A fixed weight, not a re-tuned one.** The weight of an added term is a single value used in every
block. Re-choosing it per block loses, measured three times (shape −0.040 tuned against −0.065
fixed; MV +0.0087 against −0.0202).† The per-block choice is itself a noisy selection — Section 4's
subject in miniature.

**Scoring.** The mean change in pooled RMSE across the eight blocks, and the count of blocks that
improve. Under a sign test, 7/8 is p = 0.035, 6/8 is p = 0.145 and 5/8 is chance. We report the
count alongside the mean throughout, because the mean can be carried by one block.

**Balance the blocks, and print their sizes.** Cutting X and Y independently on correlated
coordinates gave blocks of 195/35/35/195 wells; two of the eight votes then came from 35-well blocks
whose sampling noise exceeds any effect being measured. Splitting Y within each X half gives
120/115/115/115 and moved one measured gain from −0.0436 to −0.0685 — a change of conclusion caused
entirely by block construction. Our driver prints block sizes on every run.

---

## 4. The gain collapses with fitted degrees of freedom

### 4.1 Setup

Let A be a baseline design and let a *family* be a set of k candidate members that share a
mechanism. We compare two ways of adding the family to A:

* **one weight** — average the family, add the average as a single term, choose its weight q on a
  fixed grid;
* **k weights** — regress the residual of A on the k members individually under a non-negativity
  constraint, cross-fitted by well so that no well's weights are fitted on that well.

Both are then read the same way: eight held-out blocks (isopach quartiles and XY blocks, sizes
115/115/115/115 and 120/115/115/115), everything tuned outside the block, scored by the mean change
in pooled RMSE and by how many of the eight blocks improve.

**Two reference points appear in this paper and must not be confused.** The *shipped ensemble*
(5.2363 ft over a carrier baseline at 8.1089) is the finished design; Figures 1, 2 and 4 describe
it. The *comparison baseline A* (5.3109 ft) is that design with the family under test removed —
strong and lateral members only, one tuned mixture weight — so that adding the family is a change of
exactly one term. Every number in Table 1 and Figure 3 is a change relative to A, not to the shipped
ensemble. **Sign convention: a negative gain is an improvement**, since the score is an RMSE.

The tuning grid is fixed throughout: lateral mixture ∈ {0.10, 0.15, 0.20, 0.25}, move weight
w₀ ∈ {0.75, 0.85, 0.95}, gate exponent γ ∈ {0, −0.05, −0.10}, and the added term's weight
∈ {0, 0.05, 0.075, 0.10, 0.15}. Both designs search the same grid; only the rows they search on
differ.

### 4.2 One weight is free to fit

Table 1 gives, for four axes, the gain when the weight is chosen on the block being scored
("in-sample") and when it is chosen outside it ("honest"). The two columns average the same eight
per-block reads; the only difference is where the weight was chosen.

| axis | k | in-sample (1 dof) | honest (1 dof) | wins | k-dof mixture, cross-fitted |
|---|---|---|---|---|---|
| shape channel | 37 | −0.0190 | −0.0332 | 6/8 | **+0.0250**, 4/8 |
| MV family (4) | 4 | −0.0451 | −0.0588 | 6/8 | −0.0057, 6/8 |
| MV family (all) | 50 | −0.0855 | −0.0762 | 6/8 | **+0.0035**, 3/8 |
| tail constant correction | 1 | −0.0356 | −0.0383 | 5/8 | — |

The last column is the same family entering through one weight *per member* instead of one weight
for the family mean, fitted by non-negative least squares and cross-fitted by well. At k = 37 and
k = 50 it is worse than doing nothing, which is the endpoint Section 4.3 fills in continuously.

The ratios between the first two columns are 0.6, 0.8, 1.1 and 0.9. At this sample size a single
scalar is not enough freedom to overfit measurably, and the widely repeated warning that in-sample
tuning inflates a gain several fold does not describe this regime. (We had inherited exactly that warning as a 3–9× rule from our
own earlier work; it did not reproduce, and we do not claim it.)

The tail correction is worth reading as the control: 5/8 is chance, and it is the one axis we
adopted and later abandoned.

### 4.3 k weights do not transfer, on either family

Figure 3 sweeps k for two families — 50 members built to hold their move against the baseline, and
37 built around a shape channel — drawing k members at random, 100 draws per k, fitting
non-negative weights cross-fitted by well, and reading on the same eight blocks.

|  k | MV | shape |  | k | MV | shape |
|---|---|---|---|---|---|---|
| 1 | −0.0263 | +0.0016 | | 20 | −0.0160 | +0.0140 |
| 2 | −0.0271 | +0.0013 | | 24 | −0.0083 | +0.0168 |
| 4 | −0.0304 | +0.0033 | | 28 | −0.0041 | +0.0196 |
| 6 | −0.0336 | +0.0047 | | 32 | −0.0032 | +0.0229 |
| 8 | −0.0280 | +0.0060 | | 34 | +0.0033 | +0.0237 |
| 12 | −0.0264 | +0.0086 | | 36 | +0.0029 | +0.0244 |
| 16 | −0.0221 | +0.0118 | | 40 | +0.0061 | — |

Both families degrade monotonically in k and the curves are parallel. They differ in intercept, not
in slope: MV carries information the baseline lacks and helps until the fitting noise overtakes it
at k ≈ 33; the shape family carries none that survives this baseline and is harmful from k = 1. The
win rate follows (MV 5.30 → 3.55 of eight).

That the two curves have the same slope is the point. Degradation with k is not a property of a
particular family or of what it encodes; it is a property of fitting weights at this sample size.

### 4.4 The combination rule is worth more than the channel

The same 37 shape members appear in both experiments, and the two treatments disagree by as much as
the project's largest modelling gain:

    averaged, one fixed weight        −0.0332      (0 fitted weights inside the family)
    37 cross-fitted NNLS weights      +0.0244      (37 fitted weights)

Same members, same rows, same blocks. The 0.058 ft between them is entirely the combination rule —
for comparison, the largest modelling gain in Table 1 is the full MV family at −0.076, and the shape
channel's own contribution is −0.033. Choosing how to combine a fixed set of candidates is therefore
a decision of the same size as choosing what to model, and it is usually made without measurement.
Averaging is not what you do when you cannot afford to fit; here it is the better estimator, and the
family is useful or useless depending only on which rule you use.

Two cautions on reading the table. First, "k = 1" means different things in the two experiments — in
Table 1 it is the family *mean* at one weight, in Figure 3 it is one randomly drawn *member* at one
weight — and the gap between them (−0.0332 against +0.0016) is itself the averaging effect. Second,
individual points in Figure 3 are not separately significant; the draw-to-draw standard deviation is
0.02–0.03 against effects of 0.03, so the standard error at 100 draws is about 0.003. The monotone
shape, replicated on two families, is the evidence.

### 4.5 Discrete selection is the same curve

Choosing which members to keep is the k → large end of this curve with the weights restricted to
{0, 1}. Measured nested on the same frame, keeping the best three members costs +0.0767 and keeping
the best one +0.1840 against keeping all of them at a fixed weight.† Batch-level selection is
subtler and more instructive: choosing which batches to drop using only the other blocks reads
+0.0066 at 4/8, while the same comparison with the choice made once on all blocks read 8/8.† The 8/8
was not evidence of a robust finding; it was the overfitting signature.

"Ship everything at a fixed weight" is therefore not a heuristic to fall back on. It is where this
curve has its optimum.

### 4.6 The same curve under control, and where the real one sits

One field is not evidence that anything general is happening, so we reproduced the experiment where
n and k are set rather than observed (`stride/synthetic_dof.py`). Units carry a shared error
component that a combination can explain, plus independent noise it cannot; k candidates are drawn,
combined either by an equal-weight mean carried at one fitted scalar or by NNLS weights fitted
out-of-fold, and read on held-out units. Everything else matches Section 4.1.

**The first version had no crossing at all**, and the reason is worth stating because it identifies
the mechanism. With *exchangeable* candidates the equal-weight mean is already the optimal
weighting: there is nothing for fitting to find, Δ = 0, and k weights can only add variance. The
measured curve was positive at every k and every n. Introducing heterogeneity in candidate quality —
so that the optimal weights are unequal and Δ > 0 — produces the crossing immediately. Fitting is
worth doing only to the extent that the candidates are *not* interchangeable, and it is worth doing
for fewer of them than intuition suggests.

With heterogeneous candidates the crossing point moves with the number of units:

    n         60     125     250     500    1000
    k*       3.5     7.4    12.5    16.6    22.8
    k*/n   0.059   0.060   0.050   0.033   0.023

    least squares:  k* = 0.019 n + 5.2      (corr 0.956)

k* grows with n, close to linearly, which is what a variance term of order k/n predicts. It is not
exactly linear: k*/n falls by a factor of 2.6 across the range, so the simple account is an
approximation. We attribute the shortfall to the non-negativity constraint, which makes the
*effective* number of fitted parameters smaller than k and increasingly so as k grows — but we have
not verified that, and we report the deviation rather than fit a second parameter to it.

**Where the real problem sits.** Extrapolating the simulation to n = 465 units predicts k* ≈ 14. The
wellbore frame crosses at k ≈ 33, more than twice as late. The direction of the discrepancy is
interpretable: our candidates are more heterogeneous than the simulation's, so the optimal weighting
is further from equal and fitting keeps paying for longer. But we have not calibrated the
simulation's heterogeneity to the measured pool, so this remains a statement about a discrepancy we
observed and not a fit we achieved. What the simulation does establish is that the curve, the
crossing, and its growth with n are properties of combining k estimates from n units, not of
wellbores.

---

## 5. Why members stop helping: a single frontier

A member m helps the ensemble S only if its error is decorrelated enough to pay for being less
accurate. Writing e for errors, the break-even condition is

    corr(e_m, e_S) = σ_S / σ_m,

which is exactly cov(e_m, e_S) = var(e_S) — that is, the least-squares slope of e_m on e_S equals 1.
A member sitting on that line is one whose error *is* the ensemble's error plus independent noise.

Figure 4 places all 671 members we ever built on this plane. The median ratio of achieved
correlation to break-even correlation is 1.013, with a minimum of 0.808, across mechanisms that
share no evidence — different reference logs, different neighbour definitions, different dip priors,
different baselines. Members do not scatter around the line; they lie on it.

Two consequences.

**Tuning a member for accuracy moves it along the frontier and buys nothing.** This is why a project
can spend weeks improving members and see the ensemble stand still, and it is consistent with an
earlier observation of ours that a *more accurate* kriged surface blends *worse*, with the blend
optimum at a surface RMS of 19–21 ft while a neural field at 14.77 loses.

**Distance below the line is a screening statistic.** Of the three named families and the unlabelled
remainder, only MV sits clearly below the line (median ratio 0.920, against 0.995 for strong, 1.006
for lateral and 1.018 for the other 486 members), and MV is the only family that measurably paid.
That ranking is obtained without scoring anything on held-out data, so a candidate family can be
triaged before any of the validation budget in Section 3 is spent on it.

---

## 6. Negative results

These are not caveats; they are the argument that the combination rule, not the ingredient list, is
where the remaining error lives.

* **Robust aggregation loses.** Median, trimmed mean, Huber and isotonic combinations all read worse
  than non-negative least squares, and the fitted global scale of the NNLS solution is 0.993 — there
  is no shrinkage left to find.†
* **Per-member clipping before the mean loses at every threshold**, whether the clip is on the
  deviation from the baseline (10/20/30/50 ft) or on the distance from the family median (3/6/10/20
  ft). Members that fail on a well fail *together*, so no order statistic sees an outlier.
* **The post-hoc tail correction is dead.** Worth −0.109 in sample; −0.0122 at 5/8 held out.
* **Residual boosting is dead.** A gradient-boosted model on the blend's residual reads +0.354
  in-fold and −0.011 out of fold.†
* **Per-well weights are unpredictable.** On the frame this was measured, an oracle that picks the
  best ensemble weight per well reached 4.32 ft against 5.24 achieved, yet every observable we tried
  predicted those weights at leave-one-fold-out correlation ≈ 0.†

The last one is the sharpest: the information is *there* — the oracle proves it — and it is not
recoverable from anything measurable. That is the same wall Section 4 measures from the other side.

† **Provenance.** Results marked † were measured during the project rather than by the single driver
that produces Table 1 and Figure 3, on the same 465-well frame and the same eight blocks but at
earlier stages of the design, so their baselines differ by a few hundredths of a foot from the A
defined in §4.1. We report them because they are the evidence that motivated the experiments in
§4, and we mark them because they are not re-derivable from `stride.reproduce`. Everything unmarked
is.

---

## 7. Three ways we measured the wrong thing

Each produced a plausible number and survived review; each was caught because the output was
suspiciously *clean*, not suspiciously wrong.

**An unset environment variable.** Verifying that recovered build recipes reproduced their stored
outputs, we omitted the variable naming the baseline column. The baseline attached to 6 wells of 40
instead of 40, the rebuilt tracks differed from the originals, and we constructed three successive
causal explanations — a stale cache, engine-version drift, bistability of the smoother — before
running the identical-conditions control. With the variable set, the rebuild matches on 38 of 40
wells with a per-well RMS median of 0.00003 ft. The control should have been the first experiment,
not the fourth.

**A mixed aggregation.** The first version of our Claim 1 driver compared a *pooled* in-sample
number against a *mean of per-block* held-out numbers. The two aggregations differ by more than the
effect, and the bias appeared to vanish. A comparison is only interpretable when everything except
the quantity under test is held fixed — which is the paper's own thesis, violated by the paper's own
first experiment.

**A sign.** In the sufficient-statistics formulation of Section 4, passing +Dᵀr instead of −Dᵀr to a
non-negative solver drives every weight to zero. The sweep then reported a gain of exactly 0.0000 at
every k, which reads as a clean null result. A result that is *too* clean is a bug report.

Related, and older: **a cache key must carry every input.** A neighbour-profile cache computed on a
104-well run was silently reused on a 465-well frame, so that channel fired on 22% of the wells for
weeks. The fix was to put the well count in the key of every new cache family.

---

## 8. Limits

Everything here is one design on one field. The members are variants of a single smoother, so two
families are not two independent designs, and the eight blocks are geological and geographical
partitions of that field — "held out" means held out within it. Nothing here tests transfer to
another basin.

The draw-to-draw spread in Figure 3 is real: a standard deviation of 0.02–0.03 against effects of
0.03. At 100 draws the standard error is about 0.003, small enough that the monotone shape is
resolved, but no single point on the curve is separately significant. Two independent
implementations agree to four decimals at k = 50, which is what makes the algebra trustworthy.

The protocol's own resolution is the honest floor. With eight blocks, a sign test cannot separate
6/8 (p = 0.145) from noise on its own. That is why the argument rests on a monotone curve across 18
values of k on two families, and not on any single adoption decision.

**Our blocks are disjoint in space but not in provenance,** and this deserves to be stated plainly.
Nothing is fitted on a held-out block, but the candidate pool itself was assembled over the project
while all 465 units were visible. We can bound how much that matters, because two configurations of
the deployed design were later evaluated against units held back entirely — 46 and 154 of them,
never seen during development. The 465-unit frame ranked the larger configuration first by 0.103;
both external evaluations ranked it *second*, by 0.075 and 0.098 respectively. A ranking from our
frame can therefore be wrong by about 0.1 in RMSE, which is larger than several of the effects in
Table 1.

This does not touch the k-curve, which compares combination *rules* over one fixed pool and never
uses the frame to choose members. It does bound what any single adoption decision made on this frame
is worth, and it is a further instance of the paper's own thesis: the pool was curated with those
units visible, which is itself a form of fitting whose degrees of freedom nobody counted.

---

## 9. Conclusion

The practice this paper argues against is not a mistake anyone makes carelessly. Adding candidate
members and then fitting weights over them is the obvious way to use a pool of estimators, and on a
large enough sample it is the right one. That simple averages often beat estimated weights has been
known in forecasting for fifty years. What we measured is *where* the crossover sits when everything
except the number of fitted weights is held fixed, and it is earlier than intuition suggests: on 465 wells, past about thirty fitted weights a non-negative
mixture — cross-fitted by well — is *worse* than carrying the same members' average at a single
fixed weight, and the same thirty-seven members are worth −0.033 ft one way and +0.024 ft the other.

Three practical rules follow, and they cost nothing to adopt.

**Count the weights you are fitting, not the members you are adding.** A family of fifty members
entering through one weight is a one-parameter change. The same fifty entering through fifty weights
is not, and the difference is larger than most of the modelling decisions it will be compared
against.

**Screen candidates on the frontier before spending validation budget.** A member's error
correlation against the current ensemble, read against σ_S/σ_m, ranks families correctly without any
held-out scoring. In our case it identified the one family that paid, out of four, from geometry
alone.

**Make the control the first experiment, not the fourth.** Every one of the three measurement
failures in §7 produced a plausible number, and each was found only because something in the output
was suspiciously clean — a null that was exactly zero, an effect that vanished, a divergence that
had no mechanism. An identical-conditions run costs minutes and would have caught all three.

None of this makes the underlying problem easier. The oracle bounds in §1 and §6 say that most of
the remaining error is a per-well constant and slope that we can prove exists and cannot predict.
What the paper offers is the discipline for finding out, quickly and honestly, that a proposed fix
does not recover it — which on this problem was the difference between a year of measurable progress
and a year of fitted noise.

---

---

