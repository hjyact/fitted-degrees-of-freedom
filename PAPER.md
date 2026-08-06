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

