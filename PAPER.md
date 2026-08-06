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

