# Submission package — International Journal of Forecasting

Everything the IJF guide for authors requires, assembled against `PAPER.md`. The manuscript body is
`PAPER.md` (5,350 words, 4 figures, 13 references); this file holds the front matter that the
journal asks for separately and that the manuscript does not currently carry.

Author name and email are filled in. **Affiliation is not** — write your institution, or
"Independent researcher" if none. A funding statement and the authorship declaration are yours to
make; I cannot assert them on anyone's behalf.

---

## Title

**Fitted degrees of freedom, not in-sample tuning, is what fails to transfer: a measured combination
curve from 465 wells**

*(Alternative, if the editor prefers the puzzle framing explicit:*
"The forecast combination puzzle as a curve: held-out gain against the number of fitted weights"*)*

## Authors

    Jungyun Han
    Independent researcher
    hjyact@gmail.com   — corresponding author

## Abstract  (142 words — IJF asks for 100–150)

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

## Highlights  (5 bullets, each ≤ 85 characters)

    Held-out gain of stacked weights degrades monotonically in the number fitted
    The same 37 candidates are worth -0.033 averaged and +0.024 with fitted weights
    One scalar weight is free to fit at 465 units; in-sample equals held-out
    Two independent candidate families give parallel curves, so it is not family-specific
    A simulation places the crossing point linearly in the number of units

## Keywords

    forecast combination; combination puzzle; stacking; cross-fitting;
    model selection bias; held-out validation

## CRediT author statement

    Jungyun Han: Conceptualization, Methodology, Software, Formal analysis, Investigation,
                  Data curation, Writing – original draft, Visualization.

## Declaration of competing interest

    The authors declare no competing financial interests or personal relationships that
    could have appeared to influence the work reported in this paper.

## Data and code availability

    The wellbore data originate from a public modelling competition and are subject to its
    terms, so they are not redistributed. All analysis code, the intermediate tables behind
    every figure, and a self-contained simulation that reproduces the central result with no
    external data are openly available at

        https://github.com/hjyact/fitted-degrees-of-freedom

    The simulation (`code/synthetic_dof.py`) depends only on numpy and scipy and regenerates
    the controlled study in full. With the wellbore frame present, `code/reproduce.py`
    rebuilds every figure and reports a digest of each table.

---

## Cover letter (draft)

Dear Editors,

We submit *Fitted degrees of freedom, not in-sample tuning, is what fails to transfer* for
consideration in the International Journal of Forecasting.

The forecast combination puzzle — that a simple average frequently outperforms estimated optimal
weights — has been documented for over fifty years, and the accepted explanation is estimation error
in the weights. Our contribution is not to rediscover this but to measure its shape. By holding the
candidate pool, the aggregation rule and the tuning location fixed and varying only the number of
fitted weights, we obtain a continuous curve with a crossing point rather than a binary comparison,
on a problem with 465 independent units and a genuinely disjoint spatial hold-out.

Three results may interest the readership. First, the optimism usually attributed to "in-sample
tuning" is negligible for a single scalar weight at this sample size; the failure is specifically
in the number of parameters, not in where they were chosen. Second, the curve is reproduced on two
candidate families that share no mechanism, so it is not a property of one pool. Third, a controlled
simulation reproduces the crossing and places it linearly in the number of units, while the real
problem's crossing arrives later than the simulation predicts — a discrepancy we report rather than
tune away.

We also include a short section on three ways we measured the wrong thing during this work, each of
which produced a plausible number. We believe this is useful to practitioners and is rarely
reported.

The work is original, is not under consideration elsewhere, and all authors approve the submission.

Sincerely,
Jungyun Han

---

## What to change in `PAPER.md` before submitting

**Done already** — the abstract is the 142-word version, the front matter is in place, the
simulation is Section 4.6, and the hold-out's provenance limitation is stated in Section 8 with the
0.1 RMSE reversal quantified. `PAPER.md` is 5,029 words with all 40 quoted numbers verified against
`stride/figs/*.csv`.

**Left to do, and only you can do it:**

(Affiliation and data availability are now filled in.)

3. **Convert to an editable file** — IJF will not take Markdown:
   `pandoc PAPER.md -o paper.docx`, then place the four figures.
4. **Rename the figures** so production can match them:
   `Fig1_motivation.png`, `Fig2_method.png`, `Fig3_dofcurve.png`, `Fig4_frontier.png`
   (sources: `stride/figs/motivation.png`, `method.png`, `dofcurve2.png`, `frontier.png`).

---

## Submission mechanics

    1. Editorial Manager for IJF, via the Guide for Authors on ScienceDirect
    2. Upload: manuscript (docx/tex), figures, highlights, cover letter, CRediT statement
    3. Suggest 3 reviewers if asked — people who work on combination weights or stacking
    4. Expect first response in roughly 2-4 months

I cannot submit on your behalf: it requires your account, your identity, and an authorship
declaration that only you can make.
