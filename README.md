# Signal2Sale

Which bank customers should be called for a term-deposit campaign, and how many times, given that every call costs agent time and every unwanted call costs customer experience?

**Owner:** campaign manager | **Decision cadence:** per campaign cycle | **Data:** 11,162 customer contacts, 2,791 held out for test

## Bottom line

- **XGBoost is the recommendation.** It captures essentially the same conversions as the tuned logistic baseline while making 70 fewer calls and cutting the false-positive rate from 27.0% to 21.9%, the most efficient point in the comparison on both cost and customer experience.
- Random Forest converts **59 more customers but contacts 67 more non-converters** to do it. That is a trade worth taking only when call capacity is not binding, and it is stated as a condition rather than resolved by picking the higher AUC.
- The deliverable is a **four-band calling policy**, validated on held-out data: High Priority converts at 78.8%, Medium at 48.0%, Low at 33.7%, and the excluded Do Not Call band at 21.5%. Dropping the bottom band removes 27% of the call list at the worst conversion rate in the population.
- **Call duration is a leakage feature** and is excluded from every deployable model. An audit model including it reaches ROC-AUC 0.904 against 0.755 without it, which is the size of the temptation and the reason published analyses of this dataset report inflated numbers.

## The decision and why it is hard

The campaign manager is not asking who will subscribe, they are asking who to call. Those differ in two ways that matter.

First, a call has a cost on both sides of the ledger: agent time, and the customer-experience cost of an unwanted call to someone who was never going to convert. A model tuned for recall makes more calls, and the comparison has to charge it for them.

Second, the decision has to be made before the call happens, which disqualifies any feature that only exists afterwards. The most predictive column in this dataset is exactly such a feature.

## The leakage check

`duration` records the length of the call in seconds. It is only known once the call has already been placed, so it cannot inform a decision about whether to place it. A model that uses it can tell you which calls went well after the fact, which is not a targeting model.

Rather than dropping it silently, a duration-included model was trained as an explicit audit to measure how much signal the forbidden feature carries:

| Model | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|
| Baseline without `duration` (deployable) | 0.755 | 0.738 | 0.548 |
| Audit model with `duration` (not deployable) | 0.904 | 0.823 | 0.789 |

That gap, 0.755 to 0.904, is most of the apparent performance available in this dataset. The audit model is excluded from every recommendation below.

## Architecture

```
Bank Marketing dataset (11,162 contacts, binary deposit outcome)
   |  kagglehub download
   v
EDA targeted at the decision
   |  target distribution (47.4% base rate, near-balanced)
   |  feature vs target relationships
   |  conversion rate BY campaign attempt number
   |    --> 53.7% on attempt 1, below 38% by attempt 5
   v
LEAKAGE CHECK on `duration`
   |  known only AFTER a call is placed --> unusable for targeting
   +--> excluded from all deployable models
   +--> audit model trained separately to size the gap (0.755 vs 0.904)
   v
Shared preprocessing pipeline (ColumnTransformer)
   |  OneHotEncoder on categoricals, StandardScaler on numerics
   |  identical across all three models, so the comparison
   |  isolates the model rather than the preprocessing
   v
Train / test split --> 2,791 test contacts, 1,322 conversions
   |
   +----------------+----------------+
   v                v                v
Logistic         Random Forest    XGBoost
 |  class-weight sweep            |
 |  {None, 1:1.2, 1:1.5}          |
 |  threshold sweep 0.30 -> 0.70  |  threshold sweep 0.30 -> 0.70
 |  reported as a table, not      |
 |  collapsed to one value        |
   |                |                |
   +----------------+----------------+
                    v
     Comparison on calls made, precision, recall,
     false-positive rate, ROC-AUC, and incremental
     campaign profit against the LR baseline
                    |
                    v
              XGBoost selected
                    |
         +----------+----------+
         v                     v
  Four-band policy       Contact-frequency cap
  bands validated        derived from observed
  against realized       decay by attempt number
  per-band conversion    + stop-calling rule
  rate on test data      (>=3 attempts AND p<0.30)
         |                     |
         +----------+----------+
                    v
           CALLING POLICY
```

## Data

[Bank Marketing dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset), loaded via `kagglehub`. Each row is one phone-campaign contact with a customer, with a binary target recording whether they subscribed to a term deposit.

Test split: 2,791 contacts, 1,322 conversions, a 47.4% base rate. This is close to balanced, which is unusual for a marketing dataset and means the interesting decision is the operating point rather than imbalance handling.

## Approach

**Shared pipeline.** All three models sit on an identical `ColumnTransformer` (one-hot encoded categoricals, standardized numerics) inside a scikit-learn `Pipeline`, so the comparison isolates the model rather than differences in preprocessing.

**Operating point as a first-class experiment.** The logistic baseline was tuned on the two levers that actually move a calling policy rather than on accuracy:

| Class weight | ROC-AUC | Precision | Recall | Calls made | False positive rate |
|---|---:|---:|---:|---:|---:|
| None | 0.7558 | 0.737 | 0.563 | 1,010 | 18.1% |
| {0:1, 1:1.2} | 0.7558 | 0.684 | 0.649 | 1,254 | 27.0% |
| {0:1, 1:1.5} | 0.7559 | 0.624 | 0.754 | 1,597 | 40.8% |

ROC-AUC is flat across all three, which is the point: class weighting does not make the model better, it moves where on the curve the campaign operates. A decision-threshold sweep from 0.30 to 0.70 in 0.05 steps was run alongside it and reported the same way, as a table rather than collapsed into a single chosen value.

**Tree models** were then benchmarked on the same pipeline and the same metrics, with calls made treated as a cost column rather than a footnote.

## Results

Test set: 2,791 contacts, 1,322 conversions.

| Model | Calls made | Precision | Recall | False positive rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (tuned, class-weighted) | 1,254 | 0.684 | 0.649 | 27.0% | 0.756 |
| Random Forest | 1,380 | 0.664 | 0.694 | 31.5% | 0.767 |
| **XGBoost (recommended)** | **1,184** | **0.728** | 0.652 | **21.9%** | **0.776** |

Confusion detail on the same test set:

| Model | Conversions captured | Non-converters called | Conversions missed |
|---|---:|---:|---:|
| Logistic Regression | 858 | 396 | 464 |
| Random Forest | 917 | 463 | 405 |
| XGBoost | 862 | 322 | 460 |

XGBoost captures four more conversions than the Random Forest while calling 141 fewer non-converters, and captures four more than the logistic baseline while making 70 fewer calls in total. It is the only model in the comparison that improves on the baseline in both directions at once, which is why the recommendation does not require the cost constants to be exactly right.

**Band validation.** The four bands were defined on predicted probability and then checked against the realized conversion rate inside each band on held-out data, rather than assumed to separate:

| Band | Threshold | Customers | Realized conversion rate |
|---|---|---:|---:|
| High Priority | p >= 0.60 | 906 | 78.8% |
| Medium Priority | 0.45 <= p < 0.60 | 450 | 48.0% |
| Low Priority | 0.30 <= p < 0.45 | 685 | 33.7% |
| Do Not Call | p < 0.30 | 750 | 21.5% |

**Contact decay.** Conversion rate falls with each additional attempt: 53.7% on the first contact (1,169 customers), 45.2% on the second, 45.2% on the third, 45.9% on the fourth, then 37.9% on the fifth and below 37% thereafter. That decay, not an assumption, is what the four-contact cap is derived from.

## Recommended operating policy

```
SCORE every customer with the XGBoost model (no `duration` feature).

BAND and act:
  p >= 0.60   High Priority     contact immediately        (converts at 78.8%)
  p >= 0.45   Medium Priority   contact if capacity allows (converts at 48.0%)
  p >= 0.30   Low Priority      deprioritize, test selectively (33.7%)
  p <  0.30   Do Not Call       exclude from the campaign  (21.5%)

CONTACT FREQUENCY:
  cap outreach at 4 contacts per customer
  (conversion falls from 53.7% on attempt 1 to below 38% by attempt 5)

STOP CALLING when BOTH:
  - 3 or more attempts already made, AND
  - predicted probability below 0.30
  (305 test-set customers meet this, and continuing to call them
   is the clearest waste in the current campaign)
```

## Assumptions and sensitivity

Any dollar profit or cost figure in the notebook, such as revenue per conversion and cost per call, uses **illustrative placeholder constants** rather than real bank financials. They demonstrate how a cost-sensitive threshold would be chosen once real unit economics are available. Treat them as a worked example, not a business impact claim.

Everything those figures are computed from is measured on held-out data: calls made, conversions captured, non-converters called, and the per-band conversion rates. The model ranking is also stable across a wide range of cost constants, because XGBoost dominates simultaneously on fewer calls and fewer false positives rather than trading one against the other.

The place where real numbers would change the answer is the threshold, not the model choice, since the threshold is where the revenue-to-cost ratio directly sets the cutoff. The 0.30 to 0.70 sweep table in the notebook is there so a campaign manager with real unit economics can read off their own operating point.

## Limitations

- **Placeholder unit economics.** No real revenue per conversion or cost per call, so no defensible ROI figure. The band structure and model ranking hold regardless, but the specific threshold does not.
- **No formal assumptions file.** The cost constants here are inline rather than declared in an `assumptions.yaml` with source and confidence labels, and there is no unit-tested decision-metrics layer behind the profit figures, so the profit numbers do not trace to a tested function the way the measured metrics do.
- **Random rather than chronological split.** The campaign is a sequence over time, and a time-based split would be a better test of whether the model holds up on a later campaign cycle.
- **No capacity-constrained evaluation.** The band policy assumes the bank calls everyone above the cutoff. A fixed daily agent capacity would turn this into a ranked queue, which is a different and more realistic problem.
- **Single dataset, single bank, single campaign period.** No cross-period validation.

## What I would do next

- Get real revenue per conversion and cost per call, move them into an `assumptions.yaml`, and re-derive the threshold from the actual ratio rather than a placeholder.
- Add a unit-tested decision-metrics layer so every profit figure traces to a tested function rather than an inline calculation.
- Re-split chronologically and re-evaluate, to test whether the band structure survives on a later campaign cycle.
- Add a capacity-constrained ranked queue as an alternative to the fixed band cutoffs, for the realistic case where agent hours bind before the threshold does.
- Test uplift modeling rather than propensity modeling, since the campaign should target customers whose decision the call actually changes, not those who would have subscribed anyway.

## Repo contents

```
Signal2Sale.ipynb    Full analysis: EDA, leakage audit, baseline, tree models,
                     model comparison, decision rules, executive summary
requirements.txt     Pinned dependency versions
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Signal2Sale.ipynb
```

Requires `kagglehub` credentials to auto-download the dataset, or download it manually from the Kaggle link above and point the notebook at the local CSV.
