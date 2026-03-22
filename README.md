# Satellite Collision Avoidance — Full Project Report

---

## 1. The Real-World Problem

### What is Space Debris?

Every satellite, rocket stage, or fragment left in orbit becomes **space debris**. As of today:
- There are over **27,000 tracked objects** in orbit
- Millions of smaller untracked fragments also exist
- Everything moves at **7–8 km/s** (~28,000 km/h)

At that speed, even a 1 cm fragment hitting a satellite has the energy of a hand grenade.

### The Cascade Risk — Kessler Syndrome

If collisions happen, they create more debris → which causes more collisions → which creates even more debris. This chain reaction is called **Kessler Syndrome** and could make entire orbital zones unusable for centuries.

### ESA's Challenge

The **European Space Agency (ESA)** runs a Space Debris Office that monitors all tracked objects 24/7. Every week:
- Hundreds of **close approach alerts** are issued
- After filtering, ~2 actionable alerts per satellite per week reach human analysts
- ESA performs **more than 1 collision avoidance manoeuvre per satellite per year**

The challenge: **Can a machine learning model predict collision risk as accurately as human analysts — and help decide when to manoeuvre?**

---

## 2. Conjunction Data Messages (CDMs)

### What is a CDM?

A **Conjunction Data Message** is a formal report issued when two space objects are predicted to come dangerously close. Think of it as an "alert ticket" that contains:

| Field Category | Examples |
|---|---|
| Orbital state | Position (r, t, n), velocity vectors |
| Miss distance | How close the objects will get at TCA |
| Relative speed | How fast they approach each other |
| Uncertainty | Covariance matrices — how confident we are in the prediction |
| Space weather | Solar activity affecting atmospheric drag |
| Object properties | Mass, surface area, drag coefficient |

### What is TCA?

**TCA = Time of Closest Approach** — the exact predicted moment when the two objects will be nearest to each other.

`time_to_tca` in the dataset = number of days remaining until TCA when the CDM was issued.

### CDM Sequence

Each collision **event** generates multiple CDMs over time as the orbit predictions get refined:

```
Event starts (TCA is 7 days away)
    CDM #1 issued — risk = LOW      (big uncertainty, far away)
    CDM #2 issued — risk = MEDIUM   (getting closer, better data)
    CDM #3 issued — risk = HIGH     (1 day to TCA, very precise)
         ↓
    Decision: Manoeuvre or not?
```

The training dataset has ~12 CDMs per event on average. Each row = one CDM snapshot.

---

## 3. The Dataset

**Source:** https://kelvins.esa.int/collision-avoidance-challenge/data/

| | Training | Testing |
|---|---|---|
| Rows | 162,634 | 24,484 |
| Unique events | 13,154 | 2,167 |
| CDMs per event (avg) | ~12 | ~11 |
| Columns | 103 + 1 categorical | Same |

### Important Differences: Train vs Test

| Aspect | Training Data | Testing Data |
|---|---|---|
| TCA window | All CDMs included | Only events where latest CDM is within 1 day of TCA |
| Near-TCA data | Included (within 2 days) | Excluded (this is the "future" we must predict) |
| Sampling | Over-represents high-risk events | Same |

This means the test set is harder — we only see data from early CDMs, and must predict the final risk at TCA.

### The 103 Columns — Categories

| Prefix | Meaning | Example columns |
|---|---|---|
| *(none)* | Event-level info | `event_id`, `time_to_tca`, `miss_distance`, `relative_speed`, `risk` |
| `t_` | **Target** (ESA satellite) | `t_sigma_r`, `t_sigma_t`, `t_sigma_n`, `t_mass`, `t_cd_area_over_mass` |
| `c_` | **Chaser** (debris/other object) | `c_sigma_r`, `c_sigma_t`, `c_sigma_n`, `c_mass`, `c_object_type` |

**Key individual features:**

| Feature | What it means |
|---|---|
| `miss_distance` | Predicted closest distance between objects (metres) |
| `relative_speed` | Relative velocity at TCA (m/s) |
| `time_to_tca` | Days until closest approach |
| `risk` | **TARGET** — collision probability (log scale) |
| `t_sigma_r/t/n` | Position uncertainty of the ESA satellite in Radial/Tangential/Normal directions |
| `c_sigma_r/t/n` | Position uncertainty of the debris object |
| `mahal_dist` | Mahalanobis distance — statistical measure of separation considering uncertainty |

---

## 4. The Risk Score

`risk` is computed on a **base-10 logarithmic scale** of collision probability.

```
risk = log10(collision_probability)

Example:
  risk = -4  →  probability = 0.0001  (1 in 10,000)
  risk = -2  →  probability = 0.01    (1 in 100)   ← considered HIGH
  risk = 0   →  probability = 1.0     (certain collision)
```

Most events have very low or zero risk. High-risk events (risk > 0.001 in our code) are rare but critical.

---

## 5. Our Code — What It Does Step by Step

### Step 1: Install & Import Libraries

```python
pandas, numpy           → data loading and manipulation
scikit-learn            → machine learning model + scaling + evaluation
pyswarm                 → Particle Swarm Optimization
```

### Step 2: Load the ESA Dataset

```python
train = pd.read_csv("train_data.csv")   # 162,634 rows for training
test  = pd.read_csv("test_data.csv")    # 24,484 rows for prediction
```

We check the shape, columns, and distribution of the `risk` target variable.

### Step 3: Feature Selection

```python
exclude_cols = ["risk", "event_id", "mission_id", "c_object_type"]
features = all numeric columns except excluded ones
```

- `risk` is excluded because it's what we're predicting
- `event_id`, `mission_id` are identifiers, not physics
- `c_object_type` is text (categorical), needs special encoding — excluded for simplicity
- All remaining ~99 numeric columns are used as input features

### Step 4: Train/Validation Split

```python
X_train (80%) → used to train the model
X_val   (20%) → used to test how well the model generalises
X_test        → the actual ESA test set we predict on
```

This split prevents the model from simply "memorising" the training data.

### Step 5: Feature Scaling (StandardScaler)

```python
X_scaled = (X - mean) / std_deviation
```

Why? Features have very different scales:
- `miss_distance` might be in thousands of metres
- `time_to_tca` is in days (0 to ~7)
- `t_sigma_r` might be in small decimals

Without scaling, large-value features dominate the model unfairly. Scaling puts everything on equal footing.

**Important:** We only fit the scaler on training data, then apply the same transformation to validation and test data. Otherwise, information from the test set would "leak" into the model.

### Step 6: Train Random Forest Regressor

```python
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X_train_scaled, y_train)
```

**What is a Random Forest?**

A Random Forest builds 200 independent Decision Trees, each trained on a slightly different random subset of data and features. To make a prediction, all 200 trees vote, and the average is the final answer.

```
          Training Data
               ↓
    ┌──────────────────────┐
    │  200 Decision Trees  │
    │  (each slightly      │
    │   different)         │
    └──────────────────────┘
               ↓
    Average of all 200 predictions
               ↓
          Risk Score
```

**Why Random Forest for this problem?**
- The relationship between orbital parameters and risk is highly non-linear
- Random Forest handles non-linearity naturally
- Robust to outliers (space data has measurement errors)
- No assumptions about data distribution
- Works well even with many features (99 columns)

### Step 7: Evaluate on Validation Set

```python
MAE (Mean Absolute Error) = average |predicted - actual|
R²  (R-squared)           = how much variance is explained (1.0 = perfect)
```

This tells us if the model is learning real patterns or just memorising.

### Step 8: Predict Risk on Test Set

```python
predicted_risk = model.predict(X_test_scaled)
```

For each of the 24,484 CDMs in the test set, the model outputs a predicted collision risk score. We then look at the top 10 highest-risk events.

### Step 9: Particle Swarm Optimization (PSO)

This is where the collision **avoidance** part begins.

**What is PSO?**

PSO mimics how birds flock or fish school. A group of "particles" (candidate solutions) explores the solution space simultaneously. Each particle:
1. Has a current position (a manoeuvre vector `[delta_r, delta_t, delta_n]`)
2. Remembers its personal best position
3. Is attracted toward the swarm's global best position
4. Updates its position each iteration

```
Iteration 1:  30 particles scattered randomly
Iteration 2:  Particles move toward promising areas
...
Iteration 50: Particles converge on minimum-risk manoeuvre
```

**What are `delta_r`, `delta_t`, `delta_n`?**

These represent a small adjustment to a satellite's position in three orbital directions:

| Symbol | Direction | Meaning |
|---|---|---|
| `delta_r` | Radial (R) | Toward/away from Earth's centre |
| `delta_t` | Tangential (T) | Along the direction of orbital motion |
| `delta_n` | Normal (N) | Perpendicular to the orbital plane |

The RTN coordinate system is standard in orbital mechanics. A real manoeuvre would be expressed as a velocity change (delta-v) in these directions.

**What the PSO does:**

```
For the highest-risk event:
  Try 30 × 50 = 1500 different manoeuvre combinations
  For each: apply adjustment → re-run model → get new risk score
  Find the combination that gives the LOWEST risk
  Report: "If you fire thrusters by [X, Y, Z], risk drops from A to B"
```

### Step 10: Save Results

```python
test[["event_id", "predicted_risk"]].to_csv("collision_risk_results.csv")
```

Saves all predicted risk scores by event ID. This is the submission format for the ESA competition.

---

## 6. Theory Behind the Techniques

### 6.1 Random Forest — Decision Tree Basics

A single Decision Tree asks a series of yes/no questions:
```
Is miss_distance < 500m?
    YES → Is relative_speed > 3000 m/s?
              YES → HIGH RISK (0.05)
              NO  → MEDIUM RISK (0.002)
    NO  → LOW RISK (0.0001)
```

Random Forest improves this by averaging 200 such trees, each trained on different data subsets and feature subsets → **ensemble learning**.

### 6.2 Standard Scaler — Z-score Normalisation

```
z = (x - μ) / σ

where:
  x = original value
  μ = mean of the feature across training data
  σ = standard deviation of the feature
```

After scaling: mean = 0, std = 1 for every feature.

### 6.3 PSO — Mathematics

Each particle has:
- `x` = current position (manoeuvre vector)
- `v` = velocity (how fast it's moving through solution space)
- `p_best` = its personal best position so far
- `g_best` = the swarm's global best position so far

Update equations each iteration:
```
v_new = w * v_old
      + c1 * r1 * (p_best - x)   ← pull toward personal best
      + c2 * r2 * (g_best - x)   ← pull toward global best

x_new = x_old + v_new
```

Where `w` = inertia weight, `c1/c2` = learning factors, `r1/r2` = random numbers 0-1.

### 6.4 MAE and R² — Model Evaluation

**MAE (Mean Absolute Error):**
```
MAE = (1/n) * Σ |y_actual - y_predicted|
```
Average error in the same units as the target (risk score). Lower = better.

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_residual / SS_total)

SS_residual = Σ (y_actual - y_predicted)²
SS_total    = Σ (y_actual - mean(y_actual))²
```
- R² = 1.0 → perfect predictions
- R² = 0.0 → model is no better than just predicting the mean
- R² < 0  → model is worse than predicting the mean

---

## 7. Summary of the Full Pipeline

```
ESA Dataset (train_data.csv — 162,634 CDMs)
          │
          ▼
  Feature Selection (99 numeric orbital features)
          │
          ▼
  StandardScaler (zero mean, unit variance)
          │
          ▼
  RandomForestRegressor (200 trees, all CPU cores)
          │
          ├──▶  Validation: MAE + R² score
          │
          ▼
  Predict risk for all 24,484 test CDMs
          │
          ├──▶  Top 10 highest-risk events shown
          │
          ▼
  PSO (30 particles × 50 iterations = 1500 evaluations)
  Finds optimal [Δr, Δt, Δn] manoeuvre for highest-risk event
          │
          ▼
  Results: Risk BEFORE vs AFTER manoeuvre + % reduction
          │
          ▼
  collision_risk_results.csv
```

---

## 8. Limitations & Possible Improvements

| Limitation | Improvement |
|---|---|
| PSO only optimises 1 event | Run PSO for all high-risk events |
| Features not engineered | Add CDM sequence features (risk trend over time) |
| Random Forest may miss patterns | Try XGBoost / LightGBM (faster, often more accurate) |
| Manoeuvre in scaled space | Convert delta back to real metres / m/s |
| No temporal modelling | Use LSTM to model how risk evolves across CDM sequence |
| `c_object_type` ignored | One-hot encode it and include it |

---

## 9. Glossary

| Term | Meaning |
|---|---|
| CDM | Conjunction Data Message — alert report for a close approach |
| TCA | Time of Closest Approach |
| RTN | Radial-Tangential-Normal — orbital coordinate system |
| Miss distance | Closest predicted separation between two objects |
| Covariance matrix | Mathematical description of position/velocity uncertainty |
| Mahalanobis distance | Distance measure that accounts for uncertainty/correlation |
| PSO | Particle Swarm Optimisation — nature-inspired optimisation algorithm |
| Random Forest | Ensemble of decision trees, averaged for final prediction |
| StandardScaler | Normalises features to zero mean, unit variance |
| MAE | Mean Absolute Error — average prediction error |
| R² | R-squared — fraction of variance explained by the model |
| Kessler Syndrome | Chain-reaction of collisions making orbits unusable |
| Delta-v | Change in velocity applied by thrusters for a manoeuvre |

---

*Report generated for: Satellite Collision Avoidance ML Project*
*Dataset: ESA Kelvins Collision Avoidance Challenge (2019)*
*Model: Random Forest + PSO*
