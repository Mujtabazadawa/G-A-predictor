# Two-Day Plan: Premier League G+A Predictor (Actual vs Predicted)

## Aim
**Show predicted vs actual `goalsAssistsSum` for each outfield player in 2025-26, and choose the best model.**

Deliverables:
1. A model comparison table (xG+xA benchmark, Ridge, Random Forest, Gradient Boosting) and a justified choice of the best model.
2. A predicted-vs-actual chart for the chosen model (`outputs/predicted_vs_actual.png`).
3. A per-player table of actual vs predicted G+A.

Extra, if there's time: the luck gap (`gap = actual - predicted`) from `copilot-instructions.md`.

## Context
Portfolio ML project (per `copilot-instructions.md`): predict each outfield player's `goalsAssistsSum` from shooting/chance-creation stats and compare it against the provider benchmark `expectedGoals + expectedAssists`. Repo currently has only `README.md`, the instructions file and the CSV in the root.

### How the best model is chosen
- **Main metric: 5-fold cross-validated MAE** (`KFold(5, shuffle=True, random_state=42)`) over all 55 players. It's in goals + assists, so it's easy to explain ("off by ~X G+A per player"), and averaging over 5 folds is far more stable than one ~11-player test split.
- **Also reported:** CV R² (mean ± std), plus test R²/MAE on the 80/20 split so the numbers match the instructions.
- **Tie-break:** if two models are within one CV std of each other, pick the simpler one (Ridge before the tree models). It's easier to explain and less likely to overfit 55 rows.
- **Benchmark:** a model counts as good if it matches or beats xG+xA. It doesn't have to win, because xG+xA is built from shot-level data we don't have.

### Dataset facts (checked)
- `premier_league_complete_stats_whole2025-2026_season.csv`: **78 rows × 113 cols, only 3 teams** (Man Utd 27, Man City 26, Arsenal 25).
- 16 rows have no stats at all (NaN `appearances`, `minutesPlayed`, `totalShots`, …) — squad players who didn't play.
- After cleaning (drop GKs, `appearances >= 5`): **55 players**. John Stones is missing `expectedGoals` (fill 0 for benchmark).
- Real columns exist for: `totalShots`, `shotsOnTarget`, `shotsOffTarget`, `blockedShots`, `keyPasses`, `bigChancesCreated`, `bigChancesMissed`, `minutesPlayed`, `totalAttemptAssist`, `penaltiesTaken`, `touches`, etc.
- Extra leakage columns beyond the instructions list: `penaltyGoals`, `penaltyConversion`, `leftFootGoals`, `rightFootGoals`, `scoringFrequency`, `setPieceConversion`, `passToAssist`(?), `rating`/`totalRating`/`countRating`/`totwAppearances` (ratings reward goals). Exclude them.
- Filename differs from instructions (`_UPDATED` suffix missing) — use the real name.

**Key risk:** 55 rows → a 20% test split is ~11 players, so test R² will be very noisy. Mitigation: report 5-fold CV mean ± std alongside the single split, keep models simple, and note it in the README limitations.

---

## Day 1 — Setup, data, benchmark, baseline model

**Morning: repo setup (~1.5h)**
1. Create folders `data/`, `notebooks/`, `outputs/`, `.github/`.
2. Move CSV → `data/`; move `copilot-instructions.md` → `.github/copilot-instructions.md` (and fix the filename it references).
3. `.gitignore` (`data/*.csv`, `.ipynb_checkpoints/`, `__pycache__/`, `.venv/`), `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, jupyter, ipykernel).
4. Create venv, install, commit "Project structure".

**Midday: load + clean in `notebooks/ga_predictor.ipynb` (~2h)**
- Load, `df.columns`, `df.info()`, missing-value check.
- Drop rows with no stats, drop GKs, keep `appearances >= 5` → expect 55 rows.
- Explicit `LEAKAGE_COLS` list; assert none are in the feature list.
- Fill remaining count NaNs with 0.
- Quick EDA: G+A distribution, G+A vs `totalShots`, vs `keyPasses`, by position.

**Afternoon: benchmark + first model (~2.5h)**
- Benchmark: `xG + xA` vs actual → R², MAE (on all 55 and on the test split, so comparisons are fair).
- Feature set v1 (raw counts): `totalShots`, `shotsOnTarget`, `blockedShots`, `bigChancesCreated`, `bigChancesMissed`, `keyPasses`, `attemptPenaltyMiss`, `attemptPenaltyPost`, `appearances`, `minutesPlayed`, `accurateFinalThirdPasses`, `accurateOppositionHalfPasses`, `accurateCrosses`, `dispossessed`, + `position`.
- `Pipeline(ColumnTransformer(StandardScaler | OneHotEncoder(handle_unknown="ignore")), Ridge)`, `train_test_split(test_size=0.2, random_state=42)`, clip preds at 0.
- End of day: update "Current status" in instructions, commit "Cleaning, benchmark, Ridge baseline".

## Day 2 — Choose the best model, predicted vs actual, write-up

**Morning: compare models and choose (~2.5h)**
- Add RandomForest and GradientBoosting in the same pipeline (shallow trees / `max_depth` small given 55 rows).
- Results table: model | CV MAE mean±std | CV R² mean±std | test MAE | test R², with the xG+xA benchmark as the first row.
- Optional: per-90 feature variant (divide counts by `minutesPlayed/90`) — note target is a season total, so raw counts may actually fit better; compare, don't assume.
- Choose the best model using the rule above and write 2–3 sentences on why (markdown cell).

**Midday: predicted vs actual (~2h)**
- `cross_val_predict` with the same KFold on the chosen pipeline over all 55 players, clipped at 0. Every player then gets a prediction from a model that never saw them, so the chart covers the whole squad instead of just 11 test players.
- Scatter: actual (y) vs predicted (x), y=x line, coloured by position, top ~8 players labelled → `outputs/predicted_vs_actual.png` (dpi=150).
- Optional second panel: xG+xA vs actual on the same axes, to show the model next to the benchmark.
- Per-player table sorted by actual G+A: name, team, position, actual, predicted, xG+xA.
- Sanity check with football knowledge: which players are furthest from the line, and does it make sense (penalty takers, elite finishers)?

**Extra (only if time): luck gap**
- `gap = goalsAssistsSum - predicted`; top 10 over/under-performers; green/red bar chart → `outputs/luck_gap.png`.

**Afternoon: write-up + push (~2h)**
- Markdown cells above each section explaining choices (so you can defend each one).
- README: aim, data (3 teams, 55 players), leakage handling, model comparison table + chosen model and why, predicted-vs-actual plot, limitations (small sample, 3 teams only, gap mixes luck/finishing/penalties, assists depend on teammates), next steps (second season persistence test, more teams, tuning, SHAP).
- Restart kernel & Run All to confirm reproducibility; final commit and push to GitHub.

---

## Critical files
- `notebooks/ga_predictor.ipynb` (new, all analysis)
- `data/premier_league_complete_stats_whole2025-2026_season.csv` (moved)
- `.github/copilot-instructions.md` (moved; update filename + Current status)
- `README.md`, `requirements.txt`, `.gitignore` (new/updated)

## Verification
- Kernel restart + Run All completes without errors.
- Assertion that no leakage column is in the feature list passes.
- Cleaned row count = 55; comparison table printed with CV MAE for all models + benchmark; best model named with a reason.
- `outputs/predicted_vs_actual.png` exists and shows all 55 players.
- `git status` shows the CSV is ignored.
