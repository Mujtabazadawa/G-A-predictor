# Copilot Instructions: Premier League Goals + Assists Predictor (Actual vs Predicted)

## Project summary
Machine learning portfolio project. I am building a model that predicts each outfield player's **goals + assists** (`goalsAssistsSum`) for the 2025-26 Premier League season from their underlying shooting and chance-creation stats, then compares the prediction with what actually happened.

**Core idea:** shots and chances created are not the whole story when it comes to scoring. Luck plays a huge part. The gap between actual and predicted G+A (`gap = actual - predicted`) is my proxy for luck and finishing:
- Positive gap: scored/assisted more than the numbers suggest (clinical or lucky)
- Negative gap: scored/assisted less than expected (wasteful or unlucky)

I am a Year 3 AI student. Help me build and understand this, not just generate code. Explain non-obvious choices briefly.

## Tech stack
Python 3.10+, pandas, numpy, scikit-learn, matplotlib, Jupyter notebook in VS Code, Git and GitHub.

## Repo structure
```
├── .github/copilot-instructions.md
├── data/        # premier_league_complete_stats_whole2025-2026_season.csv (gitignored)
├── notebooks/   # ga_predictor.ipynb
├── outputs/     # saved plots (predicted_vs_actual.png, luck_gap.png)
├── README.md
├── requirements.txt
└── .gitignore
```

## Data
- File: `data/premier_league_complete_stats_whole2025-2026_season.csv`, one row per player for the whole season (78 rows, 3 teams: Arsenal, Man City, Man Utd; 55 players after cleaning)
- Identity columns: `player_name`, `team_name`, `position`, `appearances`
- Target: `goalsAssistsSum`
- Provider benchmark (NOT model inputs): `expectedGoals`, `expectedAssists`
- Always run `df.columns` and check the real column names before using one. Do not invent names. Look for shots, shots on target, key passes and minutes columns, and use them if they exist.

### Leakage rule (very important)
These columns contain or are derived from the target. Never use them as features:
`goals`, `assists`, `goalsAssistsSum`, `goalConversionPercentage`, `goalsFromInsideTheBox`, `goalsFromOutsideTheBox`, `headedGoals`, `freeKickGoal`, and any penalty/foot-specific goal columns.

### Candidate features
Shooting and chance creation: total shots, shots on target, `blockedShots`, `bigChancesCreated`, `bigChancesMissed`, key passes, `attemptPenaltyMiss`, `attemptPenaltyPost`.
Involvement: `appearances`, minutes played, `accurateFinalThirdPasses`, `accurateOppositionHalfPasses`, `accurateCrosses`, `dispossessed`.
Categorical: `position`.

## Cleaning
- Outfield players only: drop goalkeepers (position starting with "G")
- Keep players with at least 5 appearances (or a minutes cutoff if minutes exist)
- Fill missing counts with 0
- Consider per-90 features if minutes are available

## Modelling approach
1. **Benchmark first:** compute `expectedGoals + expectedAssists` and report its R² and MAE against actual G+A. This is the number to beat or match.
2. **My models:** Ridge, Random Forest, Gradient Boosting, all inside sklearn `Pipeline` with a `ColumnTransformer` (StandardScaler for numeric, OneHotEncoder for position)
3. `train_test_split(test_size=0.2, random_state=42)`
4. Metrics: R² and MAE (in goals + assists). Clip predictions at 0.
5. Plot predicted vs actual for the best model

## The luck gap
- Use `cross_val_predict` (5-fold, shuffled, `random_state=42`) so no player is predicted by a model that trained on them
- `gap = goalsAssistsSum - predicted`
- Show the top 10 over-performers and top 10 under-performers, and a green/red bar chart
- Sanity check the lists with football knowledge (penalty takers and elite finishers should appear as over-performers)

## Testing the "luck" claim (extra)
Finishing skill should repeat next season, luck should not. With a second season's CSV in the same format, compute the gap for both seasons and check the correlation for players in both. A low correlation supports the luck argument (regression to the mean).

## Coding conventions
- Use pipelines to avoid data leakage; never fit scalers or encoders on test data
- `random_state=42` everywhere
- Small notebook cells with a markdown cell above each section
- Short comments that explain why, not what
- Save figures to `outputs/` with `dpi=150`
- Keep `requirements.txt` updated

## How to help me
- Good uses: boilerplate, plotting, refactoring, debugging, README wording, explaining scikit-learn concepts
- Ask before changing the target, features, or metrics
- Explain why any new feature or model should help
- Warn me about data leakage, overfitting, or misleading metrics
- Do not invent column names or data; check the CSV first
- I need to be able to explain every modelling choice myself

## Limitations to keep in mind
- The gap mixes luck, finishing skill, penalties and set pieces; it cannot separate them alone
- Assists depend on teammates finishing, so they are noisier than goals
- One season of data is a small sample

## Planned extras (only when I ask)
Second season for the persistence test, hyperparameter tuning, SHAP feature importance, Streamlit app.

## Current status
Day 1 done (see `PLAN.md`): data loaded and cleaned (55 players), leakage columns removed (extra ones added in the notebook), xG+xA benchmark computed (R² 0.89 / MAE 1.58 on all players), Ridge baseline fitted, 5-fold CV added because the single 80/20 test set has too little spread to judge R² on.
Day 2 done: Ridge, Random Forest and Gradient Boosting compared on 5-fold CV MAE (1.89 / 2.29 / 2.51 vs benchmark 1.58). **Ridge chosen.** Out-of-fold predicted-vs-actual plot saved to `outputs/predicted_vs_actual.png`, per-player table and README written.
Next (optional): luck-gap chart, second-season persistence test, tuning, SHAP.
