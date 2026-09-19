"""Interactive demo: type a player's name, get their predicted and actual G+A.

Predictions are out-of-fold from the chosen Ridge model (see notebooks/ga_predictor.ipynb),
so each player is predicted by a model that never saw them during training.

Run from the project folder:  python demo.py
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "premier_league_complete_stats_whole2025-2026_season.csv"
TARGET = "goalsAssistsSum"

# Same features as the notebook (section 5): keep the two in sync
NUMERIC_FEATURES = [
    "totalShots", "shotsOnTarget", "blockedShots",
    "bigChancesCreated", "bigChancesMissed", "keyPasses",
    "attemptPenaltyMiss", "attemptPenaltyPost",
    "appearances", "minutesPlayed",
    "accurateFinalThirdPasses", "accurateOppositionHalfPasses",
    "accurateCrosses", "dispossessed",
]
CATEGORICAL_FEATURES = ["position"]


def normalise(name):
    """Lower-case and strip accents so 'gyokeres' finds 'Gyökeres'."""
    # ø and ł don't split into letter + accent, so map them by hand
    name = name.lower().replace("ø", "o").replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def build_predictions():
    # .copy() packs the 113 columns into one block, which avoids a pandas warning when adding a column
    df = pd.read_csv(DATA_PATH).copy()
    df["search_name"] = df["player_name"].map(normalise)

    # Same cleaning as the notebook (section 3)
    played = df["appearances"].notna()
    outfield = ~df["position"].str.startswith("G")
    enough_games = df["appearances"] >= 5
    players = df[played & outfield & enough_games].copy()
    players[NUMERIC_FEATURES] = players[NUMERIC_FEATURES].fillna(0)
    players["xg_xa"] = players["expectedGoals"].fillna(0) + players["expectedAssists"].fillna(0)

    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("model", Ridge(alpha=1.0)),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X = players[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    players["predicted"] = np.clip(cross_val_predict(model, X, players[TARGET], cv=cv), 0, None)
    return df, players


def why_excluded(row):
    if pd.isna(row["appearances"]):
        return "has no recorded stats this season"
    if row["position"].startswith("G"):
        return "is a goalkeeper (the model covers outfield players only)"
    return f"played only {int(row['appearances'])} games (the model needs at least 5)"


def show(row):
    # Whole numbers only: you can't score 12.3 goals + assists
    actual = int(row[TARGET])
    predicted = int(round(row["predicted"]))
    diff = actual - predicted
    verdict = "more than" if diff > 0 else "fewer than" if diff < 0 else "the same as"
    print(f"\n  {row['player_name']} ({row['team_name']}, {row['position']})")
    print(f"  Actual G+A:     {actual}  (goals {int(row['goals'])}, assists {int(row['assists'])})")
    print(f"  Predicted G+A:  {predicted}")
    print(f"  xG + xA:        {round(row['xg_xa'])}  (provider benchmark)")
    print(f"  Difference:     {diff:+d}  -> {verdict} the model expected\n")


def main():
    df, players = build_predictions()
    print(f"G+A predictor: {len(players)} players from Arsenal, Man City and Man Utd, 2025-26.")
    print("Type a player's name (or part of it). Press Enter on an empty line to quit.")

    while True:
        query = normalise(input("\nPlayer: ").strip())
        if not query:
            break

        matches = players[players["search_name"].str.contains(query, regex=False)]
        if matches.empty:
            # Tell the user if the player exists but was filtered out during cleaning
            others = df[df["search_name"].str.contains(query, regex=False)]
            if others.empty:
                print("  No player found. Try part of the surname, e.g. 'saka'.")
            for _, row in others.iterrows():
                print(f"  {row['player_name']} {why_excluded(row)}.")
            continue

        if len(matches) > 1:
            names = matches["player_name"].tolist()
            for i, name in enumerate(names, 1):
                print(f"  {i}. {name}")
            choice = input("  Several matches, pick a number: ").strip()
            if not (choice.isdigit() and 1 <= int(choice) <= len(names)):
                print("  Not a valid number.")
                continue
            matches = matches.iloc[[int(choice) - 1]]

        show(matches.iloc[0])


if __name__ == "__main__":
    main()
