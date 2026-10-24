"""
OLYMPIC SPORT CLASSIFICATION USING MULTINOMIAL NAIVE BAYES
==========================================================

Project features:
1. Dataset loading and validation
2. Exploratory data analysis (EDA)
3. Missing-value and duplicate analysis
4. Feature engineering
5. Leakage-aware train/test split
6. CountVectorizer + Multinomial Naive Bayes
7. Evaluation: accuracy, precision, recall, macro/weighted F1
8. Confusion matrix and per-class metrics
9. Baseline and model comparison
10. Feature ablation experiment
11. Example predictions with top-3 model probabilities
12. CSV/TXT/PNG outputs for your project report

Expected dataset:
A CSV such as athlete_events.csv with columns including:
ID, Sex, Age, Height, Weight, Year, Season, Sport
Common dataset: Olympic athlete-event historical records.

IMPORTANT:
- This project predicts Sport, not the exact Event.
- It intentionally excludes ID, Name, Team, NOC, Games, City,
  Medal, Event, and Sport from the input features.
- If the dataset contains repeated athletes, the script uses a
  group split by ID when possible, so the same athlete is not
  present in both training and test data.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Update this filename if your dataset uses another name.
DATASET_PATH = PROJECT_DIR / "athlete_events.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_N_SPORTS = 15
MAX_CONFUSION_CLASSES = 20

# ============================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================

def save_current_figure(filename):
    """Save the current Matplotlib figure at high resolution."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def require_columns(data, required_columns):
    """Stop early with a useful message if required columns are absent."""
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(map(str, data.columns))
        )


def make_engineered_features(data):
    """
    Transform each athlete record into a space-separated token string.

    Multinomial Naive Bayes needs non-negative numerical features.
    CountVectorizer converts these categorical tokens into counts.

    The target Sport and potentially identifying/high-leakage columns
    are deliberately not used as input features.
    """
    features = pd.DataFrame(index=data.index)

    # FEATURE 1: Sex category.
    features["sex_token"] = (
        data["Sex"].fillna("unknown").astype(str).str.strip().str.lower()
        .replace("", "unknown")
    )

    # FEATURE 2: Olympic season.
    features["season_token"] = (
        data["Season"].fillna("unknown").astype(str).str.strip().str.lower()
        .replace("", "unknown")
    )

    # Numeric conversion safely turns invalid values into NaN.
    age = pd.to_numeric(data["Age"], errors="coerce")
    height = pd.to_numeric(data["Height"], errors="coerce")
    weight = pd.to_numeric(data["Weight"], errors="coerce")
    year = pd.to_numeric(data["Year"], errors="coerce")

    # FEATURE 3: Age bands. Each bin includes its lower bound.
    features["age_group"] = pd.cut(
        age,
        bins=[0, 15, 20, 25, 30, 35, 40, np.inf],
        labels=[
            "under_15", "15_19", "20_24", "25_29",
            "30_34", "35_39", "40_plus"
        ],
        right=False,
    ).astype("string").fillna("unknown")

    # FEATURE 4: Height bands (centimetres).
    features["height_band"] = pd.cut(
        height,
        bins=[0, 140, 155, 170, 185, 200, np.inf],
        labels=[
            "under_140", "140_154", "155_169",
            "170_184", "185_199", "200_plus"
        ],
        right=False,
    ).astype("string").fillna("unknown")

    # FEATURE 5: Weight bands (kilograms).
    features["weight_band"] = pd.cut(
        weight,
        bins=[0, 45, 60, 75, 90, 105, np.inf],
        labels=[
            "under_45", "45_59", "60_74",
            "75_89", "90_104", "105_plus"
        ],
        right=False,
    ).astype("string").fillna("unknown")

    # FEATURE 6: Historical Olympic era.
    features["era"] = pd.cut(
        year,
        bins=[-np.inf, 1950, 1970, 1990, 2010, np.inf],
        labels=[
            "before_1950", "1950_1969", "1970_1989",
            "1990_2009", "2010_plus"
        ],
        right=False,
    ).astype("string").fillna("unknown")

    # Prefix values so categories from different features remain distinct.
    # Example: age_group_20_24 and height_band_20_24 are different tokens.
    tokens = pd.DataFrame(index=data.index)
    for column in features.columns:
        tokens[column] = column + "_" + features[column].astype(str)

    # One text document per source row.
    return tokens.agg(" ".join, axis=1)


def calculate_metrics(y_true, y_pred):
    """Return common classification metrics as a dictionary."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


# ============================================================
# SECTION 3: LOAD AND INSPECT DATASET
# ============================================================

print("=" * 70)
print("OLYMPIC SPORT CLASSIFICATION USING MULTINOMIAL NAIVE BAYES")
print("=" * 70)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATASET_PATH}\n\n"
        "Place your CSV beside this Python file and name it "
        "'athlete_events.csv', or update DATASET_PATH in Section 1."
    )

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset path:", DATASET_PATH)
print("Dataset shape (rows, columns):", df.shape)
print("Column names:", list(df.columns))
print("\nFirst five rows:")
print(df.head().to_string(index=False))
print("\nData types and non-null counts:")
df.info()

require_columns(
    df,
    ["Sport", "Sex", "Season", "Age", "Height", "Weight", "Year"]
)

# Save a compact dataset overview for the report.
overview = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Missing Count": [int(value) for value in df.isna().sum()],
    "Missing Percentage": [
        round(float(value), 2) for value in df.isna().mean() * 100
    ],
    "Unique Values": [int(df[column].nunique(dropna=True)) for column in df.columns],
})
overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

# ============================================================
# SECTION 4: DATA QUALITY ANALYSIS
# ============================================================

print("\n========== DATA QUALITY ANALYSIS ==========")

missing_counts = df.isna().sum().sort_values(ascending=False)
missing_percentages = (df.isna().mean() * 100).sort_values(ascending=False)

print("\nMissing values by column:")
print(pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing Percentage": missing_percentages.round(2),
}).to_string())

duplicate_count = int(df.duplicated().sum())
print("\nExact duplicate rows:", duplicate_count)

missing_table = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2),
})
missing_table.to_csv(OUTPUT_DIR / "missing_values_report.csv")

# Graph 1: Missing values by column (only columns with missing values).
missing_plot = missing_percentages[missing_percentages > 0].sort_values()
plt.figure(figsize=(10, 6))
if len(missing_plot):
    missing_plot.plot(kind="barh")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.title("Missing Values by Dataset Column")
else:
    plt.text(0.5, 0.5, "No missing values found", ha="center", va="center")
    plt.axis("off")
save_current_figure("missing_values.png")

# Keep only rows with a valid target. Missing target labels cannot be trained.
df["Sport"] = df["Sport"].astype("string").str.strip()
valid_target = df["Sport"].notna() & df["Sport"].ne("")
df = df.loc[valid_target].copy()

print("\nRows with valid Sport labels:", len(df))
print("Number of unique sports:", df["Sport"].nunique())

if df["Sport"].nunique() < 2:
    raise ValueError("At least two different Sport classes are required.")

# ============================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

sport_counts = df["Sport"].value_counts()
print("\nTop 15 sports by number of records:")
print(sport_counts.head(TOP_N_SPORTS).to_string())
sport_counts.rename_axis("Sport").reset_index(name="Records").to_csv(
    OUTPUT_DIR / "sport_distribution.csv", index=False
)

# Graph 2: Top 15 sports.
plt.figure(figsize=(11, 7))
sport_counts.head(TOP_N_SPORTS).sort_values().plot(kind="barh")
plt.title("Top 15 Olympic Sports by Athlete-Event Records")
plt.xlabel("Number of records")
plt.ylabel("Sport")
save_current_figure("top_sports.png")

# Graph 3: Olympic season distribution.
plt.figure(figsize=(7, 5))
season_counts = df["Season"].fillna("Unknown").value_counts()
season_counts.plot(kind="bar")
plt.title("Olympic Records by Season")
plt.xlabel("Season")
plt.ylabel("Number of records")
plt.xticks(rotation=0)
save_current_figure("season_distribution.png")
season_counts.rename_axis("Season").reset_index(name="Records").to_csv(
    OUTPUT_DIR / "season_distribution.csv", index=False
)

# Graph 4: Athlete age distribution.
age_numeric = pd.to_numeric(df["Age"], errors="coerce")
plt.figure(figsize=(9, 5))
age_numeric.dropna().plot(kind="hist", bins=30, edgecolor="black")
plt.title("Distribution of Athlete Ages")
plt.xlabel("Age (years)")
plt.ylabel("Number of records")
save_current_figure("age_distribution.png")

# Graph 5: Height versus weight, sampled for readability.
height_numeric = pd.to_numeric(df["Height"], errors="coerce")
weight_numeric = pd.to_numeric(df["Weight"], errors="coerce")
scatter_data = pd.DataFrame({
    "Height": height_numeric,
    "Weight": weight_numeric,
}).dropna()
if len(scatter_data) > 5000:
    scatter_data = scatter_data.sample(5000, random_state=RANDOM_STATE)
plt.figure(figsize=(9, 6))
if not scatter_data.empty:
    plt.scatter(scatter_data["Height"], scatter_data["Weight"], alpha=0.35)
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.title("Athlete Height vs Weight (sample)")
else:
    plt.text(0.5, 0.5, "No usable height/weight records", ha="center", va="center")
    plt.axis("off")
save_current_figure("height_weight_scatter.png")

# ============================================================
# SECTION 6: FEATURE ENGINEERING
# ============================================================

print("\n========== FEATURE ENGINEERING ==========")

X_text = make_engineered_features(df)
y = df["Sport"].astype(str).copy()

assert len(X_text) == len(y), "Feature and target lengths do not match."
assert X_text.notna().all(), "Missing feature strings were generated."
assert X_text.str.strip().ne("").all(), "Empty feature strings were generated."

print("Number of feature records:", len(X_text))
print("Number of target labels:", len(y))
print("Number of unique sports:", y.nunique())
print("\nExample engineered feature string:")
print(X_text.iloc[0])
print("Corresponding target sport:", y.iloc[0])
print("Feature engineering validation passed!")

# Graph 6: Distribution of engineered categories.
# We visualize the age, height and weight categories.
engineered_for_plot = pd.DataFrame(index=df.index)
engineered_for_plot["Age Group"] = pd.cut(
    pd.to_numeric(df["Age"], errors="coerce"),
    bins=[0, 15, 20, 25, 30, 35, 40, np.inf],
    labels=["under_15", "15_19", "20_24", "25_29", "30_34", "35_39", "40_plus"],
    right=False,
).astype("string").fillna("unknown")
engineered_for_plot["Height Band"] = pd.cut(
    pd.to_numeric(df["Height"], errors="coerce"),
    bins=[0, 140, 155, 170, 185, 200, np.inf],
    labels=["under_140", "140_154", "155_169", "170_184", "185_199", "200_plus"],
    right=False,
).astype("string").fillna("unknown")
engineered_for_plot["Weight Band"] = pd.cut(
    pd.to_numeric(df["Weight"], errors="coerce"),
    bins=[0, 45, 60, 75, 90, 105, np.inf],
    labels=["under_45", "45_59", "60_74", "75_89", "90_104", "105_plus"],
    right=False,
).astype("string").fillna("unknown")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, column in zip(axes, engineered_for_plot.columns):
    engineered_for_plot[column].value_counts().sort_index().plot(
        kind="bar", ax=axis
    )
    axis.set_title(column)
    axis.set_xlabel("Category")
    axis.set_ylabel("Records")
    axis.tick_params(axis="x", rotation=60)
save_current_figure("engineered_features.png")

# ============================================================
# SECTION 7: TRAIN/TEST SPLIT
# ============================================================

print("\n========== TRAIN/TEST SPLIT ==========")

# Prefer a group split by athlete ID when available, so one athlete
# does not occur in both training and test partitions.
if "ID" in df.columns and df["ID"].nunique() > 1:
    groups = df["ID"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X_text, y, groups=groups))
    X_train_text = X_text.iloc[train_idx]
    X_test_text = X_text.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    split_description = "GroupShuffleSplit by athlete ID"
else:
    # Stratification preserves class proportions where possible.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    split_description = "Stratified random split"

print("Split strategy:", split_description)
print("Training records:", len(X_train_text))
print("Testing records:", len(X_test_text))
print("Training classes:", y_train.nunique())
print("Testing classes:", y_test.nunique())

# ============================================================
# SECTION 8: VECTORIZE AND TRAIN MULTINOMIAL NAIVE BAYES
# ============================================================

print("\n========== MODEL TRAINING ==========")

# Fit CountVectorizer on training text only to avoid vocabulary leakage.
# The Pipeline also makes it easier to compare models consistently.
nb_model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

nb_model.fit(X_train_text, y_train)
y_pred = nb_model.predict(X_test_text)

vectorizer = nb_model.named_steps["vectorizer"]
X_train_matrix = vectorizer.transform(X_train_text)
X_test_matrix = vectorizer.transform(X_test_text)

print("CountVectorizer completed.")
print("Training matrix shape:", X_train_matrix.shape)
print("Testing matrix shape:", X_test_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Multinomial Naive Bayes training completed.")

# ============================================================
# SECTION 9: MODEL EVALUATION
# ============================================================

print("\n========== MODEL EVALUATION ==========")

nb_metrics = calculate_metrics(y_test, y_pred)
for metric_name, value in nb_metrics.items():
    print(f"{metric_name}: {value:.4f}")

report_text = classification_report(
    y_test, y_pred, zero_division=0
)
print("\nClassification Report:")
print(report_text)

(OUTPUT_DIR / "classification_report.txt").write_text(
    "Multinomial Naive Bayes Classification Report\n\n" + report_text,
    encoding="utf-8",
)

metrics_df = pd.DataFrame(
    [{"Model": "Multinomial Naive Bayes", **nb_metrics}]
)
metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

# Graph 7: Overall model performance (actual test-set scores).
# This graph gives one clear comparison of accuracy, macro precision,
# macro recall, macro F1, and weighted F1. Values are not hard-coded.
overall_scores = pd.Series(nb_metrics, dtype=float)
plt.figure(figsize=(10, 6))
bars = plt.bar(overall_scores.index, overall_scores.values)
plt.ylim(0, 1.05)
plt.title("Overall Model Performance - Multinomial Naive Bayes")
plt.ylabel("Score (0 to 1)")
plt.xlabel("Evaluation Metric")
plt.xticks(rotation=20, ha="right")
for bar, score in zip(bars, overall_scores.values):
    plt.text(bar.get_x() + bar.get_width() / 2, score + 0.015,
             f"{score:.3f}", ha="center", va="bottom", fontsize=9)
save_current_figure("overall_model_performance.png")

# Graph 7: Confusion matrix for up to the most frequent test classes.
test_class_counts = y_test.value_counts()
classes_to_plot = test_class_counts.head(MAX_CONFUSION_CLASSES).index.tolist()
mask = y_test.isin(classes_to_plot)
cm = confusion_matrix(
    y_test[mask],
    y_pred[mask],
    labels=classes_to_plot,
)
plt.figure(figsize=(max(10, len(classes_to_plot) * 0.7),
                    max(8, len(classes_to_plot) * 0.6)))
sns.heatmap(
    cm,
    annot=(len(classes_to_plot) <= 15),
    fmt="d",
    cmap="Blues",
    xticklabels=classes_to_plot,
    yticklabels=classes_to_plot,
)
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.xlabel("Predicted Sport")
plt.ylabel("Actual Sport")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
save_current_figure("confusion_matrix.png")

# Graph 8: Per-class precision, recall and F1-score.
report_dict = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
class_rows = []
for label, values in report_dict.items():
    if isinstance(values, dict) and label not in ("macro avg", "weighted avg"):
        class_rows.append({
            "Sport": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1-score": values["f1-score"],
            "Support": values["support"],
        })
per_class_df = pd.DataFrame(class_rows)
per_class_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

if not per_class_df.empty:
    # Separate per-sport precision graph: how reliable positive predictions
    # for each sport are. We display the 15 most common test sports.
    top_precision = (
        per_class_df.sort_values("Support", ascending=False)
        .head(15).sort_values("Precision")
    )
    ax = top_precision.plot(
        x="Sport", y="Precision", kind="barh", legend=False,
        figsize=(10, 7), xlim=(0, 1.05)
    )
    ax.set_title("Per-Sport Precision - 15 Most Common Test Sports")
    ax.set_xlabel("Precision (0 to 1)")
    ax.set_ylabel("Sport")
    save_current_figure("per_sport_precision.png")

    # Separate per-sport recall graph: how many actual examples of each
    # sport the model correctly identifies.
    top_recall = (
        per_class_df.sort_values("Support", ascending=False)
        .head(15).sort_values("Recall")
    )
    ax = top_recall.plot(
        x="Sport", y="Recall", kind="barh", legend=False,
        figsize=(10, 7), xlim=(0, 1.05)
    )
    ax.set_title("Per-Sport Recall - 15 Most Common Test Sports")
    ax.set_xlabel("Recall (0 to 1)")
    ax.set_ylabel("Sport")
    save_current_figure("per_sport_recall.png")

    top_metric_classes = (
        per_class_df.sort_values("Support", ascending=False)
        .head(15)
        .sort_values("F1-score")
    )
    ax = top_metric_classes.set_index("Sport")[
        ["Precision", "Recall", "F1-score"]
    ].plot(kind="barh", figsize=(11, 7))
    ax.set_title("Classification Metrics for 15 Most Common Test Sports")
    ax.set_xlabel("Score")
    ax.set_ylabel("Sport")
    save_current_figure("classification_metrics.png")

# ============================================================
# SECTION 10: BASELINE AND MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

# Majority-class baseline: predicts the most common training sport.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_text.to_frame(name="text"), y_train)
baseline_pred = baseline.predict(X_test_text.to_frame(name="text"))
baseline_metrics = calculate_metrics(y_test, baseline_pred)

comparison_df = pd.DataFrame([
    {"Model": "Majority-Class Baseline", **baseline_metrics},
    {"Model": "Multinomial Naive Bayes", **nb_metrics},
])
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print(comparison_df.to_string(index=False))

# Graph 9: Compare baseline and Naive Bayes metrics.
comparison_plot = comparison_df.set_index("Model")[
    ["Accuracy", "Macro F1", "Weighted F1"]
]
ax = comparison_plot.plot(kind="bar", figsize=(9, 5))
ax.set_title("Model Comparison on the Same Test Set")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Model")
plt.xticks(rotation=0)
save_current_figure("model_comparison.png")

# ============================================================
# SECTION 11: FEATURE ABLATION EXPERIMENT
# ============================================================

print("\n========== FEATURE ABLATION EXPERIMENT ==========")

# Compare all engineered features against a reduced Sex + Season baseline.
# This is an experiment: whether the extra engineered categories help
# must be determined from the held-out test results.
simple_features = pd.Series(
    "sex_token_" +
    df.loc[X_text.index, "Sex"].fillna("unknown").astype(str)
        .str.strip().str.lower().replace("", "unknown")
    + " season_token_" +
    df.loc[X_text.index, "Season"].fillna("unknown").astype(str)
        .str.strip().str.lower().replace("", "unknown"),
    index=X_text.index,
)

X_simple_train = simple_features.loc[X_train_text.index]
X_simple_test = simple_features.loc[X_test_text.index]

simple_model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])
simple_model.fit(X_simple_train, y_train)
simple_pred = simple_model.predict(X_simple_test)
simple_metrics = calculate_metrics(y_test, simple_pred)

ablation_df = pd.DataFrame([
    {"Feature Set": "Sex + Season only", **simple_metrics},
    {"Feature Set": "All engineered features", **nb_metrics},
])
ablation_df.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)
print(ablation_df.to_string(index=False))

# Graph 10: Feature ablation comparison.
ax = ablation_df.set_index("Feature Set")[
    ["Accuracy", "Macro F1", "Weighted F1"]
].plot(kind="bar", figsize=(9, 5))
ax.set_title("Feature Ablation: Does Feature Engineering Help?")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Feature set")
plt.xticks(rotation=0)
save_current_figure("feature_ablation.png")

# ============================================================
# SECTION 12: SAMPLE PREDICTIONS WITH TOP-3 PROBABILITIES
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========")

# Display predictions for up to five held-out records.
sample_count = min(5, len(X_test_text))
sample_text = X_test_text.iloc[:sample_count]
sample_actual = y_test.iloc[:sample_count]
sample_predictions = nb_model.predict(sample_text)
sample_probabilities = nb_model.predict_proba(sample_text)
model_classes = nb_model.named_steps["classifier"].classes_

sample_rows = []
for i in range(sample_count):
    top_indices = np.argsort(sample_probabilities[i])[::-1][:3]
    top_three = [
        f"{model_classes[j]} ({sample_probabilities[i][j]:.1%})"
        for j in top_indices
    ]
    sample_rows.append({
        "Actual Sport": sample_actual.iloc[i],
        "Predicted Sport": sample_predictions[i],
        "Correct": sample_actual.iloc[i] == sample_predictions[i],
        "Top 3 Predictions": " | ".join(top_three),
    })

sample_predictions_df = pd.DataFrame(sample_rows)
print(sample_predictions_df.to_string(index=False))
sample_predictions_df.to_csv(
    OUTPUT_DIR / "sample_predictions.csv", index=False
)

# ============================================================
# SECTION 13: SAVE FINAL SUMMARY
# ============================================================

summary_lines = [
    "OLYMPIC SPORT CLASSIFICATION PROJECT SUMMARY",
    "=" * 48,
    f"Dataset: {DATASET_PATH.name}",
    f"Valid target records: {len(df)}",
    f"Number of sports: {df['Sport'].nunique()}",
    f"Split strategy: {split_description}",
    f"Training records: {len(X_train_text)}",
    f"Testing records: {len(X_test_text)}",
    f"Vocabulary size: {len(vectorizer.vocabulary_)}",
    "",
    "Multinomial Naive Bayes metrics:",
]
summary_lines.extend(
    [f"{name}: {value:.4f}" for name, value in nb_metrics.items()]
)
summary_lines.extend([
    "",
    "See classification_report.txt, evaluation_metrics.csv,",
    "model_comparison.csv, feature_ablation.csv, and PNG graphs",
    "in the outputs folder.",
])
(OUTPUT_DIR / "project_summary.txt").write_text(
    "\n".join(summary_lines), encoding="utf-8"
)

print("\n========== PROJECT COMPLETED ==========")
print("All available reports and graphs were saved to:")
print(OUTPUT_DIR)
print("\nGenerated key files:")
for output_file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", output_file.name)
print("\nNote: results are specific to your dataset and split.")
