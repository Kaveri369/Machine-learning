# 🏅 Olympic Sport Classification using Multinomial Naive Bayes

<p align="center">
  <strong>An End-to-End Machine Learning Project for Olympic Sport Prediction</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/ML-Naive%20Bayes-orange" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Library-Scikit--learn-F7931E?logo=scikitlearn" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Data-Pandas-150458?logo=pandas" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Status-Academic%20Project-success" alt="Project Status"/>
</p>

## 📌 Project Overview

This project develops a machine learning classification system that predicts an athlete's **Olympic sport** from selected demographic, physical, and competition-related features.

Using the Olympic athlete-event dataset, the project demonstrates the complete machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, text vectorization, model training, and performance evaluation.

The classification model uses **Multinomial Naive Bayes**, a probabilistic algorithm that works with non-negative feature representations such as token counts.

The goal is to build a reproducible, well-documented ML project while understanding how preprocessing and feature design influence classification performance.

> **Note:** The model predicts the sport category, not the Olympic event name or medal outcome.

## 🎯 Project Objectives

- Understand and preprocess a real-world Olympic dataset.
- Perform exploratory data analysis using visualizations.
- Engineer meaningful features from athlete information.
- Convert categorical feature descriptions into numerical representations using `CountVectorizer`.
- Train a Multinomial Naive Bayes classifier.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Compare the model against a simple baseline.
- Analyze the impact of feature engineering through an ablation experiment.
- Save evaluation graphs and reports for interpretation.

## 🧠 Machine Learning Algorithm

### Multinomial Naive Bayes

Multinomial Naive Bayes is a supervised classification algorithm based on Bayes' theorem. It estimates the probability of a class given the observed features.

Bayes' theorem:

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

Where:

- \(P(C\mid X)\): posterior probability of a class.
- \(P(X\mid C)\): likelihood of the observed features given the class.
- \(P(C)\): prior probability of the class.
- \(P(X)\): evidence.

The classifier selects the sport with the highest estimated posterior probability.

In this project, engineered feature tokens are converted into a numerical count matrix using `CountVectorizer`. Multinomial Naive Bayes then learns patterns associated with different sports.

**Why this algorithm?**

- Simple and computationally efficient.
- Suitable for non-negative count-based feature representations.
- Provides class probability estimates.
- Offers a useful way to study the relationship between feature representation and classification performance.

**Important limitation:** Multinomial Naive Bayes assumes a particular feature likelihood model and conditional independence between features. It may not capture every relationship between an athlete's physical attributes and sport.

## 📂 Dataset

The project uses an Olympic athlete-event dataset, commonly distributed as `athlete_events.csv`.

Typical columns include:

| Column | Description |
|---|---|
| `ID` | Athlete identifier |
| `Name` | Athlete name |
| `Sex` | Athlete sex recorded in the dataset |
| `Age` | Age at the competition |
| `Height` | Height in centimetres |
| `Weight` | Weight in kilograms |
| `Team` | Represented team |
| `NOC` | National Olympic Committee code |
| `Games` | Olympic edition |
| `Year` | Olympic year |
| `Season` | Summer or Winter |
| `Sport` | Target classification label |
| `Event` | Specific Olympic event |
| `Medal` | Medal result, if available |

The dataset can be obtained from a legitimate public dataset source, such as Kaggle. Verify the dataset version, licensing, and column names before use.

**Target variable:** `Sport`

**Input features:** The engineered athlete and competition features selected by the implementation.

The `Sport` target must not be included in the input features. Names, team identifiers, event labels, medals, and other fields that could introduce leakage should be excluded unless their use is explicitly justified.

## ⚙️ Project Workflow

1. **Data loading** — Read the CSV dataset and validate required columns.
2. **Data cleaning** — Handle missing values, invalid values, and inconsistent categorical text.
3. **Exploratory data analysis** — Examine sport frequencies, missing values, season distribution, and physical attributes.
4. **Feature engineering** — Convert numerical ranges and categorical values into informative feature tokens.
5. **Train-test splitting** — Separate training and testing data before fitting the vectorizer.
6. **Text vectorization** — Transform engineered feature tokens into a numerical count matrix.
7. **Model training** — Fit the Multinomial Naive Bayes classifier.
8. **Evaluation** — Measure performance on held-out data.
9. **Baseline comparison** — Compare the model against a simple majority-class predictor.
10. **Feature ablation** — Compare a reduced feature set with the complete engineered feature set.
11. **Visualization and reporting** — Save plots and evaluation reports to the output directory.

## 🛠️ Feature Engineering

Feature engineering transforms raw columns into representations that the classifier can use.

Examples include:

| Original feature | Engineered representation |
|---|---|
| Sex | `sex_f`, `sex_m` |
| Season | `season_summer`, `season_winter` |
| Age | Age-range categories |
| Height | Height-range categories |
| Weight | Weight-range categories |
| Year | Olympic-era categories |

The actual categories depend on the implementation.

For example, an athlete record might be represented as:

`sex_f age_20_29 height_160_169 weight_50_59 season_summer era_2000s`

These tokens are then converted into numerical features with `CountVectorizer`.

### Why feature engineering matters

- It gives the model interpretable feature categories.
- It can capture broad patterns that raw numerical values do not express directly in a token-count representation.
- It allows experiments comparing different feature groups.
- It makes the preprocessing process explicit and reproducible.

Numerical ranges should be documented, missing values handled consistently, and the same transformation logic applied during training and prediction.

## 📊 Evaluation Metrics

The project evaluates the classifier using multiple metrics.

| Metric | What it measures |
|---|---|
| Accuracy | Fraction of all test predictions that are correct |
| Precision | Fraction of predicted positive class examples that are correct |
| Recall | Fraction of actual class examples correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Counts correct and incorrect predictions by class |

For a multiclass dataset, macro-averaged precision, recall, and F1-score give each sport equal weight. Weighted averages account for class frequency.

Accuracy alone may be misleading if the dataset contains many more examples of some sports than others. Therefore, the classification report and confusion matrix are important for understanding class-wise performance.

## 📈 Visualizations and Outputs

After running the program, the `outputs/` directory is intended to contain diagnostic plots and reports, including:

- Sport distribution chart
- Missing-value analysis
- Season distribution
- Age distribution
- Height-versus-weight visualization
- Overall model performance graph
- Per-sport precision graph
- Per-sport recall graph
- Confusion matrix heatmap
- Classification report and prediction examples

The exact files depend on the final script and the successful execution of the dataset pipeline. Include screenshots or generated plots in the repository only after verifying that they were produced by your run.

## 🗂️ Repository Structure

```text
olympics-sport-classification/
│
├── olympics_classification.py
├── athlete_events.csv
├── README.md
├── requirements.txt
│
└── outputs/
    ├── overall_model_performance.png
    ├── per_sport_precision.png
    ├── per_sport_recall.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

The output filenames above illustrate the intended organization; retain the names actually generated by your script.

**Dataset note:** If the dataset license or repository size makes redistribution inappropriate, do not commit the full CSV. Add instructions explaining where to obtain it and place it in the project directory locally.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/olympics-sport-classification.git
cd olympics-sport-classification
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository name.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing the packages required by the implementation:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install them:

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place `athlete_events.csv` in the same directory as `olympics_classification.py`, unless you have changed the dataset path in the configuration.

### 5. Run the project

```bash
python olympics_classification.py
```

Check the terminal for evaluation metrics and inspect the `outputs/` directory for generated graphs and reports.

## 🔬 Experiments and Improvements

Potential extensions include:

- Compare Multinomial Naive Bayes with Logistic Regression and Complement Naive Bayes.
- Test different feature groups and age, height, and weight bin boundaries.
- Compare `CountVectorizer` with `TfidfVectorizer`, while respecting each model's input assumptions.
- Analyze performance on rare and common sports separately.
- Evaluate how results change when athletes, rather than individual athlete-event rows, are separated between training and testing.
- Tune the smoothing parameter `alpha` using validation data.
- Add cross-validation and reproducible random seeds.

Any model comparison should use consistent splits and preprocessing rules. Fit transformations only on training data to prevent test-set leakage.

## ⚠️ Limitations and Responsible Use

- Olympic sports have different participation patterns, and the dataset may be imbalanced.
- Athlete characteristics overlap across sports, so a prediction may be uncertain.
- Historical data may contain missing or inconsistent measurements.
- The dataset contains repeated observations for athletes who participated in multiple events or Olympic editions.
- A random row-level split may place observations from the same athlete in both training and testing data, producing optimistic estimates.
- Demographic and physical features should not be used to make high-stakes judgments about an individual.

The model is an educational classification experiment, not a reliable tool for deciding which sport an individual should participate in.

## 🚀 Future Scope

- Develop a Streamlit interface for interactive predictions.
- Add probability-based top-three predictions with clear uncertainty explanations.
- Compare multiple classification algorithms.
- Add cross-validation and class-imbalance analysis.
- Track experiments and document model limitations.
- Create a reproducible pipeline for future Olympic dataset versions.

## 👩‍💻 Technologies Used

- **Python** — Core programming language
- **Pandas and NumPy** — Data handling and numerical operations
- **Matplotlib and Seaborn** — Data visualization
- **Scikit-learn** — Feature extraction, model training, and evaluation
- **Git and GitHub** — Version control and project documentation

## 📚 Learning Outcomes

Through this project, the learner can demonstrate practical understanding of:

- Supervised machine learning and multiclass classification
- Bayes' theorem and Multinomial Naive Bayes
- Data cleaning and exploratory analysis
- Feature engineering and text vectorization
- Train-test separation and leakage prevention
- Classification metrics and model comparison
- Reproducible ML experimentation
- GitHub project organization and documentation

## 📄 License

Add a license appropriate for your original code and verify the dataset's separate license before redistributing any data. A code license does not automatically grant rights to redistribute the dataset.

---

<p align="center">
  <strong>Built as an educational machine learning project.</strong>
</p>
