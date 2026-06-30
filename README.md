# Iris EDA + ML Classification Project

A small end-to-end project on the Iris dataset — first exploring the data, then building and comparing a few classification models on it with Scikit-learn.

## What's in here

The project has two parts:

1. **EDA** – loading the data, cleaning it up, looking at summary stats, and a few basic plots.
2. **Modeling** – training four different classifiers and comparing how they do.

## The dataset

Standard Iris dataset, pulled straight from `sklearn.datasets.load_iris()`. 150 rows, 4 numeric features (sepal/petal length and width, in cm), 3 species as the target. One duplicate row got removed during cleaning, so the working dataset is 149 rows.

## Tools

Python, Pandas, Matplotlib, and Scikit-learn. That's it — kept it simple on purpose.

## Files

- `datacleaning.py` – the EDA script
- `MLalgorithm.py` – training + evaluating the models
- `accuracy_comparison.png`, `confusion_matrices.png`, `decision_tree_structure.png` – plots the ML script generates
- `README.md` 

## What the EDA covers

Checked for missing values (none), checked for duplicates (found 1, dropped it), ran `describe()` for summary stats, then made three plots: a bar chart of species counts, a scatter plot of sepal length vs width, and a histogram of petal length.

## What the modeling covers

Built X/y from the cleaned data, split it 80/20 (`random_state=42`, stratified so each species is represented proportionally in both sets), then trained four models — Logistic Regression, KNN, Decision Tree, and SVM. Each one gets evaluated with accuracy, a classification report, and a confusion matrix, and then I compare them all in one table to see which did best.

### How they did

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors | 1.00 |
| SVM | 1.00 |
| Logistic Regression | 0.97 |
| Decision Tree | 0.93 |

KNN and SVM both got everything right on the test set, Logistic Regression missed one, and the Decision Tree mixed up Versicolor and Virginica a couple times — which tracks, since those two species overlap the most.

Worth flagging: the test set here is only 30 samples, so "100% accuracy" isn't as impressive as it sounds — on a bigger or noisier dataset these numbers would move around more. Wouldn't read too much into the exact ranking without cross-validation.

![Accuracy comparison](accuracy_comparison.png)
![Confusion matrices](confusion_matrices.png)
![Decision tree](decision_tree_structure.png)

## Running it

```bash
pip install pandas matplotlib scikit-learn
python eda_iris.py
python iris_ml_classification.py
```

## Things I'd add next

- Hyperparameter tuning for KNN and SVM
- Try scaling the features and see if it changes anything
- Maybe throw in Random Forest or Naive Bayes for comparison