"""
Iris Dataset - Machine Learning Classification

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# SECTION 1: RECREATE THE CLEANED DATAFRAME FROM THE EDA PHASE
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df = df.drop_duplicates()  

# SECTION 2: PREPARE FEATURE MATRIX (X) AND TARGET VARIABLE (y)
X = df.drop(columns=['species'])
y = df['species']

print("Feature matrix (X) shape:", X.shape)   # (number of rows, number of features)
print("Target vector (y) shape :", y.shape)   # (number of rows,)

# SECTION 3: SPLIT DATA INTO TRAINING AND TESTING SETS (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", X_train.shape[0], "samples")
print("Testing set size :", X_test.shape[0], "samples")

# SECTION 4: DEFINE THE FOUR CLASSIFICATION MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42),
}

# SECTION 5: TRAIN, PREDICT, AND EVALUATE EACH MODEL
results = []                
predictions_by_model = {}    

for model_name, model in models.items():
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    #  Step A: Train (fit) the model on the training data 
    model.fit(X_train, y_train)

    #  Step B: Make predictions on the unseen test data 
    y_pred = model.predict(X_test)
    predictions_by_model[model_name] = y_pred

    #  Step C: Evaluate using Accuracy Score 
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    #  Step D: Evaluate using Classification Report 
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step E: Evaluate using Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("Confusion Matrix:")
    print(cm)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

# SECTION 6: COMPARE ALL MODELS IN A SINGLE DATAFRAME
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
print(results_df)
# SECTION 7: IDENTIFY THE BEST-PERFORMING MODEL
best_model_name = results_df.iloc[0]["Model"]
best_model_accuracy = results_df.iloc[0]["Accuracy"]
print(f"\nBest Performing Model: {best_model_name} "
      f"(Accuracy: {best_model_accuracy:.4f})")

# SECTION 8: VISUALIZATION 1 - ACCURACY COMPARISON BAR CHART
plt.figure(figsize=(7, 5))
plt.bar(results_df["Model"], results_df["Accuracy"], color="steelblue")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)              # accuracy is always between 0 and 1
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# SECTION 9: VISUALIZATION 2 - CONFUSION MATRIX FOR EACH MODEL
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()  
for ax, (model_name, model) in zip(axes, models.items()):
    cm = confusion_matrix(
        y_test,
        predictions_by_model[model_name],
        labels=model.classes_
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(model_name)

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()
# SECTION 10: VISUALIZATION 3 - DECISION TREE STRUCTURE
from sklearn.tree import plot_tree

plt.figure(figsize=(14, 8))
plot_tree(
    models["Decision Tree"],
    feature_names=X.columns,
    class_names=models["Decision Tree"].classes_,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Trained Decision Tree Structure")
plt.tight_layout()
plt.savefig("decision_tree_structure.png")
plt.show()

print("\nAll visualizations have been saved as PNG files:")
print(" - accuracy_comparison.png")
print(" - confusion_matrices.png")
print(" - decision_tree_structure.png")
