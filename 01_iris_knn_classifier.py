# DecodeLabs - Project 2: Data Classification Using AI
# Dataset: Iris | Algorithm: K-Nearest Neighbors (KNN)
# Pipeline: INPUT -> PROCESS -> OUTPUT


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
import matplotlib.pyplot as plt

# STEP 1: LOAD AND UNDERSTAND THE DATA (INPUT)
iris = load_iris()
X = iris.data          # 4 features: sepal length/width, petal length/width
y = iris.target        # 3 classes: 0=setosa, 1=versicolor, 2=virginica

print("Samples   :", X.shape[0])
print("Features  :", X.shape[1])
print("Feature names:", iris.feature_names)
print("Classes   :", list(iris.target_names))
print("First 5 rows:\n", X[:5])

# STEP 2: SPLIT INTO TRAINING (80%) AND TESTING (20%) SETS
# shuffle=True mixes the data first to remove order bias
# stratify=y keeps the same class balance in both sets
# random_state=42 gives the same result every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# STEP 3: FEATURE SCALING (Mean = 0, Variance = 1)
# fit ONLY on training data, then transform both.
# (Fitting on test data would leak information.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 4: TRAIN THE MODEL (PROCESS)
# Instantiate -> Fit -> Predict
model = KNeighborsClassifier(n_neighbors=5)   # instantiate
model.fit(X_train_scaled, y_train)            # fit (learn from training data)
predictions = model.predict(X_test_scaled)    # predict on unseen test data

# STEP 5: EVALUATE THE MODEL (OUTPUT)
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Weighted F1 Score:", round(f1_score(y_test, predictions, average="weighted"), 4))

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# STEP 6: CHOOSING THE BEST "K" (try K = 1 to 20)
k_values = range(1, 21)
error_rates = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    error_rates.append(1 - knn.score(X_test_scaled, y_test))

best_k = k_values[error_rates.index(min(error_rates))]
print("Best K (lowest error):", best_k)

# STEP 7: VISUALS (confusion matrix + error vs K)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Confusion matrix plot
axes[0].imshow(cm, cmap="Blues")
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(iris.target_names)
axes[0].set_yticklabels(iris.target_names)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

# Error rate vs K plot
axes[1].plot(list(k_values), error_rates, marker="o")
axes[1].set_title("Choosing K (Error Rate vs K)")
axes[1].set_xlabel("K value")
axes[1].set_ylabel("Error rate")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
axes[1].legend()

plt.tight_layout()
plt.savefig("project2_results.png")
plt.show()

# STEP 8: PREDICT A BRAND-NEW FLOWER
# Order: sepal length, sepal width, petal length, petal width (cm)

new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled = scaler.transform(new_flower)   # must scale it too!
result = model.predict(new_flower_scaled)
print("\nNew flower prediction:", iris.target_names[result[0]])
