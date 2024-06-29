# %%
import pandas as pd
import zipfile
import os


# %%
zip_file_path = 'iris.zip'

# %%
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall()

# %%
# Load the dataset into a pandas DataFrame
data_file_path = 'bezdekIris.data'
names_file_path = 'iris.names'


# %%
with open(names_file_path, 'r') as file:
    names_content = file.readlines()

# %%
for line in names_content:
    print(line.strip())

# %%
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# %%
iris_data = pd.read_csv(data_file_path, header=None, names=column_names)

# %%
print(iris_data.head())

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Load the dataset
data_file_path = 'bezdekIris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(data_file_path, header=None, names=column_names)

# %%
print("First few rows of the dataset:")
print(iris_data.head())

# %%
print("\nSummary statistics of the dataset:")
print(iris_data.describe())

# %%
print("\nMissing values in the dataset:")
print(iris_data.isnull().sum())

# %%
sns.pairplot(iris_data, hue='species')
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# %%
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(iris_data['sepal_length'], kde=True)
plt.title('Distribution of Sepal Length')

# %%
plt.subplot(2, 2, 2)
sns.histplot(iris_data['sepal_width'], kde=True)
plt.title('Distribution of Sepal Width')


# %%
plt.subplot(2, 2, 3)
sns.histplot(iris_data['petal_length'], kde=True)
plt.title('Distribution of Petal Length')

# %%
plt.subplot(2, 2, 4)
sns.histplot(iris_data['petal_width'], kde=True)
plt.title('Distribution of Petal Width')

# %%
# Distribution of features
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(iris_data['sepal_length'], kde=True)
plt.title('Distribution of Sepal Length')

plt.subplot(2, 2, 2)
sns.histplot(iris_data['sepal_width'], kde=True)
plt.title('Distribution of Sepal Width')

plt.subplot(2, 2, 3)
sns.histplot(iris_data['petal_length'], kde=True)
plt.title('Distribution of Petal Length')

plt.subplot(2, 2, 4)
sns.histplot(iris_data['petal_width'], kde=True)
plt.title('Distribution of Petal Width')

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv('iris.data', header=None, names=column_names)

correlation_matrix = data.drop(columns=['species']).corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Iris Dataset')
plt.show()


# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
data_file_path = 'bezdekIris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(data_file_path, header=None, names=column_names)

# %%
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Display the shape of the training and testing sets
print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)
print("Training set shape (y_train):", y_train.shape)
print("Testing set shape (y_test):", y_test.shape)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# %%
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# %%
y_pred = classifier.predict(X_test)


# %%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# %%
# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# %%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# %%
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}\n")

evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")

# %%
from sklearn.preprocessing import label_binarize

# %%
# Binarize the output labels for multi-class Precision-Recall AUC calculation
y_train_bin = label_binarize(y_train, classes=y.unique())
y_test_bin = label_binarize(y_test, classes=y.unique())
n_classes = y_train_bin.shape[1]

# %%
# Hyperparameter Tuning with GridSearchCV
log_reg_params = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

knn_params = {
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance']
}

# %%
from sklearn.model_selection import train_test_split, GridSearchCV


# %%
# Perform GridSearchCV for Logistic Regression
log_reg_grid = GridSearchCV(LogisticRegression(max_iter=200), log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)
best_log_reg = log_reg_grid.best_estimator_

# %%
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_


# %%
y_pred_log_reg = best_log_reg.predict(X_test)
y_pred_knn = best_knn.predict(X_test)


# %%
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}\n")

# %%
print("Best Logistic Regression Params:", log_reg_grid.best_params_)
print("Best KNN Params:", knn_grid.best_params_)

# %%
evaluate_model(y_test, y_pred_log_reg, "Logistic Regression (Tuned)")
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors (Tuned)")

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# %%
def plot_precision_recall_curve(y_test_bin, y_scores, model_name, n_classes):
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='best')
    plt.show()


# %%

# For Logistic Regression
y_scores_log_reg = best_log_reg.decision_function(X_test)
plot_precision_recall_curve(y_test_bin, y_scores_log_reg, "Logistic Regression", n_classes)



# %%
# For K-Nearest Neighbors
y_scores_knn = best_knn.predict_proba(X_test)
plot_precision_recall_curve(y_test_bin, y_scores_knn, "K-Nearest Neighbors", n_classes)

# %%
import joblib

joblib.dump(best_log_reg, 'log_reg_model.pkl')
joblib.dump(best_knn, 'knn_model.pkl')


