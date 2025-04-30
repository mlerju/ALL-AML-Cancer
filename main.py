import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset from a local file
df = pd.read_csv("data_set_ALL_AML_train.csv") # Make sure is in your working directory

# ----------------------
# Data Cleaning
# ----------------------

# Check for missing values
print("Missing values:\n", df.isnull().sum().sum())

expression_data = df.drop(["Gene Description", "Gene Accession Number"], axis=1)

# Select only numeric columns (patient IDs)
sample_columns = [col for col in expression_data.columns if col.isnumeric()]
expression_data = expression_data[sample_columns]

# Transpose data to have patients as rows and expression values as columns
expression_data = expression_data.T

expression_data.index = expression_data.index.astype(int)

# Load the actual labels
labels = pd.read_csv('actual.csv')

train_labels = labels[labels['patient'] <= 38]
# train_labels = train_labels.isin(train_labels.index).copy()

# Make sure patient column is int
train_labels['patient'] = train_labels['patient'].astype(int)

expression_data = expression_data.merge(train_labels, left_index=True, right_on='patient')
expression_data = expression_data.set_index('patient')

print(expression_data.head())

# ----------------------
# Exploratory Data Analysis
# ----------------------

# Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=expression_data['cancer'])
plt.title('Cancer Class Distribution (ALL vs AML)')
plt.xlabel('Cancer Type')
plt.ylabel('Number of Patients')
plt.show()

# Basic Statistics
# Only genes (dropping 'cancer' column)
genes_only = expression_data.drop('cancer', axis=1)
print("Shape of gene expression data:", genes_only.shape)
print("\nBasic stats (first few genes):")
print(genes_only.describe().T.head())

# Correlation Heatmap
sample_genes = genes_only.iloc[:, :30] # Take first 30 genes

plt.figure(figsize=(12,10))
sns.heatmap(sample_genes.corr(), cmap='coolwarm', center=0)
plt.title("Gene Correlation (First 30 genes)")
plt.show()

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(genes_only)

# Create PCA dataframe
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['cancer'] = expression_data['cancer'].values

# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cancer', palette='deep')
plt.title('PCA of Gene Expression Data')
plt.show()

# Heatmap
normalized_data = (genes_only - genes_only.mean()) / genes_only.std()
plt.figure(figsize=(12,8))
sns.heatmap(normalized_data.T, cmap='viridis', annot=False, cbar=True)
plt.title('Normalized Gene Expression Heatmap')
plt.xlabel('Samples')
plt.ylabel('Genes')
plt.show()

# t-distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(genes_only)
tsne_df = pd.DataFrame(tsne_result, columns=['tSNE1', 'tSNE2'])
tsne_df['cancer'] = expression_data['cancer'].values

plt.figure(figsize=(6,8))
sns.scatterplot(data=tsne_df, x='tSNE1', y='tSNE2', hue='cancer', palette='deep')
plt.title('t-SNE Visualization of Gene Expression')
plt.show()

# ----------------------
# Machine Learning Models
# ----------------------

# Load test dataset
df_test = pd.read_csv('data_set_ALL_AML_independent.csv')

# Data cleaning for test dataset
expression_data_test = df_test.drop(["Gene Description", "Gene Accession Number"], axis=1)
sample_columns_test = [col for col in expression_data_test.columns if col.isnumeric()]
expression_data_test = expression_data_test[sample_columns_test]

# Transpose the data
expression_data_test = expression_data_test.T
expression_data_test.index = expression_data_test.index.astype(int)

# Merge with labels for test set
test_labels = labels[labels['patient'] > 38]
expression_data_test = expression_data_test.merge(test_labels, left_index=True, right_on='patient')
expression_data_test = expression_data_test.set_index('patient')

# Data Preprocessing
scaler = StandardScaler()

X_train = scaler.fit_transform(genes_only)
X_test = scaler.transform(expression_data_test.drop('cancer', axis=1))

y_train = expression_data['cancer']
y_test = expression_data_test['cancer']

# # -------------- LOGISTIC REGRESSION --------------
# log_reg = LogisticRegression(max_iter=1000)
# log_reg.fit(X_train, y_train)
# y_pred = log_reg.predict(X_test)
# print("Logistic Regression Accuracy:", log_reg.score(X_test, y_test))
# print(classification_report(y_test, y_pred))
# acc = accuracy_score(y_test, y_pred)
#
# # Confusion Matrix
# ConfusionMatrixDisplay.from_estimator(log_reg, X_test, y_test, colorbar= False)
# plt.title(f"Logistic Regression Confusion Matrix, Accuracy: {acc:.2%}")
# plt.show()
#
# # Cross-Validation
# cross_val_scores = cross_val_score(log_reg, X_train, y_train, cv=10)
# print(f"Cross-Validation scores: {cross_val_scores}")
# print(f"Mean CV Score: {cross_val_scores.mean()}")
#
# # -------------- RANDOM FOREST --------------
#
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators = number of trees
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# print("Random Forest Accuracy:", accuracy_score(y_test,y_pred))
# print("\nClassification Report:\n", classification_report(y_test,y_pred))
# cv_scores = cross_val_score(rf_model, X_train,y_train, cv=10)
# print("\nCross-Validation Scores:", cv_scores)
# print("Mean CV Score:", cv_scores.mean())
# acc = accuracy_score(y_test, y_pred)
#
# ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, colorbar=False)
# plt.title(f"Random Forest Confusion Matrix, Accuracy: {acc:.2%}")
# plt.show()

# -------------- XGBOOST --------------
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    objective='binary:logistic',
    random_state=42
)
xgb_model.fit(X_train,y_train_encoded)
y_pred_encoded = xgb_model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
print("XGBoost Accuracy:", accuracy_score(y_test_encoded,y_pred_encoded))
print("\nClassification Report:\n", classification_report(y_test_encoded,y_pred_encoded))
cv_scores = cross_val_score(xgb_model, X_train, y_train_encoded, cv=10)
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
acc = accuracy_score(y_test_encoded, y_pred_encoded)

# XGB-CMatrix
ConfusionMatrixDisplay.from_estimator(
    xgb_model,
    X_test,
    y_test_encoded,
    display_labels=label_encoder.classes_,
    colorbar=False,
    xticks_rotation=45
)
plt.title(f"XGBoost Confusion Matrix, Accuracy: {acc:.2%}")
plt.show()