# Import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv"
df = pd.read_csv(url)

# Tampilkan 5 data awal
print(df.head())

# Informasi dataset
print(df.info())

# Cek missing value
print("Missing Values:\n", df.isnull().sum())

# Deskripsi statistik
print(df.describe())

# Pisahkan fitur dan target
# The error indicates that the column 'diabetes' was not found.
# Let's check the column names in the dataframe to find the correct target column.
print("\nDataFrame columns:", df.columns)

# Based on the typical structure of this dataset and the output of df.head(),
# it appears the target column is named 'Class'. Let's use that instead of 'diabetes'.
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualisasi pohon keputusan
plt.figure(figsize=(20,10))
# Update class_names to match the values in the 'Class' column (0 and 1)
plot_tree(model, feature_names=X.columns, class_names=["0", "1"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()