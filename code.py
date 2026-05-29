import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv(r"C:\Project\Credit Card Fraud Detection\dataset.csv")

# Show first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check null values
print("\nNull Values:")
print(df.isnull().sum())

# Fraud vs Normal transactions
print("\nFraud vs Normal Transactions:")
print(df['Class'].value_counts())

# Fraud vs Normal graph

sns.countplot(x='Class', data=df)

plt.title("Fraud vs Normal Transactions")

plt.xlabel("Class")
plt.ylabel("Count")

# Save graph as image
plt.savefig("fraud_graph.png")
print("Graph saved successfully")

# Separate fraud and normal transactions
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

# Print average amounts
print("Average Fraud Transaction Amount:")
print(fraud['Amount'].mean())

print("\nAverage Normal Transaction Amount:")
print(normal['Amount'].mean())

# Create graph
plt.figure(figsize=(8,5))

sns.histplot(fraud['Amount'], bins=30)

plt.title("Fraud Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Count")

# Save graph
plt.savefig("fraud_amount_distribution.png")

print("\nGraph saved successfully")

# Correlation matrix
correlation = df.corr()

# Graph size
plt.figure(figsize=(12,8))

# Heatmap
sns.heatmap(correlation)

# Title
plt.title("Correlation Heatmap")

# Save graph
plt.savefig("correlation_heatmap.png")

print("Heatmap saved successfully")


# Features and target
X = df.drop('Class', axis=1)
Y = df['Class']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, Y_train)

# Predictions
X_test_prediction = model.predict(X_test)

# Accuracy
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy Score:")
print(test_data_accuracy)

# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)

print("\nConfusion Matrix:")
print(cm)

# Precision
precision = precision_score(Y_test, X_test_prediction)

# Recall
recall = recall_score(Y_test, X_test_prediction)

# F1 Score
f1 = f1_score(Y_test, X_test_prediction)

print("\nPrecision Score:")
print(precision)

print("\nRecall Score:")
print(recall)

print("\nF1 Score:")
print(f1)

# Sample transaction from test data
sample_data = X_test.iloc[0]

# Convert into 2D array
sample_data_reshaped = sample_data.values.reshape(1, -1)

# Prediction
prediction = model.predict(sample_data_reshaped)

# Output
if prediction[0] == 0:
    print("\nNormal Transaction")
else:
    print("\nFraud Transaction")

# Create dashboard dataset
dashboard_data = df.copy()

# Save CSV for Power BI
dashboard_data.to_csv("fraud_dashboard_data.csv", index=False)

print("\nDashboard dataset exported successfully")
