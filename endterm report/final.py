import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('spotifytracks.csv')
print("Dataset Loaded Successfully.")

print(f"Shape: {df.shape}")
print("\nColumns and Data Types:")
print(df.dtypes)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical Features: {numeric_cols}")

stats_features = ['energy', 'danceability', 'tempo'] 

for col in stats_features:
    if col in df.columns:
        print(f"\nStatistics for {col}:")
        print(f"Mean: {df[col].mean():.4f}")
        print(f"Variance: {df[col].var():.4f}")
        print(f"Std Dev: {df[col].std():.4f}")


if df.isnull().sum().sum() > 0:
    print("Missing values found.")
    df.fillna(df.mean(numeric_only=True), inplace=True)
else:
    print("No missing values found.")

df['mood'] = df['valence'].apply(lambda x: 'Happy' if x >= 0.5 else 'Sad')

sns.set(style="whitegrid")


plt.figure(figsize=(12, 4))
for i, col in enumerate(stats_features):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='mood', y='energy', data=df, palette="Set2")
plt.title("Feature Comparison: Energy vs Mood")
plt.show()

df['mood'] = df['valence'].apply(lambda x: 'Happy' if x >= 0.5 else 'Sad')
target_col = 'mood'

X = df[numeric_cols].drop(columns=['valence'], errors='ignore')
y = df[target_col]


le = LabelEncoder()
y_encoded = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42)
X_test_small, _, y_test_small, _ = train_test_split(X_test, y_test, train_size=1000, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_small, y_train_small)
y_pred_knn = knn.predict(X_test_small)

print("KNN Accuracy:", accuracy_score(y_test_small, y_pred_knn))

new_song = X.iloc[0].values.reshape(1, -1) 
new_song_scaled = scaler.transform(new_song)

prediction = log_reg.predict(new_song_scaled)
predicted_mood = le.inverse_transform(prediction)

print(f"Test Song Features: {new_song[0][:5]} ...")
print(f"Predicted Mood: {predicted_mood[0]}")