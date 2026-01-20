import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Veri yükleme
df = pd.read_csv("Smart_Bin.csv")
df = df.dropna()

# Doluluk artışı
df["Doluluk_Artisi"] = df["FL_B"] - df["FL_A"]

# Pivot tablolar
pivot_doluluk = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values="FL_B",
    aggfunc="mean"
)

pivot_hiz = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values="Doluluk_Artisi",
    aggfunc="mean"
)

print("\nOrtalama Doluluk (FL_B):")
print(pivot_doluluk)

print("\nDolma Hızı (FL_B - FL_A):")
print(pivot_hiz)

# Görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_doluluk, annot=True, cmap="YlOrRd", fmt=".1f")
plt.title("Ortalama Doluluk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_hiz, annot=True, cmap="RdYlGn", fmt=".2f")
plt.title("Dolma Hızı")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
container_means = df.groupby("Container Type")["FL_B"].mean().sort_values(ascending=False)
sns.barplot(x=container_means.values, y=container_means.index)
plt.xlabel("Ortalama Doluluk")
plt.tight_layout()
plt.show()

# Encoding
le_c = LabelEncoder()
le_w = LabelEncoder()

df["C_Encoded"] = le_c.fit_transform(df["Container Type"])
df["W_Encoded"] = le_w.fit_transform(df["Recyclable fraction"])

# Hedef değişken
median_hiz = df["Doluluk_Artisi"].median()
df["Hizli_Dolma"] = (df["Doluluk_Artisi"] > median_hiz).astype(int)

X = df[["C_Encoded", "W_Encoded", "VS"]]
y = df["Hizli_Dolma"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(10, 6))
sns.barplot(
    x=rf.feature_importances_,
    y=["Container Type", "Recyclable Fraction", "VS"]
)
plt.tight_layout()
plt.show()

# KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_s, y_train)

y_pred_knn = knn.predict(X_test_s)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
