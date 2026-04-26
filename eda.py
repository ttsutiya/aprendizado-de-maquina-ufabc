import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("clean_editable_plants.csv")


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="water_code")
plt.title("distribuicao water_code")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="nutrients_code")
plt.title("distribuicao nutrients_code")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="temperature_class_code")
plt.title("distribuicao temperature_class_code")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="cultivation")
plt.title("distribuicao cultivation")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("corr heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="cultivation", y="water_code")
plt.title("water_code vs cultivation")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="temperature_class_code", y="water_code")
plt.title("water_code vs temperature_class_code")
plt.show()
