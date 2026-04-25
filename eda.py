import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("clean_editable_plants.csv")


# Perguntas originais
# Do plants that require more sunlight also require higher temperatures?
# What cultivation classes require the most water?

df.iloc[0]

df.columns

quant = ["preferred_ph_lower", "preferred_ph_upper"]
quali_ord = ["water_code", "nutrients_code", "temperature_class_code"]
quali_nom = ["cultivation", "sunlight"]


corr = df[quant + quali_ord].corr()
corr

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("heat map")
plt.xticks(rotation=45)
plt.show()


sns.pairplot(
    df,
    vars=quant + quali_ord,
    hue="cultivation",
    corner=True,
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)
plt.show()

sns.pairplot(
    df,
    vars=quant + quali_ord,
    hue="sunlight",
    corner=True,
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)
plt.show()


sns.violinplot(
    data=df,
    x="water_code",
    y="sunlight",
    inner="quart",  # Show quartiles inside the violin
)
plt.xticks()
plt.show(0)

df.water.unique()

sun_temp = df.groupby("sunlight").temperature_class_code.mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(x=sun_temp.index, y=sun_temp.values)
plt.title("mean sunlight vs temperature_class")
plt.ylabel("Temp. Class Code")
plt.xticks(rotation=45)
plt.show()


cult_water = df.groupby("cultivation")["water_code"].mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(x=cult_water.index, y=cult_water.values)
plt.title("mean cult vs water")
plt.ylabel("Water code")
plt.xticks(rotation=45)
plt.show()
