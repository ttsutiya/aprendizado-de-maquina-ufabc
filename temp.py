import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, randint
from sklearn.model_selection import (
    LeaveOneOut,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# ### 4.1 Carregamento e Pré-processamento dos Dados
#
# - Lemos o dataset
# - Separamos os nomes das plantas
# - Removemos colunas irrelevantes
# - Definimos variáveis de entrada (X) e saída (y)

# In[52]:


df = pd.read_csv("clean_editable_plants.csv")


# In[53]:


nomes_plantas = df["common_name"]


# In[54]:


df = df.drop(
    columns=[
        "taxonomic_name",
        "common_name",
        "water",
        "nutrients",
        "temperature_class",
    ]
)


# In[55]:


y = df["water_code"]
X = df.drop(
    columns=[
        "water_code",
    ]
)

loo = LeaveOneOut()


# ### 4.2 Regressão Logística
#

# In[56]:


log_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=1000, random_state=0)),
    ]
)

log_param_dist = {"log_reg__C": loguniform(1e-3, 1e3)}

random_search = RandomizedSearchCV(
    estimator=log_pipe,
    param_distributions=log_param_dist,
    n_iter=100,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
)


# In[57]:


random_search.fit(X, y)


# In[58]:


c_values = random_search.cv_results_["param_log_reg__C"].data.astype(float)
scores = random_search.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(c_values, scores)
plt.xscale("log")
plt.xlabel("Parâmetro C")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Regressão Logística: Ajuste de parâmetro")
plt.grid(True, which="both", alpha=0.5)
plt.show()


# In[59]:


print(
    f"Melhor parâmetro (C = 1/lambda): {random_search.best_params_['log_reg__C']:.4f}"
)
print(f"Lambda: {1 / random_search.best_params_['log_reg__C']:.4f}")
print(f"Melhor acurácia LOOCV: {random_search.best_score_:.4f}")


# No EDA notou-se que uma grande quantidade de dados pertencem ao `water_code=3`. Onde 83 dos 123 dados do dataset (~67%) pertencem a esse grupo.
#
# Isso significa que os nosso modelos podem estar apenas chutando o grupo 3 para qualquer planta. Para isso vamos analisar a matriz de confusão dos modelos.

# In[60]:


best_log_model = random_search.best_estimator_
y_pred_log = cross_val_predict(best_log_model, X, y, cv=loo, n_jobs=-1)


# In[61]:


print(classification_report(y, y_pred_log, zero_division=0))


# In[62]:


cm = confusion_matrix(y, y_pred_log)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de confusão: regressão logística")
plt.show()


# Para tentar arrumar isso vamos usar o modo `balanced` na regressão logística, aumentando a penalidade ao errar os grupos menores.

# In[63]:


log_pipe_balanced = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "log_reg",
            LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced"),
        ),
    ]
)

random_search_balanced = RandomizedSearchCV(
    estimator=log_pipe_balanced,
    param_distributions=log_param_dist,
    n_iter=100,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
)


# In[64]:


random_search_balanced.fit(X, y)


# In[65]:


c_values_balanced = random_search_balanced.cv_results_["param_log_reg__C"].data.astype(
    float
)
scores_balanced = random_search_balanced.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(c_values_balanced, scores_balanced)
plt.xscale("log")
plt.xlabel("Parâmetro C")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Regressão Logística (Balanceada): Ajuste de parâmetro")
plt.grid(True, which="both", alpha=0.5)
plt.show()


# In[66]:


print(
    f"Melhor parâmetro (C = 1/lambda): {random_search_balanced.best_params_['log_reg__C']:.4f}"
)
print(f"Lambda: {1 / random_search_balanced.best_params_['log_reg__C']:.4f}")
print(f"Melhor acurácia LOOCV: {random_search_balanced.best_score_:.4f}")


# In[67]:


best_log_balanced = random_search_balanced.best_estimator_
y_pred_log_balanced = cross_val_predict(best_log_balanced, X, y, cv=loo, n_jobs=-1)


# In[69]:


print(classification_report(y, y_pred_log_balanced, zero_division=0))


# In[68]:


cm_balanced = confusion_matrix(y, y_pred_log_balanced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_balanced, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de confusão: regressão logística (balanceada)")
plt.show()

### BONUS
coefs = random_search_balanced.best_estimator_.named_steps["log_reg"].coef_
classes = random_search_balanced.best_estimator_.named_steps["log_reg"].classes_

weights = pd.DataFrame(coefs, columns=X.columns, index=classes)

plt.figure(figsize=(12, 6))
sns.heatmap(weights.T, annot=True, cmap="RdBu")
plt.show()


# ### 4.3 KNN Regression

# In[70]:


knn_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)

knn_param_grid = {"knn__n_neighbors": np.arange(1, 31)}

knn_grid = GridSearchCV(
    estimator=knn_pipe, param_grid=knn_param_grid, cv=loo, scoring="accuracy", n_jobs=-1
)


# In[71]:


knn_grid.fit(X, y)


# In[72]:


k_values = knn_param_grid["knn__n_neighbors"]
knn_scores = knn_grid.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_scores)
plt.xlabel("Número de vizinhos(K)")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("KNN: Ajuste de parâmetro")
plt.grid(True, which="both")
plt.show()


# In[73]:


print(f"Melhor K: {knn_grid.best_params_['knn__n_neighbors']}")
print(f"Melhor acurácia KNN: {knn_grid.best_score_:.4f}\n")


# In[74]:


best_knn = knn_grid.best_estimator_
y_pred_knn = cross_val_predict(best_knn, X, y, cv=loo, n_jobs=-1)


# In[75]:


print(classification_report(y, y_pred_knn, zero_division=0))


# In[76]:


cm_knn = confusion_matrix(y, y_pred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN")
plt.show()


# Para o kNN, em vez de ter um peso uniform para todos os vizinhos, iremos dar mais peso aos vizinhos próximos.

# In[77]:


knn_pipe_dist = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance")),
    ]
)

knn_grid_dist = GridSearchCV(
    estimator=knn_pipe_dist,
    param_grid=knn_param_grid,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
)


# In[78]:


knn_grid_dist.fit(X, y)


# In[79]:


k_values = knn_param_grid["knn__n_neighbors"]
knn_scores_dist = knn_grid_dist.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_scores_dist)
plt.xlabel("Número de vizinhos(K)")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("KNN (Weighted): Ajuste de parâmetro")
plt.grid(True, which="both")
plt.show()


# In[80]:


print(f"Melhor K: {knn_grid_dist.best_params_['knn__n_neighbors']}")
print(f"Melhor acurácia KNN: {knn_grid_dist.best_score_:.4f}\n")


# In[81]:


best_knn_dist = knn_grid_dist.best_estimator_
y_pred_knn_dist = cross_val_predict(best_knn_dist, X, y, cv=loo, n_jobs=-1)


# In[82]:


print(classification_report(y, y_pred_knn_dist, zero_division=0))


# In[83]:


cm_knn = confusion_matrix(y, y_pred_knn_dist)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN (Weighted)")
plt.show()


# ### 4.4 Random forest

# In[84]:


rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf_param_dist = {
    "max_depth": randint(2, 30),
}

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=15,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
)


# In[85]:


rf_random.fit(X, y)


# In[86]:


depth_vals = rf_random.cv_results_["param_max_depth"].data.astype(int)
rf_scores = rf_random.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(depth_vals, rf_scores)
plt.xlabel("Máxima Profundidade")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Random Forest: Ajuste de parâmetro")
plt.grid(True)
plt.show()


# In[87]:


print(f"Melhor Profundidade: {rf_random.best_params_['max_depth']}")
print(f"Melhor acurácia RF: {rf_random.best_score_:.4f}")


# Para o random forest, iremos usar o `balanced_subsample` que faz um balanceamento no sample coletado para cada árvore.

# In[88]:


best_rf = rf_random.best_estimator_
y_pred_rf_random = cross_val_predict(best_rf, X, y, cv=loo, n_jobs=-1)


# In[89]:


print(classification_report(y, y_pred_rf_random, zero_division=0))


# In[90]:


cm_rf = confusion_matrix(y, y_pred_rf_random)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN")
plt.show()


# In[91]:


rf_balanced = RandomForestClassifier(
    n_estimators=100, random_state=0, class_weight="balanced_subsample"
)

rf_random_balanced = RandomizedSearchCV(
    estimator=rf_balanced,
    param_distributions=rf_param_dist,
    n_iter=15,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
)


# In[92]:


rf_random_balanced.fit(X, y)


# In[98]:


depth_vals_balanced = rf_random_balanced.cv_results_["param_max_depth"].data.astype(int)
rf_scores_balanced = rf_random_balanced.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(depth_vals_balanced, rf_scores_balanced)
plt.xlabel("Máxima Profundidade")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Random Forest Balanceado: Ajuste de Parâmetro")
plt.grid(True)
plt.show()


# In[94]:


print(f"Melhor Profundidade: {rf_random_balanced.best_params_['max_depth']}")
print(f"Melhor acurácia RF: {rf_random_balanced.best_score_:.4f}")


# In[95]:


best_rf_balanced = rf_random_balanced.best_estimator_
y_pred_rf_random_balanced = cross_val_predict(best_rf_balanced, X, y, cv=loo, n_jobs=-1)


# In[96]:


print(classification_report(y, y_pred_rf_random_balanced, zero_division=0))


# In[97]:


cm_rf_balanced = confusion_matrix(y, y_pred_rf_random_balanced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf_balanced, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: Random Forest (Balanced)")
plt.show()

### BONUS
foo = pd.read_csv("editable_plants.csv", index_col=0)

drop_col = [
    "energy",
    "nutritional_info",
    "description",
    "requirements",
    "sensitivities",
]
foo = foo.drop(columns=drop_col)

foo["season"] = foo.season.str.lower().str.strip()
foo.season.unique()

season_map = {
    "annual": "annual",
    "biennial": "biennial",
    "biennial, grown as annual": "annual",
    "perennial": "perennial",
    "perrenial": "perennial",
    "annual/perannial": "mixed",
    "biennial, grown as an annual": "annual",
    "perrenial evergreen": "perenial",
    "semi-evergreen perrenial": "perenial",
    "shrub": np.nan,
}
foo["season"] = foo.season.replace(season_map)
foo["season"] = foo.season.fillna("unknown")


foo["ph_range"] = foo.preferred_ph_upper - foo.preferred_ph_lower
foo["ph"] = (foo.preferred_ph_upper + foo.preferred_ph_lower) / 2

foo = foo.drop(columns=["preferred_ph_lower", "preferred_ph_upper"])

foo.columns
foo.temperature_germination.unique()


def clean_temp_germ(value):
    if pd.isna(value) or value == "Not applicable.":
        return np.nan

    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    return float(value)


foo["temperature_germination"] = foo.temperature_germination.apply(clean_temp_germ)
foo["temperature_germination"] = foo.temperature_germination.fillna(
    foo.temperature_germination.mode()[0]
)

foo.temperature_growing.unique()


def clean_temp_grow(value):
    if pd.isna(value) or value == "nan":
        return np.nan

    if value.endswith("-"):
        return float(value.replace("-", ""))

    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    return float(value)


foo["temperature_growing"] = foo.temperature_growing.apply(clean_temp_grow)
foo["temperature_growing"] = foo.temperature_growing.fillna(
    foo.temperature_growing.mode()[0]
)

foo.days_germination.unique()


def clean_days_germ(value):
    if pd.isna(value) or value == "Not applicable.":
        return np.nan

    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    return float(value)


foo["days_germination"] = foo.days_germination.apply(clean_days_germ)
foo["days_germination"] = foo.days_germination.fillna(foo.days_germination.mode()[0])

foo.days_harvest.unique()


def clean_days_harv(value):
    if pd.isna(value) or value in ["x", "continual", "Not applicable."]:
        return np.nan

    if value.endswith("-"):
        return float(value.replace("-", ""))

    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    return float(value)


foo["days_harvest"] = foo.days_harvest.apply(clean_days_harv)
foo["days_harvest"] = foo.days_harvest.fillna(foo.days_harvest.mode()[0])

foo.soil.unique()


def clean_soil(value):
    if pd.isna(value):
        return []

    text = value.lower()

    if "all" in text or "most" in text:
        return ["sand", "loam", "clay", "chalk", "peat", "silt"]

    soils = ["sand", "loam", "clay", "chalk", "peat", "silt"]

    lst = []
    for soil in soils:
        if soil in value:
            lst.append(soil)

    return lst


foo["soil"] = foo.soil.apply(clean_soil)

mlb = MultiLabelBinarizer()
soil_binary = mlb.fit_transform(foo.soil)

foo = pd.concat([foo, pd.DataFrame(soil_binary, columns=mlb.classes_)], axis=1)
foo = foo.drop(columns="soil")


### old clean
foo.cultivation = foo.cultivation.str.lower().str.strip()
foo["cultivation"] = foo.cultivation.replace({"brassicas": "brassica"})


foo.sunlight = foo.sunlight.replace(
    {"full sun/partial shade/ full shade": "Full sun/partial shade/full shade"}
)
foo.sunlight = foo.sunlight.str.lower().str.strip()

foo.water = foo.water.str.lower().str.strip()

water_map = {
    "low": 1,
    "very low": 2,
    "medium": 3,
    "high": 4,
    "very high": 5,
}

foo["water_code"] = foo.water.map(water_map)

foo.nutrients = foo.nutrients.str.lower().str.strip()

mask = foo.nutrients == "high potassium fertiliser every 2 weeks."
foo.loc[mask, "nutrients"] = "high"


nutrients_map = {
    "low": 1,
    "medium": 2,
    "medium to high": 3,
    "high": 4,
}

foo["nutrients_code"] = foo.nutrients.map(nutrients_map)


foo.temperature_class = foo.temperature_class.str.lower().str.strip()

mask = foo.temperature_class == "very hard"
foo.loc[mask, "temperature_class"] = "very hardy"

temperature_class_map = {
    "very tender": 1,
    "tender": 2,
    "half hardy": 3,
    "hardy": 4,
    "very hardy": 5,
}

foo["temperature_class_code"] = foo.temperature_class.map(temperature_class_map)
### end old clean

foo = foo.drop(columns="taxonomic_name")
foo = foo.drop(columns=["water", "nutrients", "temperature_class"])

### dummies
foo = pd.get_dummies(
    foo,
    columns=[
        "cultivation",
        "sunlight",
        "season",
    ],
    drop_first=True,
)

foo.to_csv("clean_editable_plants_v2.csv")

### correlation
plt.figure(figsize=(10, 8))
corr = foo.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("corr heatmap")
plt.xticks(rotation=45)
plt.show()

### model

common_name = foo.common_name
foo = foo.drop(columns="common_name")

y_v2 = foo["water_code"]
X_v2 = foo.drop(
    columns=[
        "water_code",
    ]
)


log_pipe_v2 = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "log_reg",
            LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced"),
        ),
    ]
)

random_search_v2 = RandomizedSearchCV(
    estimator=log_pipe_v2,
    param_distributions=log_param_dist,
    n_iter=100,
    cv=loo,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
)


random_search_v2.fit(X, y)


c_values_v2 = random_search_v2.cv_results_["param_log_reg__C"].data.astype(float)
scores_v2 = random_search_v2.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(c_values_v2, scores_v2)
plt.xscale("log")
plt.xlabel("Parâmetro C")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Regressão Logística: Ajuste de parâmetro")
plt.grid(True, which="both", alpha=0.5)
plt.show()


print(
    f"Melhor parâmetro (C = 1/lambda): {random_search.best_params_['log_reg__C']:.4f}"
)
print(f"Lambda: {1 / random_search.best_params_['log_reg__C']:.4f}")
print(f"Melhor acurácia LOOCV: {random_search.best_score_:.4f}")


best_log_model_v2 = random_search_v2.best_estimator_
y_pred_log_v2 = cross_val_predict(best_log_model_v2, X_v2, y_v2, cv=loo, n_jobs=-1)

print(classification_report(y_v2, y_pred_log_v2, zero_division=0))


cm = confusion_matrix(y_v2, y_pred_log_v2)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de confusão: regressão logística")
plt.show()
