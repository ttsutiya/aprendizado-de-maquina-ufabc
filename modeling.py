import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("clean_editable_plants.csv")

# Guardamos os nomes em uma variável separada para referência futura
nomes_plantas = df["common_name"]


df = df.drop(
    columns=[
        "taxonomic_name",
        "common_name",
        "water",
        "nutrients",
        "temperature_class",
    ]
)


# Definimos X excluindo as colunas de identificação e as que já transformamos em valores numéricos, e o alvo (y)
y = df["water_code"]
X = df.drop(
    columns=[
        "water_code",
    ]
)

loo = LeaveOneOut()

# LOGISTIC REGRESSION
# Criamos uma pipeline para padronizar os dados antes de rodar o LOO.
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

random_search.fit(X, y)

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

print(
    f"Melhor parâmetro (C = 1/lambda): {random_search.best_params_['log_reg__C']:.4f}"
)
print(f"Lambda: {1 / random_search.best_params_['log_reg__C']:.4f}")
print(f"Melhor acurácia LOOCV: {random_search.best_score_:.4f}")

# KNN REGRESSION
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

knn_grid.fit(X, y)

k_values = knn_param_grid["knn__n_neighbors"]
knn_scores = knn_grid.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_scores)
plt.xlabel("Número de vizinhos(K)")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("KNN: Ajuste de parâmetro")
plt.grid(True, which="both")
plt.show()

print(f"Melhor K: {knn_grid.best_params_['knn__n_neighbors']}")
print(f"Melhor acurácia KNN: {knn_grid.best_score_:.4f}\n")

# Random forest
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

rf_random.fit(X, y)

depth_vals = rf_random.cv_results_["param_max_depth"].data.astype(int)
rf_scores = rf_random.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(depth_vals, rf_scores)
plt.xlabel("Máxima Profundidade")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Random Forest: Ajuste de parâmetro")
plt.grid(True)
plt.show()

print(f"Melhor Profundidade: {rf_random.best_params_['max_depth']}")
print(f"Melhor acurácia RF: {rf_random.best_score_:.4f}")
