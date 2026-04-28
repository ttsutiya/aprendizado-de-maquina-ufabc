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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

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

###

best_log_model = random_search.best_estimator_

y_pred_log = cross_val_predict(best_log_model, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_log, zero_division=0))

cm = confusion_matrix(y, y_pred_log)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de confusão: regressão logística")
plt.show()

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

random_search_balanced.fit(X, y)

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

print(
    f"Melhor parâmetro (C = 1/lambda): {random_search_balanced.best_params_['log_reg__C']:.4f}"
)
print(f"Lambda: {1 / random_search_balanced.best_params_['log_reg__C']:.4f}")
print(f"Melhor acurácia LOOCV: {random_search_balanced.best_score_:.4f}")

best_log_balanced = random_search_balanced.best_estimator_
y_pred_log_balanced = cross_val_predict(best_log_balanced, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_log_balanced, zero_division=0))

cm_balanced = confusion_matrix(y, y_pred_log_balanced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_balanced, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de confusão: regressão logística (balanceada)")
plt.show()

###

best_knn = knn_grid.best_estimator_
y_pred_knn = cross_val_predict(best_knn, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_knn, zero_division=0))

cm_knn = confusion_matrix(y, y_pred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN")
plt.show()

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

knn_grid_dist.fit(X, y)

k_values = knn_param_grid["knn__n_neighbors"]
knn_scores_dist = knn_grid_dist.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_scores_dist)
plt.xlabel("Número de vizinhos(K)")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("KNN (Weighted): Ajuste de parâmetro")
plt.grid(True, which="both")
plt.show()

print(f"Melhor K: {knn_grid_dist.best_params_['knn__n_neighbors']}")
print(f"Melhor acurácia KNN: {knn_grid_dist.best_score_:.4f}\n")

best_knn_dist = knn_grid_dist.best_estimator_
y_pred_knn_dist = cross_val_predict(best_knn_dist, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_knn_dist, zero_division=0))

cm_knn = confusion_matrix(y, y_pred_knn_dist)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN (Weighted)")
plt.show()

###

best_rf = rf_random.best_estimator_
y_pred_rf_random = cross_val_predict(best_rf, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_rf_random, zero_division=0))

cm_rf = confusion_matrix(y, y_pred_rf_random)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: KNN")
plt.show()
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

rf_random_balanced.fit(X, y)


depth_vals_balanced = rf_random_balanced.cv_results_["param_max_depth"].data.astype(int)
rf_scores_balanced = rf_random_balanced.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.scatter(depth_vals_balanced, rf_scores_balanced)
plt.xlabel("Máxima Profundidade")
plt.ylabel("Acurácia Média (LOOCV)")
plt.title("Random Forest Balanceado: Ajuste de Parâmetro")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
print(f"Melhor Profundidade: {rf_random_balanced.best_params_['max_depth']}")
print(f"Melhor acurácia RF: {rf_random_balanced.best_score_:.4f}")

best_rf_balanced = rf_random_balanced.best_estimator_
y_pred_rf_random_balanced = cross_val_predict(best_rf_balanced, X, y, cv=loo, n_jobs=-1)

print(classification_report(y, y_pred_rf_random_balanced, zero_division=0))

cm_rf_balanced = confusion_matrix(y, y_pred_rf_random_balanced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf_balanced, annot=True)
plt.xlabel("Predição Water_code")
plt.ylabel("Valor Water_code")
plt.title("Matriz de Confusão: Random Forest (Balanced)")
plt.show()
