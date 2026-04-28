#!/usr/bin/env python
# coding: utf-8

# # Projeto Final Aprendizado de Máquina
# ## Aplicação de Modelos de Classificação
# 
# integrantes
# 
# - Thiago Hideki Tsutiya - 11201811680  
# - Thiago de Lima Silva - 11202020228  
# - João Pedro Genga Carneiro - 11201810740  
# - Vinicius Hideo Miyake - 11201920257  
# 
# https://github.com/ttsutiya/aprendizado-de-maquina-ufabc

# ## Análise de Plantas Comestíveis
# 
# ## Objetivo
# 
# Este projeto tem como objetivo analisar características de plantas comestíveis e construir modelos capazes de prever a necessidade de água com base em outras variáveis.
# 
# ## Pipeline do projeto
# 
# O projeto foi dividido em 4 etapas principais:
# 
# 1. Coleta dos dados
# 2. Limpeza e pré-processamento
# 3. Análise exploratória
# 4. Modelagem preditiva

# ## 1. Coleta dos dados
# 
# 
# Os dados utilizados neste projeto foram obtidos a partir do dataset [Edibe Plants Database](https://rfordatascience.github.io/tidytuesday/data/2026/2026-02-03/readme.html)
# 
# A base contém informações sobre plantas comestíveis, incluindo características como:
# - necessidade de água
# - nutrientes requeridos
# - classe de temperatura
# - grupo de cultivação

# In[1]:


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2026/2026-02-03/edible_plants.csv")
df.to_csv("editable_plants.csv")


# ## 2. Limpeza e processamento
# 

# In[3]:


editable_plants = pd.read_csv("editable_plants.csv", index_col=0)
len(editable_plants)


# In[4]:


editable_plants.columns


# In[5]:


editable_plants.iloc[0, :]


# In[6]:


for col in editable_plants.columns:
    num = len(editable_plants[col].unique())
    print(f"name: {col:<25} num: {num:<5}")


# Foi identificado que algumas plantas possuíam o mesmo `taxonomic_name`, mas contendo um `common_name` diferente e.g. Onion, Onion (Autumn Planted), Onion (Red).
# 
# Ao invés de remover todos os registros duplicados (o que causaria perda significativa de dados), optou-se por manter apenas a primeira ocorrência, visto que a maioria das instâncias contiam o mesmo atributo, ou variam muito pouco entre si.

# In[7]:


mask = editable_plants.duplicated(subset=["taxonomic_name"], keep=False)
index = editable_plants[mask].sort_values("taxonomic_name").index
len(index)


# In[8]:


editable_plants.loc[index]


# In[9]:


editable_plants.loc[editable_plants.taxonomic_name == "Allium cepa"]


# In[10]:


editable_plants.loc[editable_plants.taxonomic_name == "Brassica oleracea"]


# Iremos manter somente a primeira instancia de cada `taxonomic_name`.

# In[13]:


mask = editable_plants.duplicated(subset=["taxonomic_name"], keep="first")
editable_plants = editable_plants[~mask]
len(editable_plants)


#  ### 2.2 Remoção de colunas irrelevantes
# 
# Colunas textuais como `description`, `nutritional_info`, `sensitivities` e `requirements` foram removidas, pois possuem natureza descritiva e não estruturada, dificultando sua utilização em modelos tradicionais.
# 

# In[14]:


editable_plants.iloc[0:5][
    ["sensitivities", "nutritional_info", "description", "requirements"]
]


# In[15]:


editable_plants.iloc[4].sensitivities


# In[16]:


editable_plants = editable_plants.drop(
    columns=[
        "nutritional_info",
        "description",
        "requirements",
        "sensitivities",
    ]
)


# In[17]:


editable_plants.soil.unique()


# O que eh um solo well-drained e fertile? Sao propriedades de qualquer solo?
# Devido ao tipos de solos como "high in organic matter", "rich-moist soil", "peat-rich soil", entre outros decidimos não usar essa coluna.

# In[18]:


editable_plants = editable_plants.drop(columns=["soil"])


# ## 2.3 Tratamento de valores ausentes
# 
# Foi realizada uma verificação de valores nulos no dataset. Como não havia uma estratégia confiável para imputação desses dados com base nas demais variáveis, optou-se pela remoção das colunas que apresentavam valores ausentes.

# In[19]:


check_null = editable_plants.isnull().sum()
check_null / len(editable_plants)


# Algumas colunas estão completas, enquanto outras apresentam uma falta considerável de dados. Como não é possível preenchê-las utilizando as demais colunas, vamos simplesmente removê-las.

# In[21]:


drop_cols = check_null[check_null > 0].index
editable_plants = editable_plants.drop(columns=drop_cols)


# ## 2.4 Transformação de variáveis categóricas
# 
# Variáveis categóricas foram convertidas para formato numérico, permitindo sua utilização em modelos de machine learning.
# 
# Para variáveis sem ordem natural, foi utilizado o método de one-hot encoding. Já para variáveis ordinais, foi realizado um mapeamento manual, preservando a relação de ordem entre os valores.

# In[22]:


editable_plants.columns


# In[23]:


editable_plants.iloc[0]


# In[24]:


editable_plants.cultivation.unique()


# In[25]:


editable_plants.cultivation = editable_plants.cultivation.str.lower().str.strip()
editable_plants["cultivation"] = editable_plants.cultivation.replace(
    {"brassicas": "brassica"}
)


# In[26]:


editable_plants = pd.get_dummies(
    editable_plants, columns=["cultivation"], drop_first=True
)


# In[27]:


editable_plants.sunlight.unique()


# In[28]:


editable_plants.sunlight = editable_plants.sunlight.replace(
    {"full sun/partial shade/ full shade": "Full sun/partial shade/full shade"}
)


# In[29]:


editable_plants.sunlight = editable_plants.sunlight.str.lower().str.strip()
editable_plants = pd.get_dummies(editable_plants, columns=["sunlight"], drop_first=True)


# ## 2.5 Codificação de variáveis ordinais
# 
# Variáveis como water, nutrients e temperature_class possuem uma ordem natural entre seus valores (por exemplo: baixo, médio, alto).
# 
# Para preservar essa relação, foi realizado um mapeamento manual para valores numéricos.
# 
# Essa abordagem permite que os modelos capturem corretamente a progressão entre os níveis, ao contrário de técnicas como one-hot encoding, que tratariam cada valor como independente.

# In[31]:


editable_plants.water = editable_plants.water.str.lower().str.strip()
editable_plants.water.unique()


# In[32]:


water_map = {
    "low": 1,
    "very low": 2,
    "medium": 3,
    "high": 4,
    "very high": 5,
}

editable_plants["water_code"] = editable_plants.water.map(water_map)


# In[33]:


editable_plants.nutrients = editable_plants.nutrients.str.lower().str.strip()
editable_plants.nutrients.unique()


# In[34]:


mask = editable_plants.nutrients == "high potassium fertiliser every 2 weeks."
editable_plants.loc[mask, "nutrients"] = "high"


# In[35]:


nutrients_map = {
    "low": 1,
    "medium": 2,
    "medium to high": 3,
    "high": 4,
}

editable_plants["nutrients_code"] = editable_plants.nutrients.map(nutrients_map)


# In[38]:


editable_plants.temperature_class = editable_plants.temperature_class.str.lower().str.strip()
editable_plants.temperature_class.unique()


# In[39]:


mask = editable_plants.temperature_class == "very hard"
editable_plants.loc[mask, "temperature_class"] = "very hardy"


# In[40]:


temperature_class_map = {
    "very tender": 1,
    "tender": 2,
    "half hardy": 3,
    "hardy": 4,
    "very hardy": 5,
}

editable_plants["temperature_class_code"] = editable_plants.temperature_class.map(
    temperature_class_map
)


# In[41]:


editable_plants.to_csv("clean_editable_plants.csv", index=False)


# ## 3 Análise Exploratória
# 
# A análise exploratória foi realizada com o objetivo de compreender a distribuição das variáveis e identificar possíveis relações entre elas.
# 
# Foram utilizados gráficos para visualizar padrões e tendências no conjunto de dados.

# ## Distribuição de Variáveis

# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[43]:


df = pd.read_csv("clean_editable_plants.csv")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="water_code")
plt.title("distribuicao water_code")
plt.show()


# In[44]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="nutrients_code")
plt.title("distribuicao nutrients_code")
plt.show()


# In[45]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="temperature_class_code")
plt.title("distribuicao temperature_class_code")
plt.show()


# ### Correlação entre variáveis

# In[48]:


plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("corr heatmap")
plt.xticks(rotation=45)
plt.show()


# ### Relação com variável alvo

# In[49]:


sns.boxplot(data=df, x="temperature_class_code", y="water_code")
plt.title("Relação entre temperatura e necessidade de água")
plt.show()


# ## 4 Modelagem
# O objetivo desta etapa é construir modelos capazes de prever a necessidade de água (water_code) com base nas demais variáveis.
# 
# ### Modelos utilizados:
# - Regressão Logística
# - K-Nearest Neighbors (KNN)
# - Random Forest
# 
# ### Validação:
# - Leave-One-Out Cross Validation (LOOCV)

# In[51]:


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

