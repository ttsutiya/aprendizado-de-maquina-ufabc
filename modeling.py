from statistics import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("clean_editable_plants.csv")

# Guardamos os nomes em uma variável separada para referência futura
nomes_plantas = df['common_name']

# Definimos X excluindo as colunas de identificação e as que já transformamos em valores numéricos, e o alvo (y)
X = df.drop(columns=['water_code', 'taxonomic_name', 'common_name', 'water', 'nutrients', 'temperature_class'])
y = df['water_code']

# Guardamos os índices durante a divisão de treino e teste
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Loop de treino e avaliação
# MSE = Erro quadrático médio
# R2 Score = Coeficiente de determinação, que indica a proporção da variância dos dados que é explicada pelo modelo. Quanto mais próximo de 1 (100%), melhor o modelo se encaixa aos dados.
print(f"{'Modelo':<20} | {'MSE':<10} | {'R2 Score':<10}")
print("-" * 40)

for nome, modelo in models.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    # Reccebem o y_test e as predições do modelo para calcular as métricas de avaliação
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    print(f"{nome:<20} | {mse:<10.4f} | {r2:<10.4f}")