import pandas as pd

editable_plants = pd.read_csv("editable_plants.csv", index_col=0)
len(editable_plants)

editable_plants.columns

editable_plants.iloc[0, :]


for col in editable_plants.columns:
    num = len(editable_plants[col].unique())
    print(f"name: {col:<25} num: {num:<5}")


# Por que temos taxmonomic_name < common_name?
# Espera-se que fossem iguais
mask = editable_plants.duplicated(subset=["taxonomic_name"], keep=False)
index = editable_plants[mask].sort_values("taxonomic_name").index
len(index)

editable_plants.loc[index]

editable_plants.loc[editable_plants.taxonomic_name == "Allium cepa"]

editable_plants.loc[editable_plants.taxonomic_name == "Brassica oleracea"]


# Nao gostaria de remover todas as 27 instancias ja que representaria cerca de ~20% dos dados totais.
# Mas nota-se que no cabbage e no onion temos pouca diferencas nos dados.
# Dito isso, iremos pegar apenas uma instancia (a primeira) e remover o resto para nao enviesar a nossa analise.
mask = editable_plants.duplicated(subset=["taxonomic_name"], keep="first")
editable_plants = editable_plants[~mask]
len(editable_plants)


editable_plants.soil.unique()

# O que eh um solo well-drained e fertile? Sao propriedades de qualquer solo?
# ex. loamy pode ser fertile e well-drained ou fertile eh um subset dos solo?
# Devido a tipos de solos como "high in organic matter", "rich-moist soil", "peat-rich soil" decidimos
# nao usar essa coluna.
editable_plants = editable_plants.drop(columns=["soil"])


editable_plants.iloc[0:5][
    ["sensitivities", "nutritional_info", "description", "requirements"]
]

editable_plants.iloc[4].sensitivities

# Devido a natureza descritiva dessas colounas, vamos remove-las tambem

editable_plants = editable_plants.drop(
    columns=[
        "nutritional_info",
        "description",
        "requirements",
        "sensitivities",
    ]
)

# Vamos checar quais dados estao faltando.
check_null = editable_plants.isnull().sum()
check_null / len(editable_plants)


# Algumas colunas estao completas, enquanto algumas faltam um numero consideravel de dados.
# Como nao sao dados que podemos preencher usando outras colunas, vamos apenas remove-las.
drop_cols = check_null[check_null > 0].index
editable_plants = editable_plants.drop(columns=drop_cols)

editable_plants.to_csv("clean_editable_plants.csv")

# Perguntas originais
# Do plants that require more sunlight also require higher tempeteratures?
# What cultivation classes require the most water?
