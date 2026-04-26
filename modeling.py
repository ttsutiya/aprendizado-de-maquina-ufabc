import pandas as pd

df = pd.read_csv("clean_editable_plants.csv")


df = df.drop(
    columns=[
        "taxonomic_name",
        "common_name",
        "water",
        "nutrients",
        "temperature_class",
    ]
)


x = df.drop(columns=["water_code"])
y = df["water_code"]
