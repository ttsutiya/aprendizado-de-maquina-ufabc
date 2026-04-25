import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2026/2026-02-03/edible_plants.csv"
)

df.to_csv("editable_plants.csv")
