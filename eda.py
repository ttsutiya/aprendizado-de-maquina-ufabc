import pandas as pd

df = pd.read_csv("clean_editable_plants.csv")


# Perguntas originais
# Do plants that require more sunlight also require higher temperatures?
# What cultivation classes require the most water?

df.iloc[0]

df.columns

quant = ["prefered_ph_lower", "prefered_ph_upper"]
quali = []
