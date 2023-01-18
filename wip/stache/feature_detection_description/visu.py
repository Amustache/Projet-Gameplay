import csv


import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("../outputs/coordinates.csv")
df = df[["x", "y"]]
print(df.head())

df.plot(x="x", y="y")
plt.show()
