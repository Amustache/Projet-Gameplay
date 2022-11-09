import csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../outputs/coordinates.csv")
df = df[["x", "y"]]
print(df.head())

df.plot(x="x", y="y")
plt.show()
