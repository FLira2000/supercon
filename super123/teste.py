import os
import pandas as pd

def xpto():
    return f"./super123Database_final"

df = pd.read_csv(xpto())
print(df)