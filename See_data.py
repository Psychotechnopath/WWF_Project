from WWF_Project.deia2_general import set_path_base, to_dataframe
import numpy as np
import pandas as pd

#%%

## I'll check Pickles => [1, 4, 9, 16, 25]
path = set_path_base('Joost')
data = to_dataframe(f"{path}/TilePickle_25.pkl")

#%%
pd.set_option('display.max.columns', None)
print(data.shape)
# print(data.head(10))
print(data.info)

#%%
file = open('pickle25data.txt', 'w')
for column in data.columns:
    file.write(f"\n{column}\n")
    file.write(str(data[column].value_counts()))
    file.write(f"\nAmount of NaN in this column ({column}) = {str(data[column].isna().sum())}")
    file.write("\n________________________________________________________________________")
file.close()

#%%
# GENERAL COMMENTS
# PalmOilConcession varies a lot between the pickles


# Strange things FOR PICKLE 1
# 9 Million missings in scaledASTER
# All missings for gradienstASTER
# EdgeDensity3 all values are 0.0
# AggIndex 3 all values are 0.0
# PatchDensity3 all values are 1.0
# Landcoverpercentage3 all values are 0.0
# Lots of missings in SarVision variables


# Strange things FOR PICKLE 4
# CoastlineDistance
# No missings for Sarvision
# Looks clean (no missings)

# Strange things FOR PICKLE 9
# EdgeDensity3 all values are 0.0
# AggIndex 3 all values are 0.0
# PatchDensity3 all values are 1.0
# Landcoverpercentage3 all values are 0.0
# Lots of missings in SarVision variables

# Strange things FOR PICKLE 16
# Ralatively few missings for SarVision variables (+/- 100.000)
# Looks clean

# Strange things FOR PICKLE 25
# Looks clean