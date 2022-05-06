import pandas as pd
import os
import glob

#read cadec dataset
datasets = []
for file in glob.glob("AMT-SCT/*.ann"):
    try:
        cadec_df = pd.read_csv(file, sep="\t", header=None)
        cadec_df[1] = cadec_df[1].str.split('|').str[1]
        cadec_df[1] = cadec_df[1].str.strip()
        cadec_df = cadec_df[[1,2]]
        cadec_df.columns = ["concepts", "phrases"]
        
        datasets.append(cadec_df)
    except:
        print("{0} is empty dataset".format(file))
    
cadec_df = pd.concat(datasets)
cadec_df.dropna(inplace=True)
cadec_df.to_csv("cadec_dataset.txt", sep='|', index=False)
print(cadec_df)

