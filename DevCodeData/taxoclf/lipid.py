# usr/bin/env python

import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *

hmdb = pd.read_csv('../data/hmdb.csv')
hmdb = hmdb[~hmdb['smiles'].isna()]
hmdb.head()

hmdb_lipid = hmdb[hmdb['super_class']=='Lipids and lipid-like molecules']

main_lipid_class = list(hmdb_lipid['class'].value_counts()[:6].keys())
hmdb_lipid['label'] = hmdb_lipid['class'] 
hmdb_lipid.loc[hmdb_lipid[~hmdb_lipid['class'].isin(main_lipid_class)].index,'label'] = 'Others'

hmdb_lipid_df = pd.DataFrame(columns=['smile','md','label'])
error_smi = []
index = 0
for i, row in hmdb_lipid.iterrows():
    smi = row['smiles']
    try:
        t = smile_to_md(smi)
        hmdb_lipid_df.loc[index,'smile'] = smi
        hmdb_lipid_df.loc[index,'md'] = t
        hmdb_lipid_df.loc[index,'label'] = row['label']
        index += 1
        print('finished %d out of %d' %(index, len(hmdb_lipid)))
    except:
        print(smi)
        error_smi.append(index)

hmdb_lipid_df.to_csv('../output/hmdb_lipid_df.csv',index=False)
