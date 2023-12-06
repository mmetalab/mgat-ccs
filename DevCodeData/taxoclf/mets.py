# usr/bin/env python

import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *

hmdb = pd.read_csv('../data/hmdb.csv')
hmdb = hmdb[~hmdb['smiles'].isna()]

hmdb_mets = hmdb[hmdb['super_class']!='Lipids and lipid-like molecules']
hmdb_mets = hmdb_mets[~hmdb_mets['super_class'].isna()]
hmdb_mets = hmdb_mets[~hmdb_mets['smiles'].isna()]
main_met_class = list(hmdb_mets['super_class'].value_counts()[:9].keys())
hmdb_mets['label'] = hmdb_mets['super_class'] 
hmdb_mets.loc[hmdb_mets[~hmdb_mets['super_class'].isin(main_met_class)].index,'label'] = 'Others'

hmdb_mets_df = pd.DataFrame(columns=['smile','md','label'])
error_smi = []
index = 0
for i, row in hmdb_mets.iterrows():
    smi = row['smiles']
    try:
        t = smile_to_md(smi)
        hmdb_mets_df.loc[index,'smile'] = smi
        hmdb_mets_df.loc[index,'md'] = t
        hmdb_mets_df.loc[index,'label'] = row['label']
        index += 1
    except:
        print(smi)
        error_smi.append(index)

hmdb_mets_df.to_csv('../output/hmdb_mets_df.csv',index=False)