# usr/bin/env python

import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *

ccsbase = pd.read_csv('../data/CCSDB.csv')
ccsbase_mets = ccsbase[ccsbase['Type'] != 'lipid']
ccs_mets_df = ccsbase_mets.copy()
ccs_mets_df['md'] = np.nan
ccs_mets_df = ccs_mets_df.reset_index()
ccs_mets_df = ccs_mets_df.astype(object)  # this line allows the signment of the array

for i, row in ccs_mets_df.iterrows():
    smi = row['SMI']
    try:
        t = smile_to_md(smi)
        ccs_mets_df.loc[i,'md'] = t
        print('finished %d out of %d' % (i,len(ccs_mets_df)))
    except:
        print(smi)
        pass

ccs_mets_df.to_csv('../output/ccs_mets_df.csv',index=False)