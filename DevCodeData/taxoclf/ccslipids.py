# usr/bin/env python

import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *

ccsbase = pd.read_csv('../data/CCSDB.csv')
ccsbase_lipid = ccsbase[ccsbase['Type'] == 'lipid']
ccs_lipid_df = ccsbase_lipid.copy()
ccs_lipid_df['md'] = np.nan
ccs_lipid_df.reset_index(inplace=True)
ccs_lipid_df = ccs_lipid_df.astype(object)  # this line allows the signment of the array

for i, row in ccs_lipid_df.iterrows():
    smi = row['SMI']
    try:
        t = smile_to_md(smi)
        ccs_lipid_df.loc[i,'md'] = t
        print('finished %d out of %d' % (i,len(ccs_lipid_df)))
    except:
        print(smi)
        pass

ccs_lipid_df.to_csv('../output/ccs_lipid_df.csv',index=False)