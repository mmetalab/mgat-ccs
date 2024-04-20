# usr/bin/env python

import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *
import pickle

moldata = pd.read_csv('../data/example_data.csv')
moldata['md'] = moldata['SMI'].apply(lambda x: smile_to_md(x))

moldata = moldata[~moldata['md'].isna()] 

# Load the label encoder
with open("./lipid_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
    
with open("./lipid_model_encoder.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoder
with open("./mets_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
    
with open("./mets_model_encoder.pkl", "rb") as f:
    model = pickle.load(f)

X = np.asarray([np.asarray(i) for i in moldata['md'].values])
y = model.predict(X)
y_label = le.inverse_transform(y)

moldata['Class label'] = y_label
moldata['Class'] = y

print(moldata.head())




t_dict = {'Steroids and steroid derivatives':'ST',
 'Fatty Acyls':'FA',
 'Glycerophospholipids':'GP',
 'Prenol lipids':'PR',
 'Sphingolipids':'SP',
 'Glycerolipids':'GL',
 'Others':'OT'}

t = list(le.inverse_transform([0,1,2,3,4,5,6]))


t = list(le.inverse_transform([0,1,2,3,4,5,6,7,8,9]))

t_dict = {'Alkaloids and derivatives':'AKA', 
        'Benzenoids':'BZO',
        'Nucleosides, nucleotides, and analogues':'NCL',
        'Organic acids and derivatives':'OAC',
        'Organic nitrogen compounds':'ONC',
        'Organic oxygen compounds':'OOC',
        'Organoheterocyclic compounds':'OCL',
        'Organosulfur compounds':'OSC',
        'Others':'OTH',
        'Phenylpropanoids and polyketides':'PHP'}

print(t)





