# import libraries
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn import preprocessing
from descriptastorus.descriptors import rdNormalizedDescriptors
import pickle5 as pickle


abbr_mode = {'Lipid positive mode':'lipid_pos',
             'Lipid negative mode':'lipid_neg',
             'Metabolite positive mode':'met_pos',
             'Metabolite negative mode':'met_neg',
             'Drug mode (beta)':'drug_pos'}


def encode_adduct(df,mode_mappings):
    le = preprocessing.LabelEncoder()
    le.fit(list(mode_mappings.keys()))
    num_adduct = le.transform(df['Adduct'].tolist())
    df['Encode adduct'] = num_adduct
    return df

def finger_print(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

def feature_generator(df):
    num_adduct = df['Encode adduct'].values.reshape(-1,1)
# Calculate fingerprint
    fp_ccs = []
    for smi in df['SMI'].tolist():
        t = finger_print(smi)
        fp_ccs.append(t)
    fp_ccs = np.asarray(fp_ccs)
# Calculate Molecular descriptor   
    md_ccs = []
    gen = rdNormalizedDescriptors.RDKit2DNormalized()
    for smi in df['SMI'].tolist():
        md = gen.process(smi)
        t = md[1:]
        md_ccs.append(t)
    md_ccs = np.asarray(md_ccs)
    nan_index = np.argwhere(np.isnan(md_ccs))
    for i in nan_index:
        md_ccs[i[0],i[1]] = 0   
    feats_fp = np.hstack((fp_ccs, num_adduct))
    feats_md = np.hstack((md_ccs, num_adduct))
    return num_adduct,feats_fp,feats_md

def generate_data_loader(df,feats_md,feats_fp,opt='train'):
    feats = []
    labels = []
    for index,row in df.iterrows():
        smiles = row['SMI']
        md_adduct_ccs = feats_md[index]
        fp_adduct_ccs = feats_fp[index]
        feats.append([index,smiles,md_adduct_ccs,fp_adduct_ccs])
    if opt == 'train':
        labels = df['CCS'].values
    else:
        labels = []
    return feats,labels

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def predict_ccs(features,mode,df_f):
    # feats = []
    # for t in features:
    #     feats.append(t[3])
    # feats = np.array(feats)
    # model_file = './Models/'+abbr_mode[mode]+'_fingerprint.sav'
    # model = pickle.load(open(model_file, 'rb'))
    # result = model.predict(feats)
    # df_f['ccs_fp'] = result
    feats = []
    for t in features:
        feats.append(t[2])
    feats = np.array(feats)
    model_file = './models/'+abbr_mode[mode]+'_descriptor.sav'
    model = pickle.load(open(model_file, 'rb'))
    result = model.predict(feats)
    df_f['ccs_md'] = result
    return df_f