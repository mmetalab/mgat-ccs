import sys
import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import rdBase
from collections import OrderedDict
from sklearn import preprocessing
import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdDescriptors
from descriptastorus.descriptors import rdNormalizedDescriptors
from sklearn.model_selection import StratifiedKFold
from pandas.api.types import CategoricalDtype

lipid_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
lipid_neg_dict = {'[M+CH3COO]-': 0, '[M+HCOO]-': 1, '[M+Na-2H]-': 2, '[M-CH3]-': 3, '[M-H]-': 4}
met_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
met_neg_dict = {'[M+Na-2H]-': 0, '[M-H]-': 1}
mode_dict = {'lipid_pos':lipid_pos_dict,'lipid_neg':lipid_neg_dict,'mets_pos':met_pos_dict,'mets_neg':met_neg_dict}


# Calculate Molecular descriptor   
def smile_to_md(smile):
    result = []
    gen = rdNormalizedDescriptors.RDKit2DNormalized()
    md = gen.process(smile)
    t = md[1:]
    result = np.asarray(t)
    nan_index = np.argwhere(np.isnan(result))
    for i in nan_index:
        result[i[0]] = 0  
    return result


def encode_adduct(df,mode):
    mode_mappings = mode_dict[mode]
    le = preprocessing.LabelEncoder()
    le.fit(list(mode_mappings.keys()))
    num_adduct = le.transform(df['Adduct'].tolist())
    df['num_adduct'] = num_adduct
    return df

def one_encode_adduct(df,mode):
    mode_mappings = mode_dict[mode]
    df["Adduct"] = df["Adduct"].astype(CategoricalDtype(list(mode_mappings.keys())))
    dummy = pd.get_dummies(df["Adduct"],prefix='Adduct')
    df = pd.concat([df,dummy],axis=1)
    return df

def finger_print(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

def feature_generator(df):
    code_adduct = df['num_adduct'].values.reshape(-1,1)
    code_class = df['Class'].values.reshape(-1,1)
# Calculate fingerprint disabled in mgat-ccs model
    # fp_ccs = []
    # for smi in df['SMI'].tolist():
    #     t = finger_print(smi)
    #     fp_ccs.append(t)
    # fp_ccs = np.asarray(fp_ccs)
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
    # feats_fp = np.hstack((fp_ccs, code_adduct, code_class)) disabled in mgat-ccs model
    feats_md = np.hstack((md_ccs, code_adduct, code_class))
    return feats_md

def generate_data_loader(df,feats_md, option='train'):
    feats = []
    labels = []
    for index,row in df.iterrows():
        name = row['Name']
        adduct_code = row['num_adduct']
        smiles = row['SMI']
        adduct_type = row['Adduct']
        class_code = row['Class']
        md_adduct_ccs = feats_md[index]
        mol_info = [index,name,smiles,adduct_code,adduct_type,class_code]
        # fp_adduct_ccs = feats_fp[index]
        feats.append([mol_info,smiles,md_adduct_ccs])
    if option == 'train':
        labels = df['CCS'].values
    else:
        labels = []
    return feats,labels

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)