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
from functions import *

lipid_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
lipid_neg_dict = {'[M+CH3COO]-': 0, '[M+HCOO]-': 1, '[M+Na-2H]-': 2, '[M-CH3]-': 3, '[M-H]-': 4}
met_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
met_neg_dict = {'[M+Na-2H]-': 0, '[M-H]-': 1}
mode_dict = {'lipid_pos':lipid_pos_dict,'lipid_neg':lipid_neg_dict,'mets_pos':met_pos_dict,'mets_neg':met_neg_dict}

if __name__ == '__main__':
    input = sys.argv
    if len(input) < 2:
        print('please input data filename')    
    csv_data = input[1]
    mode = input[2]
    option = input[3]
    output_name = mode+'_'+csv_data.split('/')[-1].replace('.csv','')+'_'+option
    print(output_name)
    ccs_data = pd.read_csv(csv_data)
    ccs_data = ccs_data[ccs_data['Adduct'].isin(list(mode_dict[mode].keys()))]
    ccs_data.reset_index(inplace=True)
    df = encode_adduct(ccs_data,mode)
    feats_md = feature_generator(df)
    print(feats_md.shape)
    features,labels = generate_data_loader(df,feats_md,option)
    # Save training data and test data
    if option == 'train':
        features_data_save = './mpnn_data/ims/'+output_name+'_'+option+'_feats.pt'
        torch.save(features, features_data_save)
        label_data_save = './mpnn_data/ims/'+output_name+'_labels.pt'
        torch.save(labels, label_data_save)
    if option == 'predict':
        features_data_save = './mpnn_data/ims/'+output_name+'_'+option+'_feats.pt'
        torch.save(features, features_data_save)
        print('Prediction mode, no labels saved.')