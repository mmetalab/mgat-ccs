# import libraries
import numpy as np
import pandas as pd
from rdkit import Chem
import pickle
import torch.nn as nn
from torch.autograd import Variable
from dgllife.utils import atom_degree, atomic_number, atom_explicit_valence, atom_formal_charge, atom_num_radical_electrons, atom_mass, BaseAtomFeaturizer
from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot
from dgllife.utils import ConcatFeaturizer
from dgllife.utils import BaseAtomFeaturizer
from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
from rdkit import Chem
from collections import OrderedDict
import sys
import dgl
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import rdBase
from collections import OrderedDict
from sklearn import preprocessing
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

def load_checkpoint(checkpoint):
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def construct_multigraph(smile):
    atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number, atom_explicit_valence, atom_formal_charge, atom_num_radical_electrons])
    mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
    bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot])
    mol_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})
    g = OrderedDict({}) # Store neighbor information of each node
    h = OrderedDict({}) # Hidden state for each node
    molecule = Chem.MolFromSmiles(smile)
    adj_mol = Chem.GetAdjacencyMatrix(molecule)
    adj_mol = torch.from_numpy(adj_mol)
    atom_feats = mol_atom_featurizer(molecule)['h']
    bond_feats = mol_bond_featurizer(molecule)['e']
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        h[i] = Variable(torch.FloatTensor(atom_feats[i]).view(1, 5))
        for j in range(molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)
            if e_ij != None:
                p = e_ij.GetIdx()
                e_ij = Variable(torch.FloatTensor(bond_feats[2*p]).view(1,11))
                atom_j = molecule.GetAtomWithIdx(j)
                if i not in g:
                    g[i] = []
                g[i].append( (e_ij, j) )
    return adj_mol, g, h

class MPLayer(nn.Module):
    def __init__(self):
        super(MPLayer, self).__init__()
        self.V = nn.Linear(5,5)
        self.U = nn.Linear(21,5)
        self.E = nn.Linear(11,11)
    def forward(self, g, h):
        for v in g.keys():
            neighbors = g[v]
            for neighbor in neighbors:
                e_vw = neighbor[0] # neighbor atom edge feature variable
                w = neighbor[1] # neighbor atom 
                m_w = self.V(h[w]) # outputsize (,74) size of node
                m_e_vw = self.E(e_vw) # outputsize (,12) size of edge 
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1) # outputsize (,74+74+12) nodesize + nodesize + edgesize
                h[v] = self.U(reshaped)
        return g, h
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(233, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.layer1 = MPLayer()
        self.layer2 = MPLayer()
        self.R = nn.Linear(10, 32)
    def forward(self, adducts, adj, g, h, h2):
        g, h = self.layer1(g,h)
        g, h = self.layer2(g,h)
        N = len(h.keys())
        catted = Variable(torch.zeros(N, 10))
        keys = list(h.keys())
        for i in range(N):
            k = keys[i]
            catted[i] = torch.cat([h[k], h2[k]], 1)
        activated_reads = self.R(catted)
        readout = torch.mean(activated_reads, dim=0)
        readout = torch.cat((readout,adducts),dim=-1)
        x = torch.relu(readout)
        x = self.fc1(x)
        embedding = x
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return embedding,x 
    
def model_embed_ccs(model,test_data_loader):
    model.eval()
    feats_embed = []
    feats_indices = []
    with torch.no_grad():
        for i in test_data_loader:
            test_index,test_smiles,test_md = i[0],i[1],i[2]
            smile = test_smiles
            feats_md = test_md
            adj1, g, h = construct_multigraph(smile) 
            h2 = h
            embedding,output = model(feats_md,adj1,g,h,h2)
            feats_embed.append(embedding.data.numpy())
            feats_indices.append(test_index)         
    feats_embed = np.array(feats_embed)
    return feats_indices,torch.Tensor(feats_embed)

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def get_ccs_pair(features,labels=None):
    if labels is not None:
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float(),torch.from_numpy(labels[i].reshape(-1,1)).float()] for i in range(len(labels))]      
    else:
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float()] for i in range(len(features))]
    class_code = []
    adduct_type = []
    adduct_code = []
    for i in range(len(features)):
        mol_info = features[i][0]
        class_code.append(mol_info[-1])
        adduct_type.append(mol_info[-2])
        adduct_code.append(mol_info[-3])
    return ccs_pair,adduct_type,adduct_code,class_code

# Step 1 molecule compound classification
def cmp_classify(moldata,mode):
    moldata['md'] = moldata['SMI'].apply(lambda x: smile_to_md(x))
    moldata = moldata[~moldata['md'].isna()] 
    with open("./models/"+mode+"_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("./models/"+mode+"_model_encoder.pkl", "rb") as f:
        model = pickle.load(f)
    X = np.asarray([np.asarray(i) for i in moldata['md'].values])
    y = model.predict(X)
    y_label = le.inverse_transform(y)
    moldata['Compound Class'] = y_label
    moldata['Class'] = y
    return moldata

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
