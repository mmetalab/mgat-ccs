import sys
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dgllife.utils import atom_degree, atomic_number, atom_explicit_valence, atom_formal_charge, atom_num_radical_electrons, atom_mass, BaseAtomFeaturizer
from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot
from dgllife.utils import ConcatFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from dgllife.utils import BaseAtomFeaturizer, atom_mass
from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import rdBase
from dgllife.utils import smiles_to_complete_graph, smiles_to_bigraph
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv
from dgl.nn.pytorch.conv import DenseGraphConv,TAGConv
from dgllife.model.gnn.graphsage import GraphSAGE
from collections import OrderedDict
from sklearn.metrics import r2_score
from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdDescriptors
from descriptastorus.descriptors import rdNormalizedDescriptors
import pickle
import torch.utils.data as data
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import KFold



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
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
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
        # self.layer3 = MPLayer()
        self.R = nn.Linear(10, 32)
    def forward(self, adducts, adj, g, h, h2):
        g, h = self.layer1(g,h)
        g, h = self.layer2(g,h)
        # g, h = self.layer3(g,h)
        ## Add attention layer here 
        N = len(h.keys())
        # catted = Variable(torch.zeros(N, 5))
        catted = Variable(torch.zeros(N, 10))
        keys = list(h.keys())
        for i in range(N):
            k = keys[i]
            catted[i] = torch.cat([h[k], h2[k]], 1)
            # catted[i] = torch.add(h[k], h2[k])
            # catted[i] = torch.mean(h[k], h2[k])
            # catted[i] = torch.div(catted[i], 2)
        # print(catted.shape,catted)
        activated_reads = self.R(catted)
        # print(activated_reads.shape,activated_reads)
        # activated_reads = self.R(catted)
        # print(activated_reads.shape,activated_reads)
        # readout = torch.sum(activated_reads, dim=0)
        readout = torch.mean(activated_reads, dim=0)
        # print(readout.shape,readout)
        # readout = torch.cat((readout,adducts.unsqueeze(-1)),dim=-1)
        readout = torch.cat((readout,adducts),dim=-1)
        # print(readout.shape,readout)
        # print(readout.shape)
        x = torch.relu(readout)
        x = self.fc1(x)
        embedding = x
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return embedding,x 

from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    #Mean absolute percentage error regression loss.
    mape = np.abs(y_true-y_pred) / np.abs(y_true)
    return np.average(mape)
def median_absolute_percentage_error(y_true, y_pred):
    #Mean absolute percentage error regression loss.
    mape = np.abs(y_true-y_pred) / np.abs(y_true)
    return  np.median(mape)

def percentile_absolute_percentage_error(y_true, y_pred):
    #Mean absolute percentage error regression loss.
    mape = np.abs(y_true-y_pred) / np.abs(y_true)
    result = pd.DataFrame(columns=['5%','25%','50%','75%','95%'])
    result.loc[0,'5%'] = np.percentile(mape,5)
    result.loc[0,'25%'] = np.percentile(mape,25)
    result.loc[0,'50%'] = np.percentile(mape,50)
    result.loc[0,'75%'] = np.percentile(mape,75)
    result.loc[0,'95%'] = np.percentile(mape,95)
    return  result

def model_eval_ccs(test_data_loader):
    model.eval()
    test_pred = []
    true_label = []
    feats_embed = []
    feats_adduct = []
    feats_indices = []
    with torch.no_grad():
        for i in test_data_loader:
            test_info,test_smiles,test_md, test_labels = i[0],i[1],i[2],i[3] 
            smile = test_smiles
            feats_md = test_md
            adj1, g, h = construct_multigraph(smile) 
            h2 = h
            embedding,output = model(feats_md,adj1,g,h,h2)
            feats_embed.append(embedding.data.numpy())
            feats_adduct.append(feats_md.data.numpy())  
            feats_indices.append(test_info)         
            test_pred.append(output.data.numpy()[0])
            true_label.append(test_labels)
    feats_embed = np.array(feats_embed)
    feats_adduct = np.array(feats_adduct)
    test_pred = np.array(test_pred)
    return feats_indices,torch.Tensor(feats_embed),torch.Tensor(feats_adduct),torch.Tensor(test_pred),torch.Tensor(true_label)


def model_embed_ccs(test_data_loader):
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


def print_eval(pred,true):
    result = {}
    print('mean absolute error: %.3f' % mean_absolute_error(pred,true))
    print('mean squared error: %.3f' % mean_squared_error(pred,true))
    print('median absolute error: %.3f' % median_absolute_error(pred,true))
    print('mean absolute percentage error: %.3f%%'  % (100*mean_absolute_percentage_error(pred,true)))
    print('median absolute percentage error: %.3f%%'  % (100*median_absolute_percentage_error(pred,true)))
    result['mae'] = mean_absolute_error(pred,true)
    result['mse'] = mean_squared_error(pred,true)
    result['mdae'] = median_absolute_error(pred,true)
    result['mare %'] = 100*mean_absolute_percentage_error(pred,true)
    result['mdare %'] = 100*median_absolute_percentage_error(pred,true)
    return result

def get_ccs_pair(features,labels=None):
    if labels is not None:
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float(),torch.from_numpy(labels[i].reshape(-1,1)).float()] for i in range(len(labels))]      
    else:
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float()] for i in range(len(features))]
    class_code = []
    adduct_type = []
    adduct_code = []
    mol_index = []
    mol_name = []
    for i in range(len(features)):
        # mol_info:[index,name,smiles,adduct_code,adduct_type,class_code]
        mol_info = features[i][0]
        class_code.append(mol_info[-1])
        adduct_type.append(mol_info[-2])
        adduct_code.append(mol_info[-3])
        mol_index.append(mol_info[0])
        mol_name.append(mol_info[1])
    return ccs_pair,adduct_type,adduct_code,class_code,mol_index,mol_name



if __name__ == '__main__':

    input = sys.argv
    model_file = input[1]
    datamode = input[2]
    mode = input[3]
    feature_file = input[4]
    note = input[8]
    print(len(input))
    if mode == 'evaluate':
        labels_file = input[5]
        model = load_checkpoint(model_file)
        features = torch.load(feature_file)
        labels = torch.load(labels_file)
        kf = KFold(n_splits=5,shuffle=True, random_state=128)
        sum_df = pd.DataFrame(columns=['dataset','mae','mse','mdae','mare %','mdare %'])
        result_df_details = pd.DataFrame(columns=['model','fold','index','name','pred','true','adduct_type','adduct_code','class_code','mode'])
        for k, (train, test) in enumerate(kf.split(labels)):
            temp1 = pd.DataFrame(columns=['model','fold','index','name','pred','true','adduct_type','adduct_code','class_code','mode'])
            temp2 = pd.DataFrame(columns=['model','fold','index','name','pred','true','adduct_type','adduct_code','class_code','mode'])
            train_features = [features[i] for i in train]
            train_labels = [labels[i] for i in train]
            test_features = [features[i] for i in test]
            test_labels = [labels[i] for i in test]
            train_set,train_adduct_type,train_adduct_code,train_class_code,train_mol_index,train_mol_name = get_ccs_pair(train_features,train_labels)
            test_set,test_adduct_type,test_adduct_code,test_class_code,test_mol_index,test_mol_name = get_ccs_pair(test_features,test_labels)                        
            train_feats_indices, train_feats_embed, train_feats_adduct, train_pred, train_labels = model_eval_ccs(train_set)

            print(train_feats_embed.shape,train_feats_adduct.shape)
            test_result = print_eval(train_pred,train_labels)
            
            gbmodel = GradientBoostingRegressor(random_state=42,n_estimators=500).fit(train_feats_embed, train_labels)

            # Save the mgat-ccs-model
            with open("./"+datamode+'_'+str(note)+"_mgat-ccs-model.pkl", "wb") as f:
                pickle.dump(gbmodel, f)
            # Load the mgat-ccs-model
            with open("./"+datamode+'_'+str(note)+"_mgat-ccs-model.pkl", "rb") as f:
                gbmodel = pickle.load(f)

            print('Evaluation on train dataset:')
            y_pred = gbmodel.predict(train_feats_embed)
            test_result = print_eval(torch.Tensor(y_pred),train_labels)
            break
        
            # Add molecule information here
            temp1['index'] = train_mol_index
            temp1['name'] = train_mol_name
            temp1['model'] = 'MGAT-CCS'
            temp1['pred'] = y_pred
            temp1['true'] = train_labels
            temp1['adduct_type'] = train_adduct_type
            temp1['adduct_code'] = train_adduct_code 
            temp1['class_code'] = train_class_code 
            temp1['fold'] = k 
            temp1['mode'] = 'train'
            test_feats_indices, test_feats_embed, test_feats_adduct, test_pred, test_labels = model_eval_ccs(test_set)
            print('Evaluation on test dataset:')
            y_pred = gbmodel.predict(test_feats_embed)
            test_result = print_eval(torch.Tensor(y_pred),test_labels)

            ttresult = percentile_absolute_percentage_error(torch.Tensor(y_pred),test_labels)
            ttresult.to_csv('./results/'+datamode+'_test_result_'+str(k)+'_'+str(note)+'.csv',index=False)

            # Add molecule information here
            temp2['index'] = test_mol_index
            temp2['name'] = test_mol_name
            temp2['model'] = 'MGAT-CCS'
            temp2['pred'] = y_pred
            temp2['true'] = test_labels
            temp2['adduct_type'] = test_adduct_type
            temp2['adduct_code'] = test_adduct_code 
            temp2['class_code'] = test_class_code 
            temp2['fold'] = k 
            temp2['mode'] = 'test'
            temp = pd.concat([temp1,temp2])
            result_df_details = pd.concat([result_df_details,temp], ignore_index=True)
            sum_df.loc[k,'dataset'] = datamode
            sum_df.loc[k,'mae'] = test_result['mae']
            sum_df.loc[k,'mse'] = test_result['mse']
            sum_df.loc[k,'mdae'] = test_result['mdae']
            sum_df.loc[k,'mare %'] = test_result['mare %']
            sum_df.loc[k,'mdare %'] = test_result['mdare %']

            # external dataset evaluation
            ex_feature_file = input[6]
            ex_labels_file = input[7]

            ex_features = torch.load(ex_feature_file)
            ex_labels = torch.load(ex_labels_file)
            test_set,test_adduct_type,test_adduct_code,test_class_code,test_mol_index,test_mol_name = get_ccs_pair(ex_features)
            test_feats_indices, test_feats_embed = model_embed_ccs(test_set)        
            print('Evaluation on external dataset:')
            y_pred = gbmodel.predict(test_feats_embed)
            test_result = print_eval(torch.Tensor(y_pred),torch.Tensor(ex_labels))



            exresult = percentile_absolute_percentage_error(torch.Tensor(y_pred),torch.tensor(ex_labels))
            exresult.to_csv('./results/'+datamode+'_exresult_'+str(k)+'_'+str(note)+'.csv',index=False)

        result_df_details.to_csv('./results/'+datamode+'_'+str(note)+'_result_details_all.csv',index=False)
        result_file = './results/'+'sum_'+datamode+'_'+str(note)+'_result_all.csv'
        # 'Save summary result file'
        sum_df.to_csv(result_file,index=False)
        sum_df = pd.read_csv(result_file)
        summary = sum_df.describe().reset_index()
        sum_result_file = './results/'+'all_sum_'+datamode+'_'+str(note)+'_result_all.csv'
        'Save summary result file'
        summary.to_csv(sum_result_file,index=False)
    
    if mode == 'predict' and len(input) == 5:



        # test_data_loader_mpnn = torch.load(test_data)
        model = load_checkpoint(model_file)
        # features = torch.load(train_feature_file)
        # labels = torch.load(train_labels_file)
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float(),features[i][3],torch.from_numpy(labels[i].reshape(-1,1)).float()] for i in range(len(features))]      
        # use 20% of training data for validation
        train_set_size = int(len(labels) * 0.8)
        test_set_size = len(labels) - train_set_size
        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, test_set = data.random_split(ccs_pair, [train_set_size, test_set_size], generator=seed)
        test_feats_indices, test_feats_embed, test_feats_adduct, test_pred, test_label = model_eval_ccs(train_set)
        test_output = [test_feats_indices, test_feats_embed, test_feats_adduct, test_pred, test_label]
        gbmodel = GradientBoostingRegressor(random_state=42,n_estimators=500).fit(test_feats_embed, test_label)
        features = torch.load(feature_file)
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float(),features[i][3]] for i in range(len(features))]      
        test_feats_indices, test_feats_embed, test_pred = model_embed_ccs(ccs_pair)
        y_pred = gbmodel.predict(test_feats_embed)
        print(y_pred)
        # test_result = print_eval(torch.Tensor(y_pred),test_label)
