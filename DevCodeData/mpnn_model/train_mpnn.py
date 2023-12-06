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
import torch.multiprocessing as mp
import torch.utils.data as data


def finger_print(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array
def feature_generator(df):
    y_ccs = df['CCS'].values
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
    feats_fp = fp_ccs
    feats_md = md_ccs
    feats_fp_md = np.hstack((fp_ccs, md_ccs))
    return feats_fp,feats_md,feats_fp_md,y_ccs


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


# atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
# bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

def construct_multigraph(smile):
    atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number, atom_explicit_valence, atom_formal_charge, atom_num_radical_electrons])
    mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
    # atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    # mol_atom_featurizer = atom_featurizer
    bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot])
    mol_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})
    # bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    # mol_bond_featurizer = bond_featurizer
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
        # h[i] = Variable(torch.FloatTensor(atom_feats[i]).view(1, 74))
        for j in range(molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)
            if e_ij != None:
                p = e_ij.GetIdx()
                e_ij = Variable(torch.FloatTensor(bond_feats[2*p]).view(1,11))
                # e_ij = Variable(torch.FloatTensor(bond_feats[2*p]).view(1,12))
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
        # self.V = nn.Linear(74,12)
        # self.U = nn.Linear(98,32)
        # self.E = nn.Linear(12,12)
    def forward(self, g, h):
        for v in g.keys():
            neighbors = g[v]
            for neighbor in neighbors:
                e_vw = neighbor[0] # neighbor atom edge feature variable
                w = neighbor[1] # neighbor atom 
                m_w = self.V(h[w]) # outputsize (,74) size of node
                m_e_vw = self.E(e_vw) # outputsize (,12) size of edge 
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1) # outputsize (,74+74+12) nodesize + nodesize + edgesize
                # print(reshaped.shape)
                h[v] = self.U(reshaped)
        return g, h
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 32 + len(md)201, 205 (lipid pos/neg, mets_pos), 202:mets/neg
        self.fc1 = nn.Linear(234, 128)
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
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x  

def model_eval_loss(model,test_data_loader):
    model.eval()
    test_pred = []
    true_label = []
    ll = nn.MSELoss()
    losses = 0
    with torch.no_grad():
        for train_index, smiles, test_md, test_fp, test_labels in test_data_loader:
            test_loss = Variable(torch.zeros(1, 1))
            for j in range(len(smiles)):
                smile = smiles[j]
                feats_md = test_md[j]
                adj1, g, h = construct_multigraph(smile) 
                adj2, g2, h2 = construct_multigraph(smile)
                output = model(feats_md,adj1,g,h,h2)
                test_loss += ll(output,test_labels[j]) 
            losses += test_loss.data.numpy()[0]   
    return losses

def train(args):
    (pid,model,epochs,loss_fn,optimizer,train_data_loader,test_data_loader,loss_history,loss_history_val) = args
    for i in range(epochs):  
        losses = 0
        losses_val = 0
        for train_index, smiles, train_md, train_fp, train_labels in train_data_loader:
            train_loss = Variable(torch.zeros(1, 1))
            for j in range(len(smiles)):
                smile = smiles[j]
                feats_md = train_md[j]
                adj1, g, h = construct_multigraph(smile) 
                adj2, g2, h2 = construct_multigraph(smile)
                output = model(feats_md,adj1,g,h,h2)
                train_loss += loss_fn(output,train_labels[j])
            optimizer.zero_grad()
            losses += train_loss.data.numpy()[0]
            train_loss.backward()
            optimizer.step()
        losses_val = model_eval_loss(model,test_data_loader)
        loss_history[pid, i] = losses.item()
        loss_history_val[pid, i] = losses_val.item()
        # loss_all.append(losses)
        print(i, losses, losses_val)
    # return loss_all



def load_pretrained_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    # model.train()
    return model



if __name__ == '__main__':
    input = sys.argv
    datamode = input[1]
    # load_model_file = input[2]
    save_model_file = input[2]
    # save_model_file = input[2].replace('_80','_160')
    print(save_model_file)
    Epoch = int(input[3])
    mode = input[4]
    spec = input[5]
    print(mode)
    
    if mode == 'train':
        if spec == 'yes':
            s = '_spec'
        else:
            s = ''
        feature_file = './mpnn_data/'+datamode+'_feature_code_adducts%s.pt' % s
        labels_file = './mpnn_data/'+datamode+'_label_code_adducts%s.pt' % s
        print(feature_file,labels_file)
        features = torch.load(feature_file)
        labels = torch.load(labels_file)
        ccs_pair = [[features[i][0],features[i][1],torch.from_numpy(features[i][2]).float(),features[i][3],torch.from_numpy(labels[i].reshape(-1,1)).float()] for i in range(len(features))]      
        
        # use 10% of training data for validation
        train_set_size = int(len(labels) * 0.9)
        test_set_size = len(labels) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, test_set = data.random_split(ccs_pair, [train_set_size, test_set_size], generator=seed)
        print(len(train_set),len(test_set))
        train_data_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                                collate_fn=None)
        test_data_loader = DataLoader(test_set, batch_size=32, shuffle=True,
                                collate_fn=None)
        # Save training data and test data
        train_data_save = './mpnn_data/'+datamode+'_training_data.pt'
        test_data_save = './mpnn_data/'+datamode+'_test_data.pt'
        torch.save(train_data_loader, train_data_save)
        torch.save(test_data_loader, test_data_save)
        # atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number, atom_explicit_valence, atom_formal_charge, atom_num_radical_electrons])
        # mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
        # bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot])
        # mol_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})
        model = Model()
        # model = load_pretrained_checkpoint(load_model_file)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
        model.share_memory()
        loss_fn = nn.MSELoss()
        # train(model,Epoch,loss_fn,optimizer,train_data_loader,loss_all)
        num_processes = 8
        processes = []
        # for rank in range(num_processes):
        #     # 4 个进程，如果每个进程为epoch = 10，也就是说其实迭代了 4*10 = 40 次 !!!
        #     # p = mp.Process(target=train, args=(model,Epoch,loss_fn,optimizer,train_data_loader,loss_all))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

        loss_history = torch.zeros(num_processes, Epoch)
        loss_history.share_memory_()

        loss_history_val = torch.zeros(num_processes, Epoch)
        loss_history_val.share_memory_()
        for pid in range(num_processes):
            p = mp.Process(target=train, args=((pid,model,Epoch,loss_fn,optimizer,train_data_loader,test_data_loader, loss_history,loss_history_val),))
            processes.append(p)
            p.start()
        [p.join() for p in processes]

        # Save model
        checkpoint = {'model': Model(),
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}

        torch.save(checkpoint, save_model_file)
        loss_data_save = './mpnn_data/'+datamode+'_loss_data.pt'
        torch.save(loss_history,loss_data_save)
        print(len(loss_history))
        loss_data_save_val = './mpnn_data/'+datamode+'_loss_data_val.pt'
        torch.save(loss_history_val,loss_data_save_val)
        print(len(loss_history_val))
    