######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from collections import OrderedDict
from sklearn import preprocessing
from descriptastorus.descriptors import rdNormalizedDescriptors


st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Arimo');
html, body, [class*="css"]  {
   font-family: 'Arimo';
}
</style>
""", unsafe_allow_html=True)

def encode_adduct(df,mode):
    mode_mappings = mode_dict[mode]
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



######################
# Page Title
######################

image = Image.open('./images/feature-generation.png')

st.image(image, use_column_width=True)

st.subheader('Generate molecular features with different adduct.')

#Load data
df = st.session_state['df']

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('Select a mode to compute the molecular features')

st.markdown('<div style="text-align: justify;"> Select a mode to compute the molecular features.\
</div>', unsafe_allow_html=True)

lipid_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
lipid_neg_dict = {'[M+CH3COO]-': 0, '[M+HCOO]-': 1, '[M+Na-2H]-': 2, '[M-CH3]-': 3, '[M-H]-': 4}
met_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
met_neg_dict = {'[M+Na-2H]-': 0, '[M-H]-': 1}
drug_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+Na]+': 3}
mode_dict = {'lipid positive mode':lipid_pos_dict,'lipid negative mode':lipid_neg_dict,'metabolite positive mode':met_pos_dict,'metabolite negative mode':met_neg_dict,'drug mode (beta)':drug_pos_dict}

mode = st.selectbox(
    'Due to computing limitations, select one mode at a time.',
    list(mode_dict.keys()))

st.write('You selected:', mode)
st.write('The server will generate features for the following mode:')
for k,v in mode_dict[mode].items():
    st.markdown("**:orange[%s]**" % k)


temp = pd.DataFrame(columns=['Adduct'])
temp['Adduct'] = list(mode_dict[mode].keys())
df_f = df.merge(temp, how='cross')

mp_button = st.button("Click here to run generate features") # Give button a variable name
if mp_button: # Make button a condition.
    st.text("Start generate molecular features")
    df_f = encode_adduct(df_f,mode)
    y_adduct,feats_fp,feats_md = feature_generator(df_f)
    features,labels = generate_data_loader(df,feats_md,feats_fp,opt='test')
    st.text("Finished")
    st.session_state['feats_fp'] = feats_fp
    st.session_state['feats_md'] = feats_md
    st.session_state['mode'] = mode
    st.download_button(label='Download features (molecular fingerprint)',
            data= pickle.dumps(feats_fp),
            file_name='feats_fp.pkl')
    
    st.download_button(label='Download features (molecular descriptor)',
        data= pickle.dumps(feats_md),
        file_name='feats_md.pkl')
    st.session_state['df_f'] = df_f