######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

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

st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Arimo');
html, body, [class*="css"]  {
   font-family: 'Arimo';
}
</style>
""", unsafe_allow_html=True)

######################
# Page Title
######################

image = Image.open('./images/ccs-prediction.png')
st.image(image, use_column_width=True)

#Load data

df_f = st.session_state['df_f'] 
mode = st.session_state['mode']
abbr_mode = {'lipid positive mode':'lipid_pos','lipid negative mode':'lipid_neg','metabolite positive mode':'met_pos','metabolite negative mode':'met_neg','drug mode (beta)':'drug_pos'}

try:
   feats_fp =  st.session_state['feats_fp'] 
   feats_md =  st.session_state['feats_md']
   st.write('Molecular features are loaded.')
   features,labels = generate_data_loader(df_f,feats_md,feats_fp,opt='test')
except:
   st.write('Please note, molecular features are not generated yet.')

st.dataframe(df_f.head())
ccs_button = st.button("Click here to predict CCS values") # Give button a variable name
if ccs_button: # Make button a condition.
   st.text("Start predicting using molecular fingerprint features")
   feats = []
   for t in features:
      feats.append(t[3])
   feats = np.array(feats)
   model_file = './models/'+abbr_mode[mode]+'_rf_fingerprint.sav'
   model = pickle.load(open(model_file, 'rb'))
   result = model.predict(feats)
   df_f['ccs_fp'] = result
   st.text("Finished")
   st.text("Start predicting using molecular fingerprint features")
   feats = []
   for t in features:
      feats.append(t[2])
   feats = np.array(feats)
   model_file = './models/'+abbr_mode[mode]+'_rf_descriptor.sav'
   model = pickle.load(open(model_file, 'rb'))
   result = model.predict(feats)
   df_f['ccs_md'] = result
   st.text("Finished")
   st.write('Prediction result')
   st.dataframe(df_f.head())
   csv_result = convert_df(df_f)
   st.download_button(
      "Press to download result",
      csv_result,
      "ccs_prediction.csv",
      "text/csv",
      key='download-csv'
   )