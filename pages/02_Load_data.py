######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw



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

image = Image.open('./images/load-data.png')
st.image(image, use_column_width=True)


######################
# Input scRNA-seq data (Side Panel)
######################

# st.sidebar.header('User scRNA-seq data (please upload files)')
uploaded_file = st.sidebar.file_uploader("Choose a CSV/TSV file with molecule structural information", type="csv")
use_example_file = st.sidebar.checkbox(
    "Use example file", False, key='1', help="Use in-built example file to demo the app"
)


# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block

st.header('Read molecule data')

#Load data
if uploaded_file is not None:
    st.write("Using uploaded file")
    df = pd.read_csv(uploaded_file,sep=',')

elif use_example_file:
    st.write("An example file was loaded")
    file = './Data/example_data.csv'
    df = pd.read_csv(file,sep=',')
 
else:
    st.write("An example file was preloaded")
    file = './Data/example_data.csv'
    df = pd.read_csv(file,sep=',')

# Download example file
with open('./Data/example_data.csv', 'rb') as f:
    s = f.read()
st.sidebar.download_button(
    label="Download example file",
    data=s,
    file_name='example_data.csv',
    # mime='tsv',
)

st.header('Characteristics of loaded molecular data')

st.dataframe(df.head().style.highlight_max(axis=0))
mol_count = df.shape[0]
col1, col2 = st.columns(2)
col1.metric("Number of molecules", mol_count)
st.session_state['df'] = df

# st.write('**Select an example molecule to show the structure**')
# example_mols = dict(zip(list(df.Name)[:10],list(df.SMI)[:10]))
# option = st.selectbox(
#     '',
#     list(example_mols.keys()))

# st.write('You selected:', option)
# if option:
#     compound_smiles = example_mols[option]
#     m = Chem.MolFromSmiles(compound_smiles)
#     Draw.MolToFile(m,'mol.png')
#     st.image('mol.png')