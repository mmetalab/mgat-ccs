import plotly.express as px
from dash import html, Input, Output, dcc, State, Dash, dash_table, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_uploader as du
import base64
import datetime
import io
import json
# file imports
from maindash import my_app
from utils.file_operation import read_file_as_str
from utils.mol import *

def feature_header():
    layout = html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src="https://images.unsplash.com/photo-1614854262340-ab1ca7d079c7?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "position": "relative",
                            },
                        ),
                    ],
                    style={
                        "height": "200px",
                        "overflow": "hidden",
                        "position": "relative",
                    },
                ),
                html.H1(
                    "Molecular Featurization",
                    style={
                        "position": "absolute",
                        "top": "80%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "color": "white",
                        "text-align": "center",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "position": "relative",
                "text-align": "center",
                "color": "white",
            },
        )
    return layout

lipid_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
lipid_neg_dict = {'[M+CH3COO]-': 0, '[M+HCOO]-': 1, '[M+Na-2H]-': 2, '[M-CH3]-': 3, '[M-H]-': 4}
met_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+NH4]+': 3, '[M+Na]+': 4}
met_neg_dict = {'[M+Na-2H]-': 0, '[M-H]-': 1}
drug_pos_dict = {'[M+H-H2O]+': 0, '[M+H]+': 1, '[M+K]+': 2, '[M+Na]+': 3}
mode_dict = {'Lipid positive mode':lipid_pos_dict,'Lipid negative mode':lipid_neg_dict,'Metabolite positive mode':met_pos_dict,'Metabolite negative mode':met_neg_dict,'Drug mode (beta)':drug_pos_dict}
mode_dict_keys = list(mode_dict.keys())
mode_dict_values = list(mode_dict.values())

def feature_layout():
    layout = html.Div(
        [
            html.Br(),
            html.H3(
                "\nPerform molecular featurization",
                style={"textAlign": "center", "color": "#082446"},
            ),
            dcc.Dropdown(id="my-mode-dropdown",
            options=[
                {'label': mode_dict_keys[0], 'value': mode_dict_keys[0]},
                {'label': mode_dict_keys[1], 'value': mode_dict_keys[1]},
                {'label': mode_dict_keys[2], 'value': mode_dict_keys[2]},
                {'label': mode_dict_keys[3], 'value': mode_dict_keys[3]},
                {'label': mode_dict_keys[4], 'value': mode_dict_keys[4]},
            ],
            placeholder="Select a mode",
            ),
            html.Br(),
            html.Div(id='mode-output'),
            html.Br(),
            dbc.Button('Click to Run', id='btn-nclicks-feats', n_clicks=0),
            html.Br(),
            html.Div(id='container-button-feats')
        ]
    )
    return layout

@my_app.callback(
    Output('mode-output', 'children'),
    Input('my-mode-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'

@my_app.callback(
    Output('container-button-feats', 'children'),
    Input('btn-nclicks-feats', 'n_clicks')
)

def displayClick(btn1):
    msg = "Molecular featurization is not yet performed. Click the button to run featurization."
    hidden = True
    if "btn-nclicks-feats" == ctx.triggered_id:
        msg = "Molecular featurization is performed." 
        hidden = False
    return html.Div(
            [html.Br(),
             html.Div(msg),
             html.Div(id='print-data-mol', hidden=hidden)
            ]
        )

@my_app.callback(
    Output('processed-feature', 'data'),
    Input('my-mode-dropdown', 'value'),
    Input('loaded-data', 'data')
)

def feats_convert(value,uploaded_df):
    df = pd.read_json(uploaded_df, orient='split')
    temp = pd.DataFrame(columns=['Adduct'])
    temp['Adduct'] = list(mode_dict[value].keys())
    df = df.merge(temp, how='cross')
    df_f = encode_adduct(df,mode_dict[value])
    df_f.to_json(orient='split')
    return df_f

@my_app.callback(Output('print-data-mol', 'children'),
              Input('processed-feature', 'data'))

def feats_update(processed_df):
    df = pd.read_json(processed_df, orient='split')
    y_adduct,feats_fp,feats_md = feature_generator(df_f)
    features,labels = generate_data_loader(df,feats_md,feats_fp,opt='test')

    result_df = predict_ccs(features,value,df_f)
    
    return html.Div([
        # html.Hr(),  # horizontal line
        html.Div('The first few lines of the uploaded molecule data.'),
        dbc.Container(
        [
            dbc.Spinner(
                dash_table.DataTable(
                    result_df.to_dict('records'),
                    id='dash-table',
                    columns=[
                        {'name': column, 'id': column}
                        for column in df.columns
                    ],
                    page_size=10,
                    style_header={
                        'font-family': 'Arial',
                        'font-weight': 'bold',
                        'text-align': 'center'
                    },
                    style_table={'overflowX': 'auto'},
                    style_data={
                        'font-family': 'Arial',
                        'text-align': 'center'
                    }
                )
            )
        ],
        style={
            'font-family': 'Arial',
            'margin-top': '50px'
        }
    ),
        html.Hr(),  # horizontal line
    ])

def feature_info():
    return (feature_header(),feature_layout())