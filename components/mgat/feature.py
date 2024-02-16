from dash import html, Input, Output, dcc, State, Dash, dash_table, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from maindash import app
from utils.mol import *

def feature_header():
    layout = html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src="https://github.com/mmetalab/mgat-ccs/raw/main/images/ccs-prediction-tab.png",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "position": "relative",
                                "height": "300px",
                                "overflow": "hidden"
                            },
                        ),
                    ],
                )
            ],
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
                "\nPerform CCS Value Prediction",
                style={'width': '100%', "textAlign": "center", "color": "#082446"},
            ),
            html.Br(),
            html.H5("\nSelect the ion mode and IMS technique for featurization."),
            dcc.Dropdown(id="my-mode-dropdown",
            options=[
                {'label': mode_dict_keys[0], 'value': mode_dict_keys[0]},
                {'label': mode_dict_keys[1], 'value': mode_dict_keys[1]},
                {'label': mode_dict_keys[2], 'value': mode_dict_keys[2]},
                {'label': mode_dict_keys[3], 'value': mode_dict_keys[3]},
                {'label': mode_dict_keys[4], 'value': mode_dict_keys[4]},
            ],
            placeholder="Select an ion mode",
            style={'width': '100%'},
            ),
            html.Br(),
            html.Div(id='ims-output'),
            dcc.Dropdown(id="my-ims-dropdown",
            options=[
                {'label': 'TIMS', 'value': 'TIMS'},
                {'label': 'DTMS', 'value': 'DTMS'},
            ],
            placeholder="Select an IMS technique",
            style={'width': '100%'},
            ),
            html.Br(),
            html.Div(id='mode-output'),
            html.Br(),
            html.Div(id='container-button-feats'),
            dbc.Button('Click to Run', id='btn-nclicks-feats', n_clicks=0),
        ]
    )
    return layout

@app.callback(
    Output('mode-output', 'children'),
    Input('my-mode-dropdown', 'value'),
    Input('my-ims-dropdown', 'value')
)
def update_output(value1,value2):
    return f'You have selected {value1} by {value2}.'

@app.callback(
    Output('container-button-feats', 'children'),
    Input('btn-nclicks-feats', 'n_clicks')
)

def displayClick(btn1):
    msg = "CCS value prediction is not yet performed. Click button to run prediction."
    hidden = True
    if "btn-nclicks-feats" == ctx.triggered_id:
        msg = "CCS value prediction is performed." 
        hidden = False
    return html.Div(
            [
             html.H5(id='print-data-mol', hidden=hidden),
             html.Div(msg,style={'width': '100%',"color": "#082446"})
            ]
        )

@app.callback(
    Output('processed-feature', 'data'),
    Input('my-mode-dropdown', 'value'),
    Input('loaded-data-input', 'data'),
    Input('loaded-data', 'data')
)

def feats_convert(value,loaded_df,uploaded_df):
    if loaded_df is not None:
        df = pd.read_json(loaded_df, orient='split')
    if uploaded_df is not None:
        df = pd.read_json(uploaded_df, orient='split')
    temp = pd.DataFrame(columns=['Adduct'])
    temp['Adduct'] = list(mode_dict[value].keys())
    df = df.merge(temp, how='cross')
    df_f = encode_adduct(df,mode_dict[value])
    return df_f.to_json(orient='split')

@app.callback(Output('print-data-mol', 'children'),
              Input('processed-feature', 'data'),
              Input('my-mode-dropdown', 'value'))

def feats_update(processed_df,value):
    df_f = pd.read_json(processed_df, orient='split')
    y_adduct,feats_fp,feats_md = feature_generator(df_f)
    features,labels = generate_data_loader(df_f,feats_md,feats_fp,opt='test')
    result_df = predict_ccs(features,value,df_f)

    return html.Div([
        html.Div('The predicted CCS value for each molecule.'),
        dbc.Container(
        [
            dbc.Spinner(
                dash_table.DataTable(
                    result_df.to_dict('records'),
                    id='dash-table',
                    columns=[
                        {'name': column, 'id': column}
                        for column in result_df.columns
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