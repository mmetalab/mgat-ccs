from dash import html, Input, Output, dcc, dash_table, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from maindash import app
from utils.mol import *
import __main__
import time

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
mode_dict = {'lipid_pos':lipid_pos_dict,'lipid_neg':lipid_neg_dict,'mets_pos':met_pos_dict,'mets_neg':met_neg_dict}
mode_abbr = {'lipid positive mode':'lipid_pos','lipid negative mode':'lipid_neg','metabolite positive mode':'mets_pos','metabolite negative mode':'mets_neg'}
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
                {'label': 'Lipid positive mode', 'value': 'lipid positive mode'},
                {'label': 'Lipid negative mode', 'value': 'lipid negative mode'},
                {'label': 'Metabolite positive mode', 'value': 'metabolite positive mode'},
                {'label': 'Metabolite negative mode', 'value': 'metabolite negative mode'},
            ],
            placeholder="Select an ion mode",
            style={'width': '100%'},
            ),
            html.Br(),
            html.Div(id='ims-output'),
            dcc.Dropdown(id="my-ims-dropdown",
            options=[
                {'label': 'TWIMS', 'value': 'TWIMS'},
                {'label': 'DTIMS', 'value': 'DTIMS'},
            ],
            placeholder="Select an IMS technique",
            style={'width': '100%'},
            ),
            html.Br(),
            html.H5(id='mode-output'),
            dbc.Button('Run prediction', id='btn-nclicks-feats', n_clicks=0),
            dbc.Tooltip("Click the button again if you change the mode.",
            target="btn-nclicks-feats",
            ),
            html.Br(),
            dcc.Loading(id="ls-loading-1", children=[html.Div(html.H5(id='CCS'))], type="default", debug=False),
        ]
    )
    return layout

@app.callback(
    Output('mode-output', 'children'),
    Input('my-mode-dropdown', 'value'),
    Input('my-ims-dropdown', 'value')
)
def update_output(ionmode,ims):
    return f'You have selected {ionmode} by {ims}.'

@app.callback(Output('CCS', 'children'),
              Output('predicted-ccs', 'data'),
            Input('my-mode-dropdown', 'value'),
            Input('my-ims-dropdown', 'value'),
            Input('loaded-data-input', 'data'),
            Input('loaded-data', 'data'),
            Input('btn-nclicks-feats', 'n_clicks'))

def feats_update(value, ims, loaded_df,uploaded_df, n_clicks):
    if 'btn-nclicks-feats' == ctx.triggered_id and n_clicks > 0:
        time.sleep(1)
        if loaded_df is not None:
            df = pd.read_json(loaded_df, orient='split')
        if uploaded_df is not None:
            df = pd.read_json(uploaded_df, orient='split')
        mode = mode_abbr[value]
        moldata = cmp_classify(df,mode)
        temp = pd.DataFrame(columns=['Adduct'])
        temp['Adduct'] = list(mode_dict[mode].keys())
        moldata = moldata.merge(temp, how='cross')
        df_f = encode_adduct(moldata,mode)
        feats_md = feature_generator(df_f)
        features,labels = generate_data_loader(df_f,feats_md,option='predict')
        mode = mode_abbr[value]
        model_file = './models/'+mode+'_'+ims+'_chkpts.pth'
        setattr(__main__, "Model", Model)
        setattr(__main__, "MPLayer", MPLayer)
        checkpoint = torch.load(model_file)
        model = load_checkpoint(checkpoint)
        with open("./models/"+mode+'_'+ims+"_mgat-ccs-model.pkl", "rb") as f:
            gbmodel = pickle.load(f)
        test_set,test_adduct_type,test_adduct_code,test_class_code = get_ccs_pair(features)
        test_feats_indices, test_feats_embed = model_embed_ccs(model,test_set)
        y_pred = gbmodel.predict(test_feats_embed)
        df_f['Predicted CCS'] = np.asarray(y_pred)
        df_f['Predicted CCS'] = df_f['Predicted CCS'].round(decimals=3)
        df_f['IMS Mode'] = ims
        result_df = df_f[['Name','Predicted CCS','Adduct','Compound Class','IMS Mode','SMI']]
        return html.Div([
            html.Hr(),
            html.H5('Predicted CCS value for each molecule',
                    style={'width': '100%', "textAlign": "center", "color": "#082446"}),
            html.Br(),
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
                            'text-align': 'center',
                            'font_size': '16px',
                        },
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'font-family': 'Arial',
                            'text-align': 'center',
                            'font_size': '16px',
                        }
                    )
                )
            ],
            style={
                'font-family': 'Arial',
                'width': '100%',
                "color": "#082446"
            }
        ),
            html.Div(
        [
            dbc.Button("Download CSV", id="btn_csv", n_clicks=0),
            dcc.Download(id="download-dataframe-csv"),
        ]
        )
        ]),result_df.to_json(orient='split')

@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    Input('predicted-ccs', 'data'),
    prevent_initial_call=True,
)

def func(n_clicks,df):
    df = pd.read_json(df, orient='split')
    if "btn_csv" == ctx.triggered_id and n_clicks > 0:   
        return dcc.send_data_frame(df.to_csv, "predicted_ccs.csv")

def feature_info():
    return (feature_header(),feature_layout())
