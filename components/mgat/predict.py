import plotly.express as px
from dash import html, Input, Output, dcc, State, Dash, dash_table, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_uploader as du
import base64
import datetime
import io
# file imports
from maindash import my_app
from utils.file_operation import read_file_as_str

def predict_header():
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
                    "CCS Prediction",
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

def predict_layout():
    layout = html.Div(
        [
            html.Br(),
            html.H3(
                "\nPerform CCS prediction",
                style={"textAlign": "center", "color": "#082446"},
            ),
            dbc.Button('Click to Run', id='btn-nclicks-ccs', n_clicks=0),
            html.Div(id='container-button-ccs')
        ]
    )
    return layout

@my_app.callback(Output('print-data-upload', 'children'),
              Input('intermediate-value', 'data'))

def update_graph(uploaded_df):
    # more generally, this line would be
    df = pd.read_json(uploaded_df, orient='split')
    print(df.head())
    return html.Div([
        html.Hr(),  # horizontal line
        html.H5('The first few lines of the uploaded molecule data.'),
        dbc.Container(
        [
            dbc.Spinner(
                dash_table.DataTable(
                    df.to_dict('records'),
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

@my_app.callback(
    Output('container-button-ccs', 'children'),
    Input('btn-nclicks-ccs', 'n_clicks')
)

def displayClick(btn1):
    msg = "CCS prediction is not yet performed. Click the button to run featurization."
    hidden = True
    if "btn-nclicks-ccs" == ctx.triggered_id:
        msg = "CCS prediction is performed." 
        hidden = False
    return html.Div(
            [html.Div(msg),
             html.Div(id='print-data-upload', hidden=hidden)
            ]
        )

def predict_info():
    return (predict_header(),predict_layout())