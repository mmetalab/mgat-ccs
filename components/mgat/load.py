import plotly.express as px
from dash import html, Input, Output, dcc, State, Dash, dash_table, callback
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
import pandas as pd


def load_header():
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
                                "height": "300px",
                                "overflow": "hidden"
                            },
                        ),
                    ],
                )
            ],
        )
    return layout

def load_preset_layout():
    layout = html.Div(
        [
            html.H3(
                "Load Data for CCS Prediction",
                style={'width': '200%', "textAlign": "center", "color": "#082446"},
            ),
            html.Br(),
            html.H5(
                "\nInput the molecule data",
                style={'width': '200%', "textAlign": "center", "color": "#082446"},
            ),
            html.Div([
                dcc.Textarea(
                    id='textarea-state-example',
                    value='Compound Name/ID,SMILE\nDaidzein,C1=CC(=CC=C1C2=COC3=C(C2=O)C=CC(=C3)O)O',
                    style={'width': '200%', 'height': 50},
                ),
                dbc.Button('Submit', id='textarea-state-example-button', n_clicks=0),
                html.Div(id='textarea-state-example-output')
            ]),
            load_layout()
        ])
    return layout

# Create a callback to update the dataframe when the button is clicked
@my_app.callback(Output('loaded-data-input', 'data'),
    Input('textarea-state-example-button', 'n_clicks'),
    State('textarea-state-example', 'value'))

def update_output(n_clicks, value):
    if n_clicks > 0:
        df = pd.DataFrame(columns = ['Name', 'Age'])
        for i in value.split('\n'):
            i = i.split(',')
            row = pd.DataFrame({'Name': i[0], 'Age': i[1]}, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
        return df.to_json(orient='split')

@my_app.callback(Output('textarea-state-example-output', 'children'),
    Input('loaded-data-input', 'data'))

def update_graph_tt(uploaded_df):
    # more generally, this line would be
    df = pd.read_json(uploaded_df, orient='split')
    print(df)
    return html.Div([
        html.Hr(),  # horizontal line
        html.H5('The first few lines of the uploaded molecule data.'),
        dbc.Container(
        [
            dbc.Spinner(
                dash_table.DataTable(
                    df.to_dict('records'),
                    id='dash-table-input',
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


def load_layout():
    layout = html.Div(
        [
            html.Br(),
            html.H5(
                "\nUpload the molecule data file",
                style={'width': '200%',"textAlign": "center", "color": "#082446"},
            ),
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '200%',
                        'height': '50px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
            ])
        ]
    )

    return layout

def parse_contents(contents):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in contents:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in contents:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
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

def parse_contents_df(contents):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in contents:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in contents:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

@my_app.callback(Output('loaded-data', 'data'),
              Input('upload-data', 'contents'))

def update_output(list_of_contents):
    if list_of_contents is not None:
        tt = [
            parse_contents_df(c) for c in list_of_contents]
        return tt[0].to_json(orient='split')

# @my_app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'))

@my_app.callback(Output('output-data-upload', 'children'),
              Input('loaded-data', 'data'))

def update_graph(uploaded_df):
    # more generally, this line would be
    df = pd.read_json(uploaded_df, orient='split')
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



def load_info():
    return (load_header(),load_preset_layout())


