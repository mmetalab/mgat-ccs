# dash imports
import dash
from dash import html
from dash import Input
from dash import Output
from dash import dcc
import dash_bootstrap_components as dbc

# file imports
from maindash import app
from utils.file_operation import read_file_as_str


#######################################
# Layout
#######################################
def documentation_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1641160858304-6aded85fa2c4?q=80&w=1332&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
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
                        "Documentation",
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
            ),
            html.Br(),
            html.Div(
                style={"display": "flex"},
                children=[
                    dcc.Markdown(
                        children=read_file_as_str("./utils/markdown/documentation/documentation.md"),
                        mathjax=True,
                    ),
                ],
            ),
            html.Br(),
            html.Hr(),
            html.H3(
                "Procedure to peform CCS prediction using MGAT-CCS",
                style={"textAlign": "center", "color": "#082446"},
            ),
            html.Br(),
            html.Img(
                src="https://github.com/mmetalab/mgat-ccs/raw/main/images/tutorials.png",
                style={
                    "width": "1200px",
                    "display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                },
            ),
            html.Br(),
        ]
    )

    return layout

