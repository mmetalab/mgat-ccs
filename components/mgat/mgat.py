from dash import html, Input, Output
import dash_bootstrap_components as dbc
from maindash import app
from components.mgat.load import load_info
from components.mgat.feature import feature_info

#######################################
# Layout
#######################################
def mgat_layout():
    layout = html.Div(
        [
            # image
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1614851099511-773084f6911d?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
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
                        "MGAT-CCS Prediction",
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
            # tab
            html.Div(
                style={"display": "flex"},
                children=[
                    html.Div(
                        [
                            dbc.Tabs(
                                id="mgat_analysis_selected_tab",
                                children=[
                                    dbc.Tab(
                                        label="Load Data and Molecular Featurization",
                                        tab_id="analysis_load",
                                    ),
                                    dbc.Tab(
                                        label="CCS Value Prediction",
                                        tab_id="analysis_feature",
                                    ),       
                                ],
                                active_tab="analysis_load",
                            ),
                        ]
                    ),
                ],
            ),
            html.Br(),
            # content: analysis & plot
            html.Div(
                style={"display": "flex"},
                children=[
                    html.Div(
                        style={
                            "width": "30%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="mgat_analysis_tab_1_layout"),
                        ],
                    ),
                    html.Div(
                        style={
                            "width": "70%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="mgat_analysis_tab_2_layout"),
                        ],
                    ),
                ],
            ),
            html.Br()
        ]
    )

    return layout

#######################################
# Callbacks
#######################################
@app.callback(
    [
        Output(
            component_id="mgat_analysis_tab_1_layout", component_property="children"
        ),
        Output(component_id="mgat_analysis_tab_2_layout", component_property="children")
    ],
    [Input(component_id="mgat_analysis_selected_tab", component_property="active_tab")],
)

def render_tab(tab_choice):
    """Renders the selected subtab's layout

    Args:
        tab_choice (str): selected subtab

    Returns:
        selected subtab's layout
    """
    if tab_choice == "analysis_load":
        return load_info()
    if tab_choice == "analysis_feature":
        return feature_info()


