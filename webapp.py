# dash imports
import dash
import json
from dash import html
from dash import Input
from dash import Output
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc, State, Dash, dash_table, callback, ctx
# file imports
from maindash import app
from components.overview import overview
from components.mgat import mgat
from components.documentation import documentation
from components.about import about

#######################################
# Initial Settings
#######################################
server = app.server

CONTENT_STYLE = {
    "transition": "margin-left .1s",
    "padding": "1rem 1rem",
}

#######################################
# Layout
########################################
sidebar = html.Div(
    [
        html.Div(
            [
                html.H2("MGAT-CCS", style={"color": "white"}),
            ],
            className="sidebar-header",
        ),
        html.Br(),
        html.Div(style={"border-top": "2px solid white"}),
        html.Br(),
        # nav component
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-star me-2"),
                        html.Span("Overview"),
                    ],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-line-chart me-2"),
                        html.Span("MGAT-CCS Prediction"),
                    ],
                    href="/mgat",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa fa-book me-2"),
                        html.Span("Documentation"),
                    ],
                    href="/documentation",
                    active="exact",
                ),

                dbc.NavLink(
                    [
                        html.I(className="fas fa-user me-2"),
                        html.Span("About"),
                    ],
                    href="/about",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        dcc.Store(id='loaded-data',storage_type='local',data={}),
        dcc.Store(id='loaded-data-input',storage_type='local',data={}),
        dcc.Store(id='processed-feature',storage_type='local',data={}),
        dcc.Store(id='predicted-ccs',storage_type='local',data={}),
        html.Div(
            [
                dash.page_container,
            ],
            className="content",
            style=CONTENT_STYLE,
            id="page-content",
        ),
    ]
)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
        return overview.overview_layout()
    elif pathname == "/mgat":
        return mgat.mgat_layout()
    elif pathname == "/documentation":
        return documentation.documentation_layout()
    elif pathname == "/about":
        return about.about_layout()
    return dbc.Container(
        children=[
            html.H1(
                "404 Error: Page Not found",
                style={"textAlign": "center", "color": "#082446"},
            ),
            html.Br(),
            html.P(
                f"Oh no! The pathname '{pathname}' was not recognised...",
                style={"textAlign": "center"},
            ),
            # image
            html.Div(
                style={"display": "flex", "justifyContent": "center"},
                children=[
                    html.Img(
                        src="https://elephant.art/wp-content/uploads/2020/02/gu_announcement_01-1.jpg",
                        alt="hokie",
                        style={"width": "400px"},
                    ),
                ],
            ),
        ]
    )

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080)
