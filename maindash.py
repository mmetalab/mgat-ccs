import dash
import dash_bootstrap_components as dbc
import pandas as pd

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
)
app.title = "MGAT-CCS"
server = app.server