import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

# from utils import compute_plot_gam
# from modeling import lrr, df, fb, col_map
# from modeling import dfTrain, dfTrainStd, dfTest, dfTestStd, yTrain, yTest

path = 'Apple-Twitter-Sentiment-DFE.csv'

apple_data = pd.read_csv(path, encoding='ISO-8859-1')
data = apple_data


def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 50}
    )

    return dbc.Row([dbc.Col(title, md=9), dbc.Col(logo, md=3)])


# def LabeledSelect(label, **kwargs):
#     return dbc.FormGroup([dbc.Label(label), dbc.Select(**kwargs)])


# Compute the explanation dataframe, GAM, and scores
# xdf = lrr.explain().rename(columns={"rule/numerical feature": "rule"})
# xPlot, yPlot, plotLine = compute_plot_gam(lrr, df, fb, df.columns)
# train_acc = accuracy_score(yTrain, lrr.predict(dfTrain, dfTrainStd))
# test_acc = accuracy_score(yTest, lrr.predict(dfTest, dfTestStd))

data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')

# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Card components
cards = [
    dbc.Card(
        [
            html.H2("13,700", className="card-title"),
            html.P("Total Reviews", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2("Last 30 Days", className="card-title"),
            html.P("Time Period", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2("3.6/5", className="card-title"),
            html.P("Overall Sentiment", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2("4", className="card-title"),
            html.P("Number of Data Sources", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),





]



# Graph components
# graphs = [
#     [
#         LabeledSelect(
#             id="select-coef",
#             options=[{"label": v, "value": k} for k, v in col_map.items()],
#             value=list(xPlot.keys())[0],
#             label="Filter Features",
#         ),
#         dcc.Graph(id="graph-coef"),
#     ],
#     [
#         LabeledSelect(
#             id="select-gam",
#             options=[{"label": col_map[k], "value": k} for k in xPlot.keys()],
#             value=list(xPlot.keys())[0],
#             label="Visualize GAM",
#         ),
#         dcc.Graph("graph-gam"),
#     ],
# ]

app.layout = dbc.Container(
    [
        Header("Cross Platform Customer Review Manager", app),
        dcc.Dropdown(
        id="dropdown",
        options=["Postive", "Neutral", "Negative"],
        value="Positive",
        clearable=False,
        ),

        html.Hr(),
        dbc.Row([dbc.Col(card) for card in cards]),


        html.Br(),

        dbc.Row([
            dbc.Col(
                html.Img(src=r'../assets/wordcloud.png', alt='image'),
            ),
            dbc.Col(
                dcc.Graph(id="graph"),
            )
        ])
        


        
        # dbc.Row([dbc.Col(graph) for graph in graphs]),

        


    ],
    fluid=False,
)



@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))
def update_bar_chart(type):
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')
    return fig


# @app.callback(
#     Output("graph", "figure"), 
#     Input("dropdown", "value"))
# def update_bar_chart(type):
#     data_canada = px.data.gapminder().query("country == 'Canada'")
#     fig = px.bar(data_canada, x='year', y='pop')
#     return fig


# @app.callback(
#     [Output("graph-gam", "figure"), Output("graph-coef", "figure")],
#     [Input("select-gam", "value"), Input("select-coef", "value")],
# )
# def update_figures(gam_col, coef_col):

#     # Filter based on chosen column
#     xdf_filt = xdf[xdf.rule.str.contains(coef_col)].copy()
#     xdf_filt["desc"] = "<br>" + xdf_filt.rule.str.replace("AND ", "AND<br>")
#     xdf_filt["condition"] = [
#         [r for r in r.split(" AND ") if coef_col in r][0] for r in xdf_filt.rule
#     ]

#     coef_fig = px.bar(
#         xdf_filt,
#         x="desc",
#         y="coefficient",
#         color="condition",
#         title="Rules Explanations",
#     )
#     coef_fig.update_xaxes(showticklabels=False)

#     if plotLine[gam_col]:
#         plot_fn = px.line
#     else:
#         plot_fn = px.bar

#     gam_fig = plot_fn(
#         x=xPlot[gam_col],
#         y=yPlot[gam_col],
#         title="Generalized additive model component",
#         labels={"x": gam_col, "y": "contribution to log-odds of Y=1"},
#     )

#     return gam_fig, coef_fig


if __name__ == "__main__":
    app.run_server(debug=True)
