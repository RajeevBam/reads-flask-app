"""Instantiate a Dash app."""

import os

from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot
import dash_bootstrap_components as dbc

from .layout import html_layout
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time
from sklearn.neural_network import MLPClassifier
os.getcwd()
os.listdir() 

print(os.getcwd())
print(os.listdir())

df = pd.read_csv("plotly_flask_tutorial/dashboard/electrochemicaldata-2.csv")


y=df["Cu Crystal"].values
x =df.drop(columns=['Day ', 'Cu Crystal', 'Microbe', 'Layer', 'OCP (mV vs. Ag/AgCl)', 'Grouped Rates', 'Grouped Rate Value', 'Data Set'] ,axis = 1)

column_names = list(x.columns)

data=(x,y)


X_train,X_test,Y_train,Y_test = train_test_split(*data, test_size=0.2, random_state=0)

def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5) # to let the print get out before any progress bars

shap.initjs()
nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(X_train, Y_train)


grouped = df.groupby('Grouped Rates').mean()


grouped_max= df.max()
grouped_min=df.min()
grouped_mean=df.mean()
grouped_mode=df.mode()
a = pd.DataFrame(grouped_max)
b=pd.DataFrame(grouped_min)
c=pd.DataFrame(grouped_mean)
a["1"]= b
a["2"]= c
a["3"]= grouped_mode.loc[0,:]


a = a.reset_index()
a.columns=["Name","maxValue","MinValue","Mean", "Mode"]

slider_card=html.Div([
        html.H3("Rsoln"),    
          html.Div(
            dcc.Slider(0, 10, 0.01,
                id='slider-Rsoln',
                marks={i: '{}'.format(0++i) for i in range(11)},
#                 value=2,
                updatemode='drag',
                
            ),style={"margin-bottom": "15px"}),
    
        html.H3("Rct"),    
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-Rct',
            marks={i: '{}'.format(0++i) for i in range(11)}, 
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
        html.H3("Rpo"),  
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-Rpo',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
        html.H3("Cc"),
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-Cc',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
    
        html.H3("m"),    
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-m',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
        html.H3("Cdl"),  
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-Cdl',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
        html.H3("Corrosion rate"), 
        html.Div(
        dcc.Slider(0, 10, 0.01,
            id='slider-Corrosion-rate',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        ),style={"margin-bottom": "15px"}),
    
        html.H3("Normaized Corrosion Rates"), 
        dcc.Slider(0, 10, 0.01,
            id='slider-Normaized-Corrosion-Rates',
            marks={i: '{}'.format(0++i) for i in range(11)},
#             value=2,
            updatemode='drag'
        )
                
])

graph_main=dbc.Card(dcc.Graph(id='updatemode-output-container'))


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = dbc.Container([

        dbc.Row(
            dbc.Col(html.H1("Data",
                            style={
                                'textAlign': 'center',

                            }))),

        dbc.Row(

            dbc.Col([
                html.P("Heatmap"),
                dcc.Dropdown(
                    id='heatmap_dropdown',
                    options=[{'label': i, 'value': i} for i in column_names],
                    value=["Rsoln"],
                    multi=True
                ),
                dcc.Graph(id="graph"),
            ],  # width={'size':5, 'offset':1, 'order':1},
                xs=12, sm=12, md=12, lg=5, xl=5
            ), justify='center'),  # Horizontal:start,center,end,between,around
        dbc.Row(

            dbc.Col([
                html.H3("Statistics"),
                dbc.Container([
                    dbc.Label('Click a cell in the table:'),
                    dash_table.DataTable(a.to_dict('records'), [
                        {"name": i, "id": i} for i in a.columns], id='tbl'),
                    dbc.Alert(id='tbl_out'),
                ])
            ], width={'size': 6, 'offset': 2, 'order': 1},
            )
        ),
        dbc.Row([
            dbc.Col([
                html.H4("Slider"),
                slider_card], width=4),
            dbc.Col([
                html.H4("Graph"),
                graph_main], width=8)
        ])

    ], fluid=True)
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(dash_app):
    @dash_app.callback(Output('updatemode-output-container', 'figure'),
                  Input('slider-Rsoln', 'value'),
                  Input('slider-Rct', 'value'),
                  Input('slider-Rpo', 'value'),
                  Input('slider-Cc', 'value'),
                  Input('slider-m', 'value'),
                  Input('slider-Cdl', 'value'),
                  Input('slider-Corrosion-rate', 'value'),
                  Input('slider-Normaized-Corrosion-Rates', 'value'))
    def display_value(Rsoln, Rct, Rpo, Cc, m, Cdl, Corrosion_rate, NCR):
        dicts = {'Rsoln': Rsoln,
                'Rct': Rct,
                'Rpo': Rpo,
                'Cc': Cc,
                'm': m,
                'Cdl': Cdl,
                'Corrosion rate': Corrosion_rate,
                'Normaized Corrosion Rates': NCR
                }

        dataDicts = pd.DataFrame([dicts])
       
        explainer = shap.Explainer(nn.predict, X_train)
        shap_values = explainer(dataDicts)
        array = []
        for i in shap_values.values[0]:
            array.append(i)
        array = np.array(array, dtype=float)
        fig = go.Figure(go.Waterfall(
            name="Gain and Loss",
            x=['Rsoln', 'Rct', 'Rpo', 'Cc', 'm', 'Cdl',
                'Corrosion rate', 'Normaized Corrosion Rates'],
            measure=["relative", "relative", "relative", "relative",
                    "relative", "relative", "relative", "relative"],
            y=array, base=0.5,  # shap_values.base_values[0],
            decreasing={"marker": {"color": "Maroon",
                                "line": {"color": "red", "width": 2}}},
            increasing={"marker": {"color": "Teal"}}
            #         totals={"marker":{"color": "deep sky blue","line":{"color": "blue","width":3}}},
        ))
        fig.update_layout(title="Shap Value Waterfall Chart",
                        waterfallgap=0.3, showlegend=True)
        return fig


    @dash_app.callback(
        Output("graph", "figure"),
        Input("heatmap_dropdown", "value"))
    def filter_heatmap(cols):
        fig1 = px.imshow(grouped[cols])
        return fig1


    @dash_app.callback(
        Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        return str(active_cell) if active_cell else "Click the table"
