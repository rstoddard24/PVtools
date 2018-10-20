#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:25:41 2018

@author: ryanstoddard
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from dateutil import parser
import pandas as pd
import numpy as np

import sys
sys.path.append('../../')
from PVtools.PL import PLtools

def model(theta=1.5,gam=37,Eg1=1.7,Eg2=1.8,x1=.03,QFLS=1.3,T=100):
    """
    Model two phase PL
    """
    theta = float(theta)
    gam = float(gam)
    Eg1 = float(Eg1)
    Eg2 = float(Eg2)
    x1 = float(x1)
    QFLS = float(QFLS)
    T = float(T)
    Emin = np.mean((Eg1,Eg2))-.15
    Emax = np.mean((Eg1,Eg2))+.15
    E = np.linspace(Emin,Emax,num=100)
    AIPL = PLtools.LSWK_2phase_gfunc(E,theta,gam/1000,Eg1,Eg2,x1,QFLS,T)
    
    return (E, np.exp(AIPL))





app = dash.Dash()
css_url = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css"
app.css.append_css({
    "external_url": css_url
})


app.layout = html.Div(children=[
    html.H1(children='PL model of 2-Phase Nanostructure'),

    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.P("Eg1 [eV]")
            ], className='row'),
            html.Div(children=[
                html.P("Eg2 [eV]")
            ], className='row'),
            html.Div(children=[
                html.P("x1")
            ], className='row'),
            html.Div(children=[
                html.P("theta")
            ], className='row'),
            html.Div(children=[
                html.P("gamma [meV]")
            ], className='row'),
            html.Div(children=[
                html.P("QFLS [eV]")
            ], className='row'),
            html.Div(children=[
                html.P("T [K]")
            ], className='row')
        ], className='col'),
    
        html.Div(children=[
            html.Div(children=[
                dcc.Input(
                    id='Eg1',
                    placeholder='Eg1',
                    type='float',
                    value='1.7'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='Eg2',
                    placeholder='Eg2',
                    type='float',
                    value='1.8'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='x1',
                    placeholder='x1',
                    type='float',
                    value='.03'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='theta',
                    placeholder='theta',
                    type='float',
                    value='1.5'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='gamma',
                    placeholder='gamma',
                    type='float',
                    value='37'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='QFLS',
                    placeholder='QFLS',
                    type='float',
                    value='1.3'
                ),
            ], className='row'),
            html.Div(children=[
                dcc.Input(
                    id='T',
                    placeholder='T',
                    type='float',
                    value='300'
                ),
            ], className='row')
        ], className='col')
    ], className='row'),

    html.Div(children=[
        html.Div(children=[
            html.Button('Submit', id="final-submit-button", className="btn btn-primary btn-lg"),
        ], className='row', style={'text-align': 'center'})
    ], className='col'),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(id='graph-linear'),
        ], className='col', style={'text-align': 'center'}),
        html.Div(children=[
            dcc.Graph(id='graph-log')
        ], className='row', style={'text-align': 'center'}),
    ], className='col'),
], className='container', style={'margin-top': 25})

@app.callback(
    Output('graph-linear', 'figure'),
    [Input('final-submit-button', 'n_clicks')],
    [State('Eg1', 'value'),
     State('Eg2', 'value'),
     State('x1', 'value'),
     State('theta', 'value'),
     State('gamma', 'value'),
     State('QFLS', 'value'),
     State('T', 'value')]
)
def update_graph_linear(_, Eg1, Eg2, x1, theta, gamma, QFLS, T):
    (E, AIPL) = model(theta,gamma,Eg1,Eg2,x1,QFLS,T)
    
    traces = []
    
    traces.append(go.Scatter(
        x=E,
        y=AIPL,
        mode="lines",
        
    ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'E [eV]'},
            yaxis={'title': 'AIPL [photons/m^2-eV-s]'},
            height=650,
            width=500
        )
    }



@app.callback(
    Output('graph-log', 'figure'),
    [Input('final-submit-button', 'n_clicks')],
    [State('Eg1', 'value'),
     State('Eg2', 'value'),
     State('x1', 'value'),
     State('theta', 'value'),
     State('gamma', 'value'),
     State('QFLS', 'value'),
     State('T', 'value')]
)
def update_graph_log(_, Eg1, Eg2, x1, theta, gamma, QFLS, T):
    (E, AIPL) = model(theta,gamma,Eg1,Eg2,x1,QFLS,T)
    
    traces = []
    
    traces.append(go.Scatter(
        x=E,
        y=AIPL,
        mode="lines",
        
    ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'E [eV]'},
            yaxis={'title': 'AIPL [photons/m^2-eV-s]','type': 'log'},
            height=650,
            width=500
        )
    }


if __name__ == '__main__':
    app.run_server()
