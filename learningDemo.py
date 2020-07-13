import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import numba
import time
import flask
import os
import math
import random
import base64
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

dt = 0.001  
T = 1.0   
t = np.arange(0,T,dt)

outputArr = np.zeros((t.size,1),dtype=float, order='C');
inputArr = np.zeros((t.size,1),dtype=float, order='C'); 
endrawn = np.zeros((t.size,1),dtype=float, order='C'); 
endrawn = endrawn.flatten()
W = np.random.rand(1,1)

learningRate = 0.01    
learning = 1
nTrials = 20
learnSparse = 0
CSDistr = 0

tau = 0
gwidth = 250


@numba.jit()
def learnPattern():
    W2 = W;   
    output_i = 0
    outputHolder = np.zeros((1, t.size),dtype=float, order='C');
    vectorHolder = np.zeros((1, t.size),dtype=float, order='C');
    inds = np.arange(tau,t.size,1)
    ir = inputArr
    
    if learnSparse:
        i = random.randint(tau,t.size-1)
        
        if CSDistr:
            for i in inds:
                W_resolved = ir[i-tau]*W2
                output_i = sum(W_resolved)
                outputHolder[0][i] = output_i
            E = (outputHolder - outputArr)*outputHolder
            E = np.abs(E)
            Esum = np.cumsum(E)
            Esel = np.random.uniform(0,E.sum(),1)
            Epos = np.argmin(np.abs((Esum - Esel)))
            i = Epos 
            print("Selector was: " +str(Esel) + " at postion: " + str(Epos))
            

        # W_resolved = np.matmul(inputArr[i-tau],W2)
        W_resolved = ir[i-tau]*W2
        output_i = sum(W_resolved)
        
        if learning:
            Error  = (output_i - outputArr[i])
            # W2 = W2 - np.matmul(Error,W_resolved)*learningRate
            W2 = W2 - ((Error*W_resolved)*(learningRate*100))
            nanfinder = np.argwhere(np.isnan(W2))
            W2[nanfinder] = 0
            vectorHolder[0][i] = -(Error*W_resolved) 
    
    
    else:
        for i in inds:
            # W_resolved = np.matmul(inputArr[i-tau],W2)
            W_resolved = ir[i-tau]*W2
            output_i = sum(W_resolved)
            
            if learning:
                Error  = (output_i - outputArr[i])
                # W2 = W2 - np.matmul(Error,W_resolved)*learningRate
                W2 = W2 - ((Error*W_resolved)*learningRate)
                nanfinder = np.argwhere(np.isnan(W2))
                W2[nanfinder] = 0
                vectorHolder[0][i] = -(Error*W_resolved) 
        
    for i in inds:
        W_resolved = ir[i-tau]*W2
        output_i = sum(W_resolved)
        outputHolder[0][i] = output_i
        
    return W2, outputHolder, vectorHolder  
    
@numba.jit()   
def updateInput(intype):
    in2 = np.zeros((t.size,1),dtype=float, order='C');
    if "COS" in intype:
        in2 = (np.cos(t*np.pi/.20))+1
        
    if "SIN" in intype:
        in2 = (np.sin(t*np.pi/.20))+1
    
    if "GAUSS" in intype:
        v = gwidth
        m = T/2
        in2 = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        
    if "2G" in intype:
        v = gwidth
        m = T/4*3
        A = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        m = T/4*1
        B = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        in2 = A + B
  
    in2 = in2/in2.max()
    if "FLAT" in intype:
        in2 = t*0 + .5;
        
    if "FTR" in intype:
        in2 =  t*0;
        in2[100:500] = .5
    
    if "CUS" in intype:
        in2 =  endrawn
    
    return in2

@numba.jit()   
def updateOutput(outtype):
    out2 = np.zeros((t.size,1),dtype=float, order='C');
       
    if "COS" in outtype:
        out2 = (np.cos(t*np.pi/.20))+1
        
    if "SIN" in outtype:
        out2 = (np.sin(t*np.pi/.20))+1
    
    if "GAUSS" in outtype:
        v = gwidth
        m = T/2
        out2 = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        
    if "2G" in outtype:
        v = gwidth
        m = T/4*3
        A = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        m = T/4*1
        B = np.exp(-np.square(t-m)/2*v)/(np.sqrt(2*np.pi*v))
        out2 = A + B
    
    out2 = out2/out2.max()
    if "FLAT" in outtype:
        out2 =  t*0 + .5;
        
    if "FTR" in outtype:
        out2 =  t*0;
        out2[100:500] = .5
        
    if "CUS" in outtype:
        out2 =  endrawn
        
    return out2
    
def updateW(W2):
    global W
    W = W2
    
def repatW():
    global W
    W = np.random.rand(1,1)
    # Wresolved = W * inputArr;
    # W = W/np.mean(Wresolved)
    

@numba.jit()     
def updateFigs():
    F1 = go.Figure()
    A = t
    B = inputArr
    F1.add_trace(go.Scatter(x = A, y = B, mode='lines',name="Input Signal" ,line=dict(color='blue')))
    B2 = np.concatenate((inputArr[:tau]*0,inputArr[:inputArr.size-tau]))
    F1.add_trace(go.Scatter(x = A, y = B2, mode='lines',name="Input with Delay",line=dict(color='red')))
    F1.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
    
    F2 = go.Figure()
    A = t
    B = outputArr
    F2.add_trace(go.Scatter(x = A, y = B, mode='lines',name="Desired Output",line=dict(color='green')))
    B2 = np.concatenate((inputArr[:tau]*0,inputArr[:inputArr.size-tau]))
    F2.add_trace(go.Scatter(x = A, y = B2, mode='lines',name="Input with Delay",line=dict(color='red')))
    F2.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
    
    F3 = makeF3()
    
    F3.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
    
    return F1, F2, F3
    
def makeF3():
    global learning
    F3 = go.Figure(data= [], frames =[])
    nFrames = nTrials
    frameArr = []

    for i in range(0,nFrames):
        if i == 1:
            learning = 0
        else:
            learning = 1
        v2 = []
        W2, out2, v2 = learnPattern()
        updateW(W2)
        gs = go.Frame(data=[
        go.Scatter(x=t, y=out2.flatten(),name="Predicted Output",line=dict(color='black')), 
        go.Scatter(x=t, y=outputArr,name="Desired Output",line=dict(color='blue')),
        go.Scatter(x=t, y=v2.flatten(),line=dict(color='green'), name="Positive Delta W vector"),
        go.Scatter(x=t, y=-v2.flatten(),line=dict(color='red'), name="Negative Delta W vector")
        ],
        traces=[0,1,2,3],name='frame{}'.format(i))
        frameArr.append(gs)
        
        if i == 0:
            data=[
            go.Scatter(x=t, y=out2.flatten(),name="Predicted Output",line=dict(color='black')), 
            go.Scatter(x=t, y=outputArr,name="Desired Output",line=dict(color='blue')),
            go.Scatter(x=t, y=v2.flatten(),line=dict(color='green'), name="Positive Delta W vector"),
            go.Scatter(x=t, y=-v2.flatten(),line=dict(color='red'), name="Negative Delta W vector")
            ]
    
    
    lo = go.Layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args= [None, {"frame": {"redraw": True},
                                "fromcurrent": False, "transition": {"duration": 25,
                                                                    "easing": "none"}}],
                          )
                          ])
                          ])
      
    F3 = go.Figure(data= data, frames =frameArr, layout = lo)
    F3.update_layout(xaxis = dict(
        range=(0, 1),
        constrain='domain'
    ),
     yaxis = dict(
        range=(0,1),
        constrain='domain'
    )
    )  
    return F3
 
# Callback: ----------------------------------------
  
@app.callback(
    [
    Output('input-fig', 'figure'),
    Output('output-fig', 'figure'),
    Output('learning-fig', 'figure')
    ],
    [
    Input('re-button', 'n_clicks'),
    ],
    [
    State('input-dropdown', 'value'),
    State('output-dropdown', 'value'),
    State('tau-input', 'value'),
    State('LR-input', 'value'),
    State('trials-input', 'value'),
    State(component_id='CS-set', component_property='value'),
    State(component_id='CSerr-set', component_property='value')  
    ]
)
def updateParams(dummy,intype,outtype,tau_in,LR,nTR,CS,PCS):
    tstart = time.time()
    global tau
    tau = tau_in 
    
    global learningRate
    learningRate = LR

    global inputArr
    inputArr = updateInput(intype)

    global outputArr
    outputArr = updateOutput(outtype)
    
    global nTrials
    nTrials = nTR
    
    global learnSparse
    learnSparse = 0
    if CS:
        learnSparse = 1 
    repatW()
    
    global CSDistr
    CSDistr = 0
    if CS:
        CSDistr = 1 
        
    repatW()
    F3 = [];
    F1,F2,F3 = updateFigs()

    print(time.time() - tstart)
    
    return F1, F2, F3
    
@app.callback(Output('upload', 'children'),
              [Input('upload', 'contents')],
              [State('upload', 'filename'),
               State('upload', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    global endrawn
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names, list_of_dates)
        print(df)
        print(df.iloc[:,0].values)
        A = df.iloc[:,0].values
        A = np.atleast_2d(A)
        A = A.flatten()
        
        Al = A.size
        if Al >= 1000:
            Al = 999
            A = A[:Al]
        endrawn = np.zeros((t.size,1),dtype=float, order='C'); 
        endrawn = endrawn.flatten()
        endrawn[:Al] = A[:Al]
        endrawn = endrawn.flatten()
        endrawn = endrawn - np.min(endrawn)
        print(endrawn)
        
        children=html.Div([
            'For Custom Data: Drag and Drop or ',
            html.A('Select Files')
        ]),
        return children
    


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        
    

    
# End Callback ----------------------------------------- 

app.layout = html.Div([
    html.Div([
    
        html.Div([
            html.P('Input Signal Type:'),
            dcc.Dropdown(
                id = 'input-dropdown',
                options =
                [
                    {'label': 'Cosine', 'value': 'COS'},
                    {'label': 'Sine', 'value': 'SIN'},
                    {'label': 'Gaussian', 'value': 'GAUSS'},
                    {'label': 'Double Peaked Gaussian', 'value': '2G'},
                    {'label': 'Flat Signal', 'value': 'FLAT'},
                    {'label': 'Flat Transient', 'value': 'FTR'},
                    {'label': 'Uploaded Function', 'value': 'CUS'}
                ],
                value='COS'
            ),    
        ], className="one-third column" 
        ),
        
        html.Div([
            html.P('Desired Signal Type:'),
            dcc.Dropdown(
                id = 'output-dropdown',
                options =
                [
                    {'label': 'Cosine', 'value': 'COS'},
                    {'label': 'Sine', 'value': 'SIN'},
                    {'label': 'Gaussian', 'value': 'GAUSS'},
                    {'label': 'Double Peaked Gaussian', 'value': '2G'},
                    {'label': 'Flat Signal', 'value': 'FLAT'},
                    {'label': 'Flat Transient', 'value': 'FTR'},
                    {'label': 'Uploaded Function', 'value': 'CUS'}
                ],
                value = 'COS'
            ),
        ], className="one-third column"
        ),

    ],style = {'display': 'inline-block', 'width': '70%'}
    ),
    
    html.Div([
        html.Div([
            dcc.Graph
                (
                id='input-fig',
                )
        ], className="one-third column"
        ),
        
        html.Div([
            dcc.Graph
                (
                    id='output-fig',
                ),  
        ], className="one-third column"
        ),
        
        html.Div([
            dcc.Graph
                (
                    id='learning-fig',
                ),  
        ], className="one-third column"
        ),
        
    ],
    style = {'display': 'inline-block', 'width': '70%'},
    ),
    
    html.Div([
         html.Div([
                html.H6("Learning parameters:"),
                html.P('Prediction Delay:'),
                html.Div([dcc.Input(id='tau-input', value=150,type='number',debounce = True)]),
                html.Br(),
                html.P('Learning Rate:'),
                html.Div([dcc.Input(id='LR-input', value=0.001,type='number',debounce = True,step = 0.001)]),
                html.Br(),
                html.P('Number of Trials:'),
                html.Div([dcc.Input(id='trials-input', value=20,type='number',debounce = True,step = 1)]),
                html.Br(),
                html.P('Complex Spikes Features:'),
                html.Div([
                    dcc.Checklist(
                        options=[
                            {'label': 'Sparse CS', 'value': 'SCS'}
                        ],
                        value=[],
                        id = 'CS-set',
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'P(CD) linked to Error Probability', 'value': 'CSE'}
                        ],
                        value=[],
                        id = 'CSerr-set',
                    )                    
                    ]),
                html.Br(),
                html.Button('Submit', id='re-button', n_clicks=0),
                ], className="one-third column"
        ),
        ],
    style = {'display': 'inline-block', 'width': '80%'},
    className="row"
    ),
        html.Div([
    dcc.Upload(
        id='upload',
        children=html.Div([
            'For Custom Data: Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '40%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'left',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])
])

app.css.append_css
({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)