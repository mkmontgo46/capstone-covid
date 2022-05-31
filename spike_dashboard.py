# Import libraries
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly_express as px
import plotly.graph_objects as go
import md_utils as mdu
import classification_utils as clu
import glycan_bionames
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --------------- Initialize Variables -----------------------

# Locations to read data from
data_dir = os.getenv('SPIKEDATASET_DIR')#'set in terminal with "export SPIKEDATASET=[dirname]"
if data_dir is None:
    sys.exit('Error: No spike dataset directory provided as an environment variable.')

traj_dirs = glob.glob(os.path.join(data_dir,'*TRAJECTOR*'))
traj_opts = []
for i in range(len(traj_dirs)):
    traj_opts.append({'label': traj_dirs[i], 'value': traj_dirs[i]})
    
# Options for types of features to include
feat_opts = [{'label':'RBD Distances','value':'RBD__2__'},
             {'label':'Radius of Gyration','value':'ROF'},
             {'label':'RMSD','value':'RMSD'},
             {'label':'x location','value':'_x'},
             {'label':'y location','value':'_y'},
             {'label':'z location','value':'_z'}]
feat_vals = ['RBD__2__','ROF','RMSD','_x','_y','_z']

# Define colors to use
closed_clr, open_clr = clu.get_label_colors()
label_cmap = {'Closed':closed_clr,'Open':open_clr}
cmap=clu.get_substruct_cmap()

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig
# -------------- Create dashboard --------------
app = Dash(__name__)

app.layout = html.Div(
    children=[
        # Title
        html.Div(children=[
                           html.Label(children='Spike',
                                style={'font-weight':'bold',
                                       'color':'#000080',
                                       'font-size':48}),
                           html.Label(children='Analytics',
                                   style={'font-weight':'bold',
                                          'color':'#ffa500',
                                          'font-size':48})
                          ],
                 style = {'display-content':'flex',
                          'justify-content':'flex-start'}),
       # Subtitle
       html.Label(children=' by the Amaro Lab at UC San Diego',
          style={'font-size':18}),
        html.Br(), html.Label(children=' '), html.Br(),
        html.Label(children="Predicting Effects of SARS-CoV-2 Variant Mutations on Spike Protein Dynamics and Mechanism",
                  style={'font-size':18}),
        
        
        html.Br(),        
        html.Div(children=[
            # Drop-down list for user to select feature sets
            html.H4('Select Profiled Data-sets to analyze:'),
            #html.Br(),
            dcc.Dropdown(id='featureset_select',
                         options= traj_dirs,
                         multi=True,
                         placeholder='Select at least 2 feature sets')
                        ],
                        style={'width': '45%', 'float': 'center', 'display': 'inline-block','padding': '0 20'}
                ),
        html.Br(),
        html.Br(),
        
        html.Div(children=[
            html.Div(children=[
                # Drop-down list for features to include
                html.Label('Select Features to Use:'),
                html.Br(),
                dcc.Dropdown(id='feature_select',
                             options=feat_opts,
                             multi=True,
                             value=feat_vals,
                             placeholder='Select which features to use'),
                            ],
                
            ),
            html.Div(children=[
                # Input option for max distance between glycans & rbd
                html.Label('Set Max RBD Neighborhood:'),
                html.Br(),
                dcc.Input(id='rbd_wind',
                          type='number',
                          min=0,
                          max=40,
                          step=1,
                          value=4),
                html.Label(' (nm)'),
            ]),
            html.Div(children=[
                # Input option for max correlation between two features
                html.Label('Set Max Feature Correlation:'),
                html.Br(),
                dcc.Input(id='corr_thresh',
                          type='number',
                          min=0,
                          max=1,
                          step=0.01,
                          value=0.5),
            ]),
            
        ],
                style={'display':'flex',
                       'justify-content':'space-around',
                       'align-items':'flex-start',
                       'column-gap':'50px'}
        ),
        
        html.Div(children=[
            html.Div(children=[
                # Feature Engineering
                html.Br(),
                html.Button('Preview Model Features',
                            id='feature_eng',
                            n_clicks=0,
                            disabled=True,
                            style = {'height':'35px'}
                            ),
                html.Br(),html.Label(' '),html.Br(),
                html.Div(children=[
                    html.Label('View:'),
                    html.Br(),
                    dcc.Dropdown(id='view_feat',
                         options= feat_opts,
                         multi=False,
                         disabled=True,
                         ),
                ],style={'width': '10%','float': 'center', 'display': 'inline-block','padding': '0 20'}
                        ),

                # Data loading indicator
                dcc.Loading(
                id="loading-feat",
                type="default",
                    # Histograms of Potential Features
                    children=dcc.Graph(id='feature_ext', figure = blank_fig(),
                    ),
                ),
                dbc.Popover(
                    [
                        dbc.PopoverHeader("Select at least 1 Open and 1 Closed DataSet"),
                    ],
                    id="popover",
                    is_open=False,
                    target="featureset_select",
                ),
                html.Br(),
                # Train Model button
                html.Button('Train Model',
                            id='train_go',
                            n_clicks=0,
                            disabled=True, 
                            style = {'height':'35px'}
                           ),
                # Model training indicator
                dcc.Loading(
                id="loading-1",
                type="default",
                    # Performance label
                    children=html.H4(id='performance_label', children ='',
                    ),
                ),
                ]
            ),
          
        ],

        
        ),
        html.Div(children=[
            # Plot bar graph of feature importances
            dcc.Graph(id='feat_imp', figure = blank_fig()),            
        ]),
        # Plot top feature over trajectory
        html.Div(children=[
            html.Br(),
            dcc.Loading(
                id='loading_trace',
                type='default',
                children=dcc.Graph(id='feat_trace',
                                   figure = blank_fig())
            )
        ]),

        
        # Plot open & closed spike with important features highlighted
        html.Div(children=[
            html.Br(),
            # Plot Scatter Button
            html.Button('Show Features in 3D',
                        id='scatter_go',
                        disabled=True,
                        style = {'height':'35px'}),
            # Scatter plotting indicator
            dcc.Loading(
                id='loading-scatter',
                type='default',
                
                # Scatter plots
                children = html.Div(children=[
                    dcc.Graph(id='spike1',
                               figure = blank_fig(),
                    style={'display': 'inline-block'}
                    ),
                    dcc.Graph(id='spike2',
                            figure=blank_fig(),
                    style={'display': 'inline-block'}
                    )
                ])
            )
        ]),
    ]
)

# -------------- Callbacks -----------------
# Enable Feature Engineering Callback
@app.callback([Output('feature_eng','disabled'),
               Output('view_feat','disabled')],
              [Input('featureset_select','value')],
              prevent_initial_call=True)
def enable_feature_engineering(traj_sel):
    '''Enable button to trigger feature engineering'''
    # Confirm at least 2 feature sets selected
    if traj_sel is None or len(traj_sel) < 2:
        return [True, True]
    else:
        return [False, False]
      
# Update feature to view list
@app.callback(Output('view_feat','options'),
              [Input('feature_select','value')],
              prevent_initial_call=True)
def update_view_list(feats):
    opts = []
    for f in feats:
        opts.append(feat_opts[feat_vals.index(f)])
    return opts

# Trigger Feature Engineering Callback
@app.callback(
              [
               Output('feature_ext','figure'),  
               Output('popover', 'is_open'),
               Output('train_go','disabled'),
               Output('view_feat','value')
              ],
               [
               Input('featureset_select','value'),
               State('popover', 'is_open'),
               Input('feature_select','value'),
               Input('rbd_wind','value'),
               Input('corr_thresh','value'),
               Input('feature_eng','n_clicks'),
               Input('view_feat','value')
              ],
               prevent_initial_call = True,
              )
def feature_Engineering(traj_sel,Iso,feat_sel,rbd_wind,corr_thresh,n_go,view_feat):
    '''Curate features and plot histograms'''
    # Confirm at least 2 feature sets selected
    if traj_sel is None or len(traj_sel) < 2:
        return [blank_fig(), True, True, view_feat]
    
    # Get list of files containing features
    feat_files = []
    for t in traj_sel:
        feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

        # For now, assume if dataset not labeled as "closed", is open
        is_open = ['closed' not in d for d in feat_files]
        
    # Confirm both open & closed data present
    if len(np.unique(is_open)) < 2:
        return [blank_fig(), True, True, view_feat]    
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    
    
    if buttonID == 'feature_eng':
        # Get feature to plot
        if view_feat is None:
            view_feat = feat_sel[0]
        
        # Load data
        print('Loading feature sets')
        global_state_df = clu.load_data(feat_files,is_open)
        global_state_df = clu.curate_feats(global_state_df,rbd_wind=rbd_wind,feat_incl=feat_sel,corr_thresh=corr_thresh)
        global_state_df.to_csv('./current_tmp_df.csv')
        print('Data loaded')
        
        # Plot feature histograms
        feat_stats_fig = clu.getfeatureStats(global_state_df,feat_incl=feat_sel,feat_type=view_feat)
        print('Feature Histograms Plotted')
        
        return [feat_stats_fig, False, False, view_feat]
#         return [feat_stats_fig_dict[feat_sel[0]], True, False]
    elif buttonID == 'view_feat':
        # Get feature to plot
        if view_feat is None:
            view_feat = feat_sel[0]
            
        # Plot feature histograms
        global_state_df  = pd.read_csv('./current_tmp_df.csv')
        global_state_df.drop(['Unnamed: 0'],inplace=True,axis=1)
        feat_stats_fig = clu.getfeatureStats(global_state_df,feat_incl=feat_sel,feat_type=view_feat)
        print('Feature Histograms Plotted')
        return [feat_stats_fig, False, False, view_feat]
    else:
        return [blank_fig(), False, True, view_feat]


# Train Model Callback
@app.callback(
              [
               Output('feat_imp','figure'),
               Output('performance_label','children'),
               Output('feat_imp','clickData')
              ],
               [
               Input('train_go','n_clicks'),   
              ],
               prevent_initial_call = True,
              )
def train_Spike_classifier_new(n_go):
    '''Train classifier and plot feature importances'''
    update = 'Return'
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    if buttonID=='train_go' and n_go > 0 :
        
        # Train Model
        global_state_df  = pd.read_csv('./current_tmp_df.csv')
        global_state_df.drop(['Unnamed: 0'],inplace=True,axis=1)
        tr_p, tr_r, ts_p, ts_r, df_feat = clu.train_sgd_model(global_state_df.copy())
        testResults = 'Test precision: ' + str(round(ts_p,3)) + ', Test recall: ' + str(round(ts_r,3))
        print(f'Training Completed : {testResults}')
        
        # Filter dataframe to top 10 features
        top_feats = df_feat[:10]['feats'].to_list() + ['Replicant','isopen','frame']    
        df = global_state_df[top_feats]
        df_feat = df_feat[:10]
        
        # Save data for use in other callbacks
        df.to_csv('./current_tmp_df.csv')
        df_feat.to_csv('./current_tmp_topfeats.csv')
        
        # Generate plot of feature importances
        feat_imp_fig = clu.plot_feature_importances(df_feat)
        
        # Set clickdata to trigger feature trace
        clickData = {'points':[{'x':glycan_bionames.rename_feat(glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'feat'))}]}
        
        return [feat_imp_fig,testResults, clickData]
    else:
             
        return [blank_fig(), {}, {}]
    
# Feature traces callback
@app.callback([Output('feat_trace','figure'),
               Output('scatter_go','disabled')],
              [Input('feat_imp','clickData')],
              prevent_initial_call=True)
def plot_feature_traces(clickData):
    '''Plot trace of all replicants for top feature'''
    
    # Load data
    df = pd.read_csv('./current_tmp_df.csv')
    df_feat = pd.read_csv('./current_tmp_topfeats.csv')
    
    # Determine feature to be plotted
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    if clickData:
        feat = clickData['points'][0]['x']
        feat_color = cmap[feat.split('at ')[-1]]
    
        # Plot
        feat_trace = clu.trace_single_feat(df,feat,feat_color,label_cmap)
    
        return [feat_trace, False]
    else:
        return [blank_fig(), True]
    
# Show Features in 3D Callback
@app.callback([Output('spike1','figure'),
               Output('spike2','figure')],
              [Input('featureset_select','value'),
               Input('scatter_go','n_clicks'),
               Input('spike1','figure'),
               Input('spike2','figure'),
               Input('spike1','relayoutData'),
               Input('spike2','relayoutData')],
              prevent_initial_call=True)
def scatterplot_trajectories(traj_sel,n_go,spike1_fig,spike2_fig,scene1,scene2):
    '''Load trajectories and plot in 3D'''
    if n_go is None or n_go == 0:
        return [spike1_fig, spike2_fig]
    
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
       
    if buttonID == 'scatter_go':
        # Load important features
        df_feat = pd.read_csv('./current_tmp_topfeats.csv')
        
        # Load trajectories - for now load first directories with and without 'closed' in the name
        print('Loading trajectories...')
        closed_dirs = traj_sel[['closed' in i for i in traj_sel].index(True)]
        open_dirs = traj_sel[['closed' in i for i in traj_sel].index(False)]
        traj_open = mdu.load_traj(open_dirs)
        traj_closed = mdu.load_traj(closed_dirs)

        # Parse trajectories
        print('Parsing trajectories...')
        atom_id_open = mdu.parse_traj(traj_open)
        atom_id_closed = mdu.parse_traj(traj_closed)

        # Create figures
        spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, df_feat,'Closed Spike',closed_clr)
        spike2_fig = mdu.viz_traj(traj_open,atom_id_open, df_feat,'Open Spike',open_clr)
        
        return [spike1_fig, spike2_fig]
    elif buttonID == 'spike1':
        # Align cameras to fig 1
        cam1 = scene1['scene.camera']
        spike1_fig['layout']['scene']['camera'] = cam1
        spike2_fig['layout']['scene']['camera'] = cam1
        
    elif buttonID == 'spike2':
        # Align cameras to fig 2
        cam2 = scene2['scene.camera']
        spike1_fig['layout']['scene']['camera'] = cam2
        spike2_fig['layout']['scene']['camera'] = cam2

    return [spike1_fig, spike2_fig]


# Run app
if __name__ == "__main__":
     app.run_server(host='0.0.0.0',debug=True, port=8050)