# Import libraries
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly_express as px
import plotly.graph_objects as go
import md_utils as mdu
import classification_utils as clu
import glycan_bionames
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dash_bootstrap_components as dbc


# --------------- Initialize Variables -----------------------

# Locations to read data from
# data_dir = './'
data_dir = '/net/jam-amaro-shared/dse_project/Spike_Dataset/'

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
       html.Label(children=' by the Amaro Lab at UC San Diego',
          style={'font-size':18}),
        html.Br(), html.Label(children=' '), html.Br(),
        html.Label(children="Predicting Effects of SARS-CoV-2 Variant Mutations on Spike Protein Dynamics and Mechanism",
                  style={'font-weight':'italic',
                         'font-size':18}),
        html.Br(),
        
        #html.H3(id='update_window',
        #       children='Please select feature sets to begin',
               #style={'font-weight':'bold',
               #       'background-color':'#ebecf0',
               #      'width':'33%'}
        #        style={'width': '49%', 'display': 'inline-block'}
        #    ),
        
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
                html.Label('Select attributes for feature Engineering:'),
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
                html.Label('Glycans to RBD lookout:'),
                html.Br(),
                dcc.Input(id='rbd_wind',
                          type='number',
                          min=0,
                          max=40,
                          step=1,
                          value=8),
                html.Label(' (nm)'),
            ]),
            html.Div(children=[
                # Input option for max correlation between two features
                html.Label('Feature correlation threshold:'),
                html.Br(),
                dcc.Input(id='corr_thresh',
                          type='number',
                          min=0,
                          max=1,
                          step=0.01,
                          value=0.5),
            ]),
            
        ],
        #style={'width': '25%', 'display': 'inline-block','padding': '0 20'}
        #style={'width': '25%', 'display': 'inline-block'},
                style={'display':'flex',
                       'justify-content':'space-around',
                       'align-items':'flex-start',
                       'column-gap':'50px'}
        ),
        
        html.Div(children=[
            html.Div(children=[
                # Feature Engineering
                html.Br(),
                #html.Br(),
                html.Button('Trigger Feature Engineering',
                            id='feature_eng',
                            n_clicks=0,
                            disabled=False,
                            
                            ),
                # Data loading indicator
                dcc.Loading(
                id="loading-feat",
                type="default",
                    # Histograms of Potential Features
                    children=dcc.Graph(id='feature_ext', figure = blank_fig(),
                    ),
                ),
                html.Br(),
                # Train Model button
                html.Button('Train Model',
                            id='train_go',
                            n_clicks=0,
                            disabled=False, 
                            
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
        #style={'display':'flex',
        #               'justify-content':'space-around',
        #               'align-items':'flex-start',
        #               'column-gap':'50px'},
        
        ),
        html.Div(children=[
            # Feature Stats
            #dcc.Graph(id='feat_trace', figure = blank_fig()),
            # Plot bar graph of feature importances
            dcc.Graph(id='feat_imp', figure = blank_fig()),
            
            
            
        ]),
        # Plot top feature over trajectory
        html.Div(children=[
            html.Br(),
            html.Button('Plot top feature over time',
                        id='trace_go'),
            dcc.Loading(
                id='loading_trace',
                type='default',
                children=dcc.Graph(id='feat_trace',
                                   figure = blank_fig())
            )
        ]),
#         # Button to do everything else
#         html.Button('Map Important Features',id='go',n_clicks = 0,
#                    style = {'height':'35px'}),
        
        # Plot open & closed spike with important features highlighted
        html.Div(children=[
            html.Br(),
            # Plot Scatter Button
            html.Button('Show Features in 3D',
                        id='scatter_go'),
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
        
        

        # Store variables
        dcc.Store(id='df'),
        #dcc.Store(id='global_state_df'),
        dcc.Store(id='df_feat')
    ]
)

# -------------- Callbacks -----------------
# Trigger Feature Engineering Callback
@app.callback(
              [
               Output('feature_ext','figure'),   
               #Output('global_state_df','data')
              ],
               [
               Input('featureset_select','value'),
               Input('feature_select','value'),
               Input('rbd_wind','value'),
               Input('corr_thresh','value'),
               Input('feature_eng','n_clicks')       
              ],
               prevent_initial_call = True,
              )
def feature_Engineering(traj_sel,feat_sel,rbd_wind,corr_thresh,n_go):
    '''Curate features and plot histograms'''
    update = 'Return'
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    if buttonID == 'feature_eng':
        # Confirm at least 2 feature sets selected
        if traj_sel is None or len(traj_sel) < 2:
            update = 'Please select at least 2 feature sets!'
            
            return [blank_fig()]
        
        # Get list of csv files containing features
        print('Loading feature sets')
        feat_files = []
        for t in traj_sel:
            feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

        # For now, assume if dataset not labeled as "closed", is open
        is_open = ['closed' not in d for d in feat_files]
        
        # Confirm both open & closed data present
        if len(np.unique(is_open)) < 2:
            update = 'Please select both an open and a closed dataset!'
            
            return [blank_fig()]
        
        # Load data
        global_state_df = clu.load_data(feat_files,is_open)
        global_state_df = clu.curate_feats(global_state_df,rbd_wind=rbd_wind,feat_incl=feat_sel,corr_thresh=corr_thresh)
        global_state_df.to_csv('./current_tmp_df.csv')
        print('Data loaded')
        feat_stats_fig_dict = clu.getfeatureStats(pd.DataFrame(global_state_df),feat_incl=feat_sel)
        print('Feature Histograms Plotted')
        return [feat_stats_fig_dict[feat_sel[0]]]
    else:
        if len(traj_sel) == 0:
            update = 'Please select at least 2 feature sets'
        elif len(traj_sel) == 1:
            update = 'Almost there! Please select 1 more feature set'
        elif len(traj_sel) > 1:
            update = "Ready to train the model! Feel free to adjust the model parameters, then hit the green button when you're ready!"
        else:
            update = "I'm confused..."
        #return [{}, {}, {}, update]
        return [blank_fig()]



# Train Model Callback
@app.callback(
              [
               Output('feat_imp','figure'),
               Output('performance_label','children'),
              #Output('feat_trace', 'figure'),
              # Output('df_feat','data'),
              # Output('update_window','children')
              ],
               [
               #Input('featureset_select','value'),
               #Input('feature_select','value'),
               #Input('rbd_wind','value'),
               #Input('corr_thresh','value'),
               #Input('global_state_df','data'),
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
        
        print('HERE-NEW')
        # Train Model
        global_state_df  = pd.read_csv('./current_tmp_df.csv')
        global_state_df.drop(['Unnamed: 0'],inplace=True,axis=1)
        #print(f'Preparing to train model-NEW , colu : {global_state_df.columns.to_list()}')
        tr_p, tr_r, ts_p, ts_r, df_feat = clu.train_sgd_model_new(pd.DataFrame(global_state_df.copy()))
        testResults = 'Test precision: ' + str(round(ts_p,3)) + ', Test recall: ' + str(round(ts_r,3))
        print(f'Training Completed : {testResults}')
        # Filter dataframe to top 10 features
        top_feats = df_feat[:10]['feats'].to_list() + ['Replicant','isopen','frame']    
        df = global_state_df[top_feats]
        df_feat = df_feat[:10]
        df.to_csv('./current_tmp_df.csv')
        df_feat.to_csv('./current_tmp_topfeats.csv')
        feat_imp_fig = clu.plot_feature_importances(df_feat)
        #return [feat_imp_fig, testResults, df_feat, update]
        return [feat_imp_fig,testResults]
    else:
             
        return [blank_fig(), {}]
    
    
@app.callback([Output('feat_trace','figure')],
              [Input('trace_go','n_clicks'),
               Input('feat_imp','clickData')],
              prevent_initial_call=True)
def plot_feature_traces(n_go,clickData):
    '''Plot trace of all replicants for top feature'''
    cmap = {'Monomer A':'royalblue','Monomer B':'indianred','Monomer C':'forestgreen','Core':'orange','RBD':'mediumpurple'}
    
    # Load data
    df = pd.read_csv('./current_tmp_df.csv')
    df_feat = pd.read_csv('./current_tmp_topfeats.csv')
    
    # Determine feature to be plotted
    ctx = callback_context
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    if buttonID == 'feat_imp':
        feat = clickData['points'][0]['x']
        feat_color = cmap[feat.split('at ')[-1]]
    else:
        feat = glycan_bionames.rename_feat(glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'feat'))
        feat_color = cmap[glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'chain')]
    
    # Plot
    feat_trace = clu.trace_single_feat(df,feat,feat_color)
    
    return [feat_trace]
    
# Show Features in 3D Callback
@app.callback([Output('spike1','figure'),
               Output('spike2','figure')],
              [Input('featureset_select','value'),
               Input('scatter_go','n_clicks')],
              prevent_initial_call=True)
def scatterplot_trajectories(traj_sel,n_go):
    '''Load trajectories and plot in 3D'''
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
        spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, df_feat,'Closed Spike','red')
        spike2_fig = mdu.viz_traj(traj_open,atom_id_open, df_feat,'Open Spike','blue')
        
        return [spike1_fig, spike2_fig]
    else:
        return [blank_fig(), blank_fig()]

# Do everything callback
"""
@app.callback(
              [Output('feat_imp','figure'),
               Output( 'performance_label','children'),
               Output('spike1','figure'),
               Output('spike2','figure'),
               Output('feat_trace','figure'),
              Output('df_feat','data'),
              Output('update_window','children')],
              [Input('featureset_select','value'),
               Input('feature_select','value'),
               Input('rbd_wind','value'),
               Input('corr_thresh','value'),
               Input('df_feat','data'),
               Input('feat_imp','clickData'),
               Input('feat_imp','figure'),
               Input('performance_label','children'),
               Input('spike1','figure'),
               Input('spike2','figure'),
               Input('feat_trace','figure'),
              Input('train_go','n_clicks'),
#               Input('go','n_clicks')
              ],
              prevent_initial_call = True
             )
def do_everything(traj_sel,feat_sel,rbd_wind,corr_thresh,
                  df_feat,
                  clickData,
                  feat_imp_fig,testResults,spike1_fig,spike2_fig,feat_trace,
                  n_train):
#                   ,n_go):
    update = 'Return'
    # Figure out why callback running
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        update = 'No trigger'
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
       
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]

    cmap = {'Monomer A':'royalblue','Monomer B':'indianred','Monomer C':'forestgreen','Core':'orange','RBD':'mediumpurple'}
        
    # ---------- Train Model Callback -------------
    if buttonID == 'train_go':
        # Confirm at least 2 feature sets selected
        if traj_sel is None or len(traj_sel) < 2:
            update = 'Please select at least 2 feature sets!'
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
        
        # Get list of csv files containing features
        print('Loading feature sets')
        feat_files = []
        for t in traj_sel:
            feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

        # For now, assume if dataset not labeled as "closed", is open
        is_open = ['closed' not in d for d in feat_files]
        print(is_open)
        
        # Confirm both open & closed data present
        if len(np.unique(is_open)) < 2:
            update = 'Please select both an open and a closed dataset!'
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
        
        # Load data
        df = clu.load_data(feat_files,is_open)
        print('Data loaded')
        
        # Train Model
        print('Preparing to train model')
        tr_p, tr_r, ts_p, ts_r, df_feat = clu.train_sgd_model(pd.DataFrame(df),
                                                              feat_incl=feat_sel,
                                                              rbd_wind=rbd_wind,
                                                              corr_thresh=corr_thresh)
        testResults = 'Test precision: ' + str(ts_p) + ', Test recall: ' + str(ts_r)
        
        # Filter dataframe to top 10 features
        top_feats = df_feat[:10]['feats'].to_list() + ['Replicant','isopen']        
        df = df[top_feats]
        df_feat = df_feat[:10]

        # Create figures
        feat_imp_fig = clu.plot_feature_importances(df_feat)
        feat_trace = clu.trace_single_feat(df,glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'feat'),cmap[glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'chain')])
#         feat_hist = clu.hist_single_feat(df,glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'feat'),cmap[glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'chain')])
        
        # Load trajectories - for now load first directories with and without 'closed' in the name
        print(traj_sel)
        print('Loading trajectories...')
        closed_dirs = traj_sel[['closed' in i.lower() for i in traj_sel].index(True)]
        open_dirs = traj_sel[['closed' in i.lower() for i in traj_sel].index(False)]
        traj_open = mdu.load_traj(open_dirs)
        traj_closed = mdu.load_traj(closed_dirs)
        
         # Parse trajectories
        print('Parsing trajectories...')
        atom_id_open = mdu.parse_traj(traj_open)
        atom_id_closed = mdu.parse_traj(traj_closed)

        # Create figures
        spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, pd.DataFrame(df_feat),'Closed Spike','red')
        spike2_fig = mdu.viz_traj(traj_open,atom_id_open, pd.DataFrame(df_feat),'Open Spike','blue')
        
        update = 'All trajectories loaded for visualization!'
        
        update = 'Training completed!'
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat.to_dict(),update]
    
    elif buttonID == 'feat_imp':
        # Get list of csv files containing features
        print('Loading feature sets')
        feat_files = []
        for t in traj_sel:
            feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

        # For now, assume if dataset not labeled as "closed", is open
        is_open = ['closed' not in d for d in feat_files]
        
        # Confirm both open & closed data present
        if len(np.unique(is_open)) < 2:
            update = 'Please select both an open and a closed dataset!'
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
        
        # Load data
        df = clu.load_data(feat_files,is_open)
        
        
        
        # Update trace of important feature
        feat = clickData['points'][0]['x']
        feat_trace = clu.trace_single_feat(df,feat,cmap[feat.split('_')[-1]])
        
        # Update histogram
#         feat_hist = clu.hist_single_feat(df,feat,cmap[feat.split('_')[-1]])
        
        
        update = 'Plotted ' + feat + ' data'
        
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
    
    
    
    # ----------- Do Everything Else Callback --------
#     elif buttonID == 'go':
#         # Confirm model has been trained
#         if len(df_feat) ==0:
#             update = 'Oops! You need to train the model first!'
#             return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
        
#         # Get list of csv files containing features
#         print('Loading feature sets')
#         feat_files = []
#         for t in traj_sel:
#             feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

#         # For now, assume if dataset not labeled as "closed", is open
#         is_open = ['closed' not in d for d in feat_files]
        
#         # Confirm both open & closed data present
#         if len(np.unique(is_open)) < 2:
#             update = 'Please select both an open and a closed dataset!'
#             return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
        
#         # Load data
#         df = clu.load_data(feat_files,is_open)
            
#         # Load trajectories - for now load first directories with and without 'closed' in the name
#         print('Loading trajectories...')
#         closed_dirs = traj_sel[['closed' in i for i in traj_sel].index(True)]
#         open_dirs = traj_sel[['closed' in i for i in traj_sel].index(False)]
#         traj_open = mdu.load_traj(open_dirs)
#         traj_closed = mdu.load_traj(closed_dirs)

#         # Parse trajectories
#         print('Parsing trajectories...')
#         atom_id_open = mdu.parse_traj(traj_open)
#         atom_id_closed = mdu.parse_traj(traj_closed)

#         # Create figures
#         spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, pd.DataFrame(df_feat),'Closed Spike','red')
#         spike2_fig = mdu.viz_traj(traj_open,atom_id_open, pd.DataFrame(df_feat),'Open Spike','blue')
        
#         update = 'All trajectories loaded for visualization!'
#         return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
    
    
    # ------------ All other callbacks -------------
    else:
        if len(traj_sel) == 0:
            update = 'Please select at least 2 feature sets'
        elif len(traj_sel) == 1:
            update = 'Almost there! Please select 1 more feature set'
        elif len(traj_sel) > 1:
            update = "Ready to train the model! Feel free to adjust the model parameters, then hit the green button when you're ready!"
        else:
            update = "I'm confused..."
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df_feat, update]
"""
# Run app
if __name__ == "__main__":
    app.run_server(debug=True)