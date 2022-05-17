# Import libraries
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly_express as px
import md_utils as mdu
import classification_utils as clu
import glycan_bionames
import os, glob
import pandas as pd
import numpy as np


# --------------- Initialize Variables -----------------------

# Locations to read data from
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

# -------------- Create dashboard --------------
app = Dash(__name__)

app.layout = html.Div(
    children=[
        # Title
        html.H1(children="Predicting Effects of SARS-CoV-2 Variant Mutations on Spike Protein Dynamics and Mechanism",),
        
        html.P(id='update_window',
               children='Welcome! Please select feature sets to begin!',
               style={'font-weight':'bold',
                      'background-color':'#ebecf0',
                      'width':'33%'}),
        
        html.Div(children=[
            # Drop-down list for user to select feature sets
            html.Label('Select feature sets to analyze:'),
            html.Br(),
            dcc.Dropdown(id='featureset_select',
                         options= traj_dirs,
                         multi=True,
                         placeholder='Select at least 2 feature sets')
        ]),
        html.Br(),
        
        html.Div(children=[
            html.Div(children=[
                # Drop-down list for features to include
                html.Label('Select which features to use:'),
                html.Br(),
                dcc.Dropdown(id='feature_select',
                             options=feat_opts,
                             multi=True,
                             value=feat_vals,
                             placeholder='Select which features to use'),
            ]),
            html.Div(children=[
                # Input option for max correlation between two features
                html.Label('Exclude glycans further away from the RBD than:'),
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
                html.Label('Exclude features with a correlation higher than:'),
                html.Br(),
                dcc.Input(id='corr_thresh',
                          type='number',
                          min=0,
                          max=1,
                          step=0.01,
                          value=0.5),
            ]),
            html.Div(children=[
                # Button to train model
                html.Button('Train Model',
                            id='train_go',
                            n_clicks=0,
                            style={'color':'green',
                                   'background-color':'#d3f8d3',
                                   'height':'75px',
                                   'width':'100px'})
            ]),
        ],
                style={'display':'flex',
                       'justify-content':'space-around',
                       'align-items':'flex-start',
                       'column-gap':'50px'}),
        
        html.Div(children=[
            # Label performance of model
            html.P(id='performance_label',
                   children=''),
            # Plot bar graph of feature importances
            dcc.Graph(id='feat_imp',figure={}),
        ]),
        
        # Plot top feature over trajectory
        html.Div(children=[
            dcc.Graph(id='feat_trace',
                      figure={})
        ]),
        
        # Button to do everything else
        html.Button('Map Important Features',id='go',n_clicks = 0,
                   style = {'height':'35px'}),
        # Plot open & closed spike with important features highlighted
        html.Div(children=[
            dcc.Graph(id='spike1',
            figure={},
            style={'display': 'inline-block'}
            ),
            dcc.Graph(id='spike2',
            figure={},
            style={'display': 'inline-block'}
            ),
        ]),
        
        

        # Store variables
        dcc.Store(id='df'),
        dcc.Store(id='df_feat')
    ]
)

# -------------- Callbacks -----------------

# Do everything else callback
@app.callback(
              [Output('feat_imp','figure'),
               Output( 'performance_label','children'),
               Output('spike1','figure'),
               Output('spike2','figure'),
               Output('feat_trace','figure'),
               Output('df','data'),
              Output('df_feat','data'),
              Output('update_window','children')],
              [Input('featureset_select','value'),
               Input('feature_select','value'),
               Input('rbd_wind','value'),
               Input('corr_thresh','value'),
               Input('df','data'),
               Input('df_feat','data'),
               Input('feat_imp','clickData'),
               Input('feat_imp','figure'),
               Input('performance_label','children'),
               Input('spike1','figure'),
               Input('spike2','figure'),
               Input('feat_trace','figure'),
              Input('train_go','n_clicks'),
              Input('go','n_clicks')],
              prevent_initial_call = True
             )
def do_everything(traj_sel,feat_sel,rbd_wind,corr_thresh,
                  df, df_feat,
                  clickData,
                  feat_imp_fig,testResults,spike1_fig,spike2_fig,feat_trace, 
                  n_train,n_go):
    update = 'Return'
    # Figure out why callback running
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        update = 'No trigger'
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
       
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
        
    # ---------- Train Model Callback -------------
    if buttonID == 'train_go':
        # Confirm at least 2 feature sets selected
        if traj_sel is None or len(traj_sel) < 2:
            update = 'Please select at least 2 feature sets!'
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
        
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
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
        
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

        # Create figure
        feat_imp_fig = clu.plot_feature_importances(df_feat)
        feat_trace = clu.trace_single_feat(df,glycan_bionames.get_elem(df_feat.iloc[0]['feats'],'feat'))
        
        update = 'Training completed!'
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df.to_dict(), df_feat.to_dict(),update]
    
    elif buttonID == 'feat_imp':
        # Update trace of important feature
        feat = clickData['points'][0]['x']
        feat_trace = clu.trace_single_feat(pd.DataFrame(df),feat)
        
        
        update = 'Plotted ' + feat + ' data'
        
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
    
    
    
    # ----------- Do Everything Else Callback --------
    elif buttonID == 'go':
        # Confirm model has been trained
        if len(df_feat) ==0:
            update = 'Oops! You need to train the model first!'
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
            
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
        spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, pd.DataFrame(df_feat),'Closed Spike')
        spike2_fig = mdu.viz_traj(traj_open,atom_id_open, pd.DataFrame(df_feat),'Open Spike')

        
        update = 'All trajectories loaded for visualization!'
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]
    
    
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
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, feat_trace, df, df_feat, update]

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)