# Import libraries
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly_express as px
import md_utils as mdu
import classification_utils as clu
import os, glob
import pandas as pd


# --------------------------------------

# Set locations to read data from
data_dir = '/net/jam-amaro-shared/dse_project/Spike_Dataset/'
traj_dirs = glob.glob(os.path.join(data_dir,'*TRAJECTOR*'))
traj_opts = []
for i in range(len(traj_dirs)):
    traj_opts.append({'label': traj_dirs[i], 'value': traj_dirs[i]})

# Initialize dashboard
app = Dash(__name__)

app.layout = html.Div(
    children=[
        # Title
        html.H1(children="Predicting Effects of SARS-CoV-2 Variant Mutations on Spike Protein Dynamics and Mechanism",),
        html.Div(children=[
            # Drop-down list for user to select feature sets
            dcc.Dropdown(id='featureset_select',
                         options= traj_dirs,
                         multi=True,
                         placeholder='Select 2 feature sets'),
            # Button to train model
            html.Button('Train Model',
                        id='train_go',
                        n_clicks=0)
        ]),
        # Label performance of model
        html.P(id='performance_label',
            children=''
        ),
        # Button to do everything else
        html.Button('Do Everything Else',id='go',n_clicks = 0),
        # Plot bar graph of feature importances
        html.Div(
            dcc.Graph(id='feat_imp',figure={}),
        ),
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
        dcc.Store(id='df_feat')
    ]
)


# Do everything else callback
@app.callback(
              [Output('feat_imp','figure'),
               Output( 'performance_label','children'),
               Output('spike1','figure'),
               Output('spike2','figure'),
              Output('df_feat','data')],
              [Input('featureset_select','value'),
               Input('df_feat','data'),
               Input('feat_imp','figure'),
               Input('performance_label','children'),
               Input('spike1','figure'),
               Input('spike2','figure'),
              Input('train_go','n_clicks'),
              Input('go','n_clicks')],
              prevent_initial_call = True
             )
def do_everything(traj_sel,df_feat,feat_imp_fig,testResults,spike1_fig,spike2_fig,n_train,n_go):
    # Figure out why callback running
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        print('Doing nothing')
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, df_feat]
       
    buttonID = ctx.triggered[0]['prop_id'].split('.')[0]
    
        
    if buttonID == 'train_go':
        # Get list of csv files containing features
        print('Loading feature sets')
        feat_files = []
        for t in traj_sel:
            feat_files.extend(glob.glob(os.path.join(t,'results','*FinalExtractedFeature*.csv')))

        # For now, assume if dataset not labeled as "closed", is open
        is_open = ['closed' not in d for d in feat_files]
        
        df = clu.load_data(feat_files,is_open)
        print('Data loaded')
        
        # Train Model
        print('Preparing to train model')
        tr_p, tr_r, ts_p, ts_r, df_feat = clu.train_sgd_model(pd.DataFrame(df),feat_incl=['RBD__2__','RMSD','ROF'])
        testResults = 'Test precision: ' + str(ts_p) + ', Test recall: ' + str(ts_r)

        # Create figure
        feat_imp_fig = clu.plot_feature_importances(df_feat)
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, df_feat.to_dict()]
    elif buttonID == 'go':
        if len(df_feat) ==0:
            print('Train model first')
            return [feat_imp_fig, testResults, spike1_fig, spike2_fig, df_feat]
            
        # Load trajectories - for now load first directories with and without 'closed' in the name
        print('Loading trajectories...')
        closed_dirs = traj_sel[['closed' in i for i in traj_sel].index(True)]
        print(closed_dirs)
        open_dirs = traj_sel[['closed' in i for i in traj_sel].index(False)]
        print(open_dirs)
        traj_open = mdu.load_traj(open_dirs)
        traj_closed = mdu.load_traj(closed_dirs)

        # Parse trajectories
        print('Parsing trajectories...')
        atom_id_open = mdu.parse_traj(traj_open)
        atom_id_closed = mdu.parse_traj(traj_closed)

        # Create figures
        print(df_feat)
        spike1_fig = mdu.viz_traj(traj_closed,atom_id_closed, pd.DataFrame(df_feat),'Closed Spike')
        spike2_fig = mdu.viz_traj(traj_open,atom_id_open, df_feat,'Open Spike')
        
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, df_feat]
    else:
        print('Callback trigger ' + buttonID + ' is not recognized. Will not run')
        return [feat_imp_fig, testResults, spike1_fig, spike2_fig, df_feat]

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)