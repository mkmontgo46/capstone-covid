# Import libraries
from dash import Dash, html, dcc, Input, Output, State
import plotly_express as px
import md_utils as mdu
import classification_utils as clu

# Initialize figure objects
fig1 = px.bar()
figOpen = px.scatter_3d()
figClosed = px.scatter_3d()

testResults = ''

# --------------- TEMP --------------- 
# Set input variables
fnames = ['/net/jam-amaro-shared/dse_project/Spike_Dataset/TRAJECTORIES_spike_open_prot_glyc_amarolab/results/FinalExtractedFeature_open.csv',
        '/net/jam-amaro-shared/dse_project/Spike_Dataset/TRAJECTORIES_spike_closed_prot_glyc_amarolab/results/FinalExtractedFeature_closed.csv']
is_open = [1, 0]
open_dir = '/net/jam-amaro-shared/dse_project/Spike_Dataset/TRAJECTORIES_spike_open_prot_glyc_amarolab/'
closed_dir = '/net/jam-amaro-shared/dse_project/Spike_Dataset/TRAJECTORIES_spike_closed_prot_glyc_amarolab/'
               


# Train Model
print('Preparing to train model')
tr_p, tr_r, ts_p, ts_r, df_feat = clu.train_sgd_model(fnames,is_open,feat_incl=['RBD__2__','RMSD','ROF'])
testResults = 'Test precision: ' + str(ts_p) + ', Test recall: ' + str(ts_r)

# Load trajectories
print('Loading trajectories...')
traj_open = mdu.load_traj(open_dir)
traj_closed = mdu.load_traj(closed_dir)

# Parse trajectories
print('Parsing trajectories...')
atom_id_open = mdu.parse_traj(traj_open)
atom_id_closed = mdu.parse_traj(traj_closed)

# Create figures
fig1 = clu.plot_feature_importances(df_feat)
figClosed = mdu.viz_traj(traj_closed,atom_id_closed, df_feat,'Closed Spike')
figOpen = mdu.viz_traj(traj_open,atom_id_open, df_feat,'Open Spike')

# --------------------------------------



# Initialize dashboard
app = Dash(__name__)

app.layout = html.Div(
    children=[
        # Title
        html.H1(children="Predicting Effects of SARS-CoV-2 Variant Mutations on Spike Protein Dynamics and Mechanism",),
        # Label performance of model
        html.P(
            children=testResults
        ),
        # Button to train model
#         html.Button('Train Model',id='train_model',
#                    children=train_model()),
#         # Button to load trajectories
#         html.Button('Load Trajectories',id='load_trajectories',
#                    children=load_trajectories()),
#         # BUtton to parse trajectories
#         html.Button('Parse Trajectories',id='parse_trajectories',
#                    children=parse_trajectories()),
#         # Button to generate plots
#         html.Button('Visualize',id='gen_viz',
#                    children=update_figures()),
        # Plot bar graph of feature importances
        html.Div(
            dcc.Graph(figure=fig1),
        ),
        # Plot open & closed spike with important features highlighted
        html.Div(children=[
            dcc.Graph(
            figure=figClosed,
            style={'display': 'inline-block'}
            ),
            dcc.Graph(
            figure=figOpen,
            style={'display': 'inline-block'}
            ),
        ]),
    ]
)



# Run app
if __name__ == "__main__":
    app.run_server(debug=False)