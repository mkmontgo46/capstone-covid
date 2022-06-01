# Python ≥3.5 is required
import sys, re
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from tkinter.tix import COLUMN
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

# Common imports
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import time

# Import custom utility functions
import glycan_bionames

# ---------- Define custom functions ---------- 
def get_label_colors():
    '''Define colors to use for open and closed figures'''
    closed_clr = px.colors.qualitative.Set1[0]
    open_clr = px.colors.qualitative.Set1[1]
    return closed_clr, open_clr
closed_clr, open_clr = get_label_colors()

def get_substruct_cmap():
    '''Create colormap for encoding various substructures of features'''
    substructs = ['Monomer A','Monomer B','Monomer C','Core','RBD']
    indices = [3,4,5,6,7]
    cmap={}
    for s in range(len(substructs)):
        cmap[substructs[s]] = px.colors.qualitative.G10[indices[s]]
    return cmap
    
def restrict_RBD_window(df,nm):
    '''Function to drop features of dataframe that correspond to glycans which are outside a given RBD neighborhood (in nm)'''
    #Get list of glycans
    glycans = list(np.unique([x.replace('RBD__2__','') for x in df.keys().to_list() if 'RBD__2__GLY' in x]))
    
    flist = []
    for g in glycans:
        if df['RBD__2__' + g].mean() > nm:
            for f in ['RBD__2__'+g,g+':ROF',g+':RMSD',g+'_x',g+'_y',g+'_z']:
                if f in df.keys().to_list():
                    flist.append(f)
    df.drop(flist,axis=1,inplace=True)
    return df
        
def drop_feats(df,flag):
    '''Drops all features in df containing flag'''
    flist = []
    for f in df.keys().to_list():
        if flag in f:
            flist.append(f)
    df.drop(flist,axis=1,inplace=True)
    return df

def remove_corr_feats(full_df,corr_thresh= 0.65):
    '''Remove highly correlated features'''
    corr_matrix = full_df.corr()
    final_features = corr_matrix['RBD_CA0:RMSD'][(corr_matrix['RBD_CA0:RMSD'] < corr_thresh) & (corr_matrix['RBD_CA0:RMSD'] > -corr_thresh)].reset_index().loc[:,'index'].to_list()
    
    for keep in ['label','Replicant','frame']:
        if keep not in final_features:
            final_features.append(keep)
    clf_df = full_df.loc[:,final_features]
    return clf_df

def gen_pipeline():
    '''Create pipeline for normalization of input data'''
    num_pipeline = Pipeline([
           ('std_scaler', StandardScaler()),
        ])
    return num_pipeline

def curate_feats(df,rbd_wind=4,feat_incl=['_x','_y','_z','RBD__2__','ROF','RMSD'], corr_thresh=0.5):
    '''Restrict rbd window, drop unwanted features, and threshold correlation bw features'''
    # Limit RBD_window
    df = restrict_RBD_window(df,rbd_wind)
    all_feats = ['_x','_y','_z','RBD__2__','ROF','RMSD']
    
    # Remove highly correlated features
    df = remove_corr_feats(df,corr_thresh)
    
    # Drop features user selected not to include
    for f in all_feats:
        if f not in feat_incl:
            df=drop_feats(df,f)
            
    return df

def prep_ML_data(clf_df,ts,rs,labelnames):
    '''Prepare data for use in training machine learning algorithm'''
    # Split training and testing data
    train_set, test_set = train_test_split(clf_df,test_size=ts, random_state=rs,stratify=labelnames)
    print(f'Train set : {train_set.shape}, Test set : {test_set.shape}')

    # Split data and labels
    train_X = train_set.drop("label", axis=1) # drop labels for training set
    train_labels = train_set["label"].copy()
    test_X = test_set.drop("label", axis=1) # drop labels for training set
    test_labels = test_set["label"].copy()
    
    # Normalize data
    num_pipeline = gen_pipeline()
    train_X_prepared = num_pipeline.fit_transform(train_X)
    test_X_prepared = num_pipeline.transform(test_X)
    
    return train_X, test_X, train_X_prepared, test_X_prepared, train_labels, test_labels
   
def load_data(fnames, is_open):
    '''Load and concatenate all datasets'''
    # fnames = list of files corresponding to featuresets to use in training. Should include full path
    # is_open = list of labels for corresponding fnames. 1 is open & 0 is closed
    openlabels = ['Closed','Open']
    dfs = []
    for f in range(len(fnames)):
        # Read data
        df = pd.read_csv(fnames[f]).assign(label = is_open[f]).iloc[:,1:]
        # Assign tracking variables
        df['Replicant'] = '/'.join(fnames[f].split('/')[-3:])
        df['isopen'] = is_open[f]
        # Add to list
        dfs.append(df)
    return pd.concat(dfs,join='inner')

def getfeatureStats(df, feat_incl=['RBD__2__','ROF','RMSD','_x','_y','_z'], feat_type='RBD__2__'):
    '''Plot histograms of potential features'''
    # Create mapping of feature names to columns
    feat_descMap = {'RBD__2__': 'RBD Distances',
                    'ROF' : 'Radius of Gyration',
                    'RMSD' : 'RMSD',
                    '_x' : 'x location',
                    '_y' : 'y location',
                    '_z' : 'z location',
                    }
    featureMap = {}
    for f in feat_incl:
        featureMap[f] = [col for col in df.columns.to_list() if f in col]

    # Select colors to use for plotting. Default is 58 options
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS +  plotly.colors.qualitative.Dark24 + plotly.colors.qualitative.Light24
    
    k = feat_type; xmins = []; xmaxs=[]
    fig = make_subplots(1,2, subplot_titles= ['Closed', 'Open'],shared_yaxes=True)
    for c in featureMap[k]:
        for l in df.label.unique():
            cur_mask = df.label == l
            if l :
                shl,i = True,1
            else :
                shl,i = False,0
            # Add histogram to figure
            fig.add_trace(go.Histogram(x=df[cur_mask][c],name=c,legendgroup=c,showlegend=shl,marker = dict(color = cols[featureMap[k].index(c)])) ,1, i+1)
                
        # Sync x axis
        xmins.append(min(df[c]))
        xmaxs.append(max(df[c]))
    
    # Format graph
    fig.update_xaxes(title_text=feat_descMap[k],
                     range=[min(xmins), max(xmaxs)],
                     row=1,col=1)
    fig.update_xaxes(title_text=feat_descMap[k],
                     range=[min(xmins), max(xmaxs)],
                     row=1,col=2)
    fig.update_layout(
        template='plotly_white',
        title_text = f'Features Engineered from {feat_descMap[k]}', # title of plot
        yaxis_title_text = 'Count' #yaxis label
    )

    return fig   
    

def train_sgd_model(df):
    '''Train SGD classifier on input data using input features'''
    # df = dataframe 
    # rbd_wind = distance in nm. Only glycans with COM < rbd_wind away from RBD will be included in analysis
    # feat_incl = list of features to include. Options are '_x','_y','_z','RBD__2__','ROF','RMSD'
    # corr_thresh = maximum threshold for correlations between features. Features with higher correlations will be dropped
    
    # Set some variables
    tt_split = 0.3
    rand_seed = 42
    
    # Drop non-feature columns
    non_features = ['frame','frame_num','isopen','Replicant']
    for f in non_features:
        if f in df.keys():
            df = df.drop([f],axis=1)
        
    # Perform train/test split & normalize
    train_X, test_X, train_X_prepared, test_X_prepared, train_labels, test_labels = prep_ML_data(df,tt_split,rand_seed,df.label)
    
    # Train model
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(train_X_prepared,train_labels)
    
    # Get performance on train & test data
    y_train_pred = sgd_clf.predict(train_X_prepared)
    y_test_pred = sgd_clf.predict(test_X_prepared)
    train_prec = precision_score(train_labels, y_train_pred)
    train_recall = recall_score(train_labels, y_train_pred)
    test_prec = precision_score(test_labels, y_test_pred)
    test_recall = recall_score(test_labels, y_test_pred)
    
    # Get feature importances
    dfFeats = pd.DataFrame({'feats':train_X.columns.to_list(),'importance':np.abs(sgd_clf.coef_[0])}).sort_values('importance', ascending=False)

    return train_prec, train_recall, test_prec, test_recall, dfFeats


def plot_feature_importances(df_feats):
    '''Plot bar chart of feature importances'''
    x_vals = [glycan_bionames.rename_feat(glycan_bionames.get_elem(i,'feat')) for i in df_feats['feats'].to_list()]
    y_vals = df_feats['importance'].to_list() 
    col_vals = [glycan_bionames.get_elem(i,'chain') for i in df_feats['feats'].to_list()]
    cmap = get_substruct_cmap()
        
    fig1 = px.bar(x=x_vals,y=y_vals,color=col_vals,title='Feature Importance',color_discrete_map=cmap, labels={'x':'Feature','y':'Importance','color':'Chain'}).update_xaxes(categoryorder='total descending')
    fig1.update_layout(template='simple_white')
    return fig1

def trace_single_feat(df,f,title_clr,cmap):
    '''Draws line plot for feature f for all replicants in dataframe df'''
    # Convert feature names to bionames 
    rename_cols = {'Replicant':'Replicant','isopen':'isopen'}
    for c in df.keys().to_list():
        rename_cols[c] = glycan_bionames.rename_feat(glycan_bionames.get_elem(c,'feat'))
    df = df.rename(columns=rename_cols)
        
    # Create new dataframe with each replicant having different trace of feature f
    df['frame'] = df.apply(lambda row: int(row['frame'].replace('frame_','')), axis=1)
    df= df.set_index('frame')
    df1 = df.loc[df['isopen']==0]
    df2 = df.loc[df['isopen']==1]
    df_f1 = pd.DataFrame()
    df_f2 = pd.DataFrame()
    for r in df1['Replicant'].unique():
       df_f1[r] = df1.loc[df1['Replicant']==r][f]
    for r in df2['Replicant'].unique():
       df_f2[r] = df2.loc[df2['Replicant']==r][f]
    
    # Plot
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])
    for c in df_f1.columns:
        fig.append_trace(go.Scatter(x=df_f1.index,y=df_f1[c],mode='lines', name='Closed', legendgroup='group1',showlegend=False,marker_color=closed_clr, opacity=0.67),1,1)
    for c in df_f2.columns:
        fig.append_trace(go.Scatter(x=df_f2.index,y=df_f2[c], mode='lines',name='Open',legendgroup='group2',showlegend=False,marker_color=open_clr, opacity=0.67),1,1)
    
    
    # ------------ Combine w/ Histogram ----------------
    # Get hist traces
    t1,t2 = hist_single_feat(df,f,title_clr,cmap)
    # Add to figure
    fig.append_trace(t1,1,2)
    fig.append_trace(t2,1,2)
    # Format full figure
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        bargap=0,
        barmode='stack',
        template='plotly_white',
        title={'font':{'color':title_clr},
                            'text':'<b>' + glycan_bionames.rename_feat(f) + ' Over a Full Trajectory<b>'}
    )
    # Format axes
    fig.update_yaxes(showticklabels=False,row=1,col=2)
    fig.update_yaxes(title_text=glycan_bionames.rename_feat(f),row=1,col=1)
    fig.update_xaxes(title_text='Frame',row=1,col=1)
    fig.update_xaxes(title_text='Frequency',row=1,col=2)
    
    return fig

def hist_single_feat(df,f, title_clr,cmap):
    '''Draw histogram for feature f'''
    # Convert feature names to bionames 
    rename_cols = {'Replicant':'Replicant','isopen':'isopen'}
    for c in df.keys().to_list():
        rename_cols[c] = glycan_bionames.get_elem(c,'feat')
    df = df.rename(columns=rename_cols)
    
    # Relabel isopen
    labels = ['Closed','Open']
    df['state'] = df.apply(lambda row: labels[int(row['isopen'])], axis=1)

    # Create histogram objects
    fig1 = go.Histogram(y=df[df['state']=='Closed'][f],name='Closed', legendgroup='group1',showlegend=True,marker_color=closed_clr)
    fig2 = go.Histogram(y=df[df['state']=='Open'][f],name='Open', legendgroup='group2',showlegend=True, marker_color=open_clr)
    return fig1,fig2