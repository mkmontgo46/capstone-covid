# Python ≥3.5 is required
import sys, re
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
import time

# Import custom utility functions
import glycan_bionames

# to make this notebook's output stable across runs
np.random.seed(42)

# # To plot pretty figures
# %matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)

# Define custom functions
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
def restrict_RBD_window(df,nm):
    '''Function to drop features of dataframe that correspond to glycans which are outside a given RBD neighborhood (in nm)'''
    #Get list of glycans
    glycans = list(np.unique([x.replace('RBD__2__','') for x in df.keys().to_list() if 'RBD__2__GLY' in x]))
    
    for g in glycans:
        if df['RBD__2__' + g].mean() > nm:
            for f in ['RBD__2__'+g,g+':ROF',g+':RMSD',g+'_x',g+'_y',g+'_z']:
                if f in df.keys().to_list():
                    df.drop([f],axis=1,inplace=True)    
    return df

def overlapping_hist(open_df,closed_df,feat):
    '''Plot overlapping histograms for a given feature of all datasets'''
    open_df[feat].hist(bins=50)
    closed_df[feat].hist(bins=50)
    mutant_df[feat].hist(bins=50)
    plt.legend(['Open','Closed','Mutant (open)'])
    plt.title(feat)
    if 'RBD__2__' in feat:
        plt.xlabel('nm')
        
def drop_feats(df,flag):
    '''Drops all features in df containing flag'''
    for f in df.keys().to_list():
        if flag in f:
            df.drop(f,axis=1,inplace=True)
    return df

def read_n_filter_dfs(fname,num_reps,RBD_wind,val_reps_open,val_reps_closed,label_val,dfs_train=None,dfs_val=None):
    '''Reads data and filters columns, then places in either train or validation dataframe list'''
    if dfs_train is None:
        dfs_train = []
    if dfs_val is None:
        dfs_val = []
        
    for i in range(1,num_reps+1):
        df = pd.read_csv(fname+'.csv').assign(label=label_val).iloc[:,1:]
        # Only use glycans within certain range of the RBD
        df = restrict_RBD_window(df,RBD_wind)
        # Drop _x, _y, and _z features
        df = drop_feats(df,'_x')
        df = drop_feats(df,'_y')
        #df = drop_feats(df,'RBD__2__')
        df = drop_feats(df,'_z')
        
        # Withold some replicants for use in a separate validation set
        if (label_val==1) & (i in val_reps_open):
            dfs_val.append(df)
        elif (label_val==0) & (i in val_reps_closed):
            dfs_val.append(df)
        else:
            dfs_train.append(df)
            
    return dfs_train, dfs_val

def remove_corr_feats(full_df,corr_thresh= 0.65):
    '''Remove highly correlated features'''
    corr_matrix = full_df.corr()
    final_features = corr_matrix['RBD_CA0:RMSD'][(corr_matrix['RBD_CA0:RMSD'] < corr_thresh) & (corr_matrix['RBD_CA0:RMSD'] > -corr_thresh)].reset_index().loc[:,'index'].to_list()
    if 'label' not in final_features:
        final_features.append('label')
    clf_df = full_df.loc[:,final_features]
    return clf_df

def gen_pipeline():
    '''Create pipeline for normalization of input data'''
    num_pipeline = Pipeline([
           ('std_scaler', StandardScaler()),
        ])
    return num_pipeline

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

    dfs = []
    for f in range(len(fnames)):
        df = pd.read_csv(fnames[f]).assign(label = is_open[f]).iloc[:,1:]
        df['Replicant'] = fnames[f].split('TRAJECTORIES_')[1]
        df['isopen'] = is_open[f]
        dfs.append(df)
    return pd.concat(dfs,join='inner')

    
def train_sgd_model(df, rbd_wind=8, feat_incl=['_x','_y','_z','RBD__2__','ROF','RMSD'], corr_thresh=0.5):
    '''Train SGD classifier on input data using input features'''
    # df = dataframe 
    # rbd_wind = distance in nm. Only glycans with COM < rbd_wind away from RBD will be included in analysis
    # feat_incl = list of features to include. Options are '_x','_y','_z','RBD__2__','ROF','RMSD'
    # corr_thresh = maximum threshold for correlations between features. Features with higher correlations will be dropped
    
    # Set some variables
    tt_split = 0.3
    rand_seed = 42
    
    # Restrict RBD window 
    df = restrict_RBD_window(df,rbd_wind)
    
    # Drop unwanted features
    all_feats = ['_x','_y','_z','RBD__2__','ROF','RMSD']
    for f in all_feats:
        if f not in feat_incl:
            df = drop_feats(df,f)
            
    # Drop non-feature columns
    non_features = ['frame','frame_num','isopen','Replicant']
    for f in non_features:
        if f in df.keys():
            df = df.drop([f],axis=1)
    
    # Remove highly correlated features
    df = remove_corr_feats(df,corr_thresh)
    
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
    x_vals = [glycan_bionames.get_elem(i,'feat') for i in df_feats['feats'].to_list()]
    y_vals = df_feats['importance'].to_list() 
    col_vals = [glycan_bionames.get_elem(i,'chain') for i in df_feats['feats'].to_list()]
    cmap = {'Monomer A':'royalblue','Monomer B':'indianred','Monomer C':'forestgreen','Core':'orange','RBD_CA0:ROF':'mediumpurple'}

    fig1 = px.bar(x=x_vals,y=y_vals,color=col_vals,title='Important Features',color_discrete_map=cmap, labels={'x':'Feature','y':'Importance','color':'Substructure'}).update_xaxes(categoryorder='total ascending')
    fig1.update_layout(template='simple_white')
    return fig1

def trace_single_feat(df,f,title_clr):
    '''Draws line plot for feature f for all replicants in dataframe df'''
    # Define colors: open = blue & closed = red
    df_r = df[['Replicant','isopen']].drop_duplicates()
    cmap = {}; colors = ['red','blue']
    for i in range(len(df_r)):
        cmap[df_r.iloc[i]['Replicant']] = colors[df_r.iloc[i]['isopen']]
    
    # Convert feature names to bionames 
    rename_cols = {'Replicant':'Replicant','isopen':'isopen'}
    for c in df.keys().to_list():
        rename_cols[c] = glycan_bionames.get_elem(c,'feat')
    df = df.rename(columns=rename_cols)
        
    # Create new dataframe with each replicant having different trace of feature f
    df_f = pd.DataFrame()
    for r in df['Replicant'].unique():
       df_f[r] = df.loc[df['Replicant']==r][f]

    # Plot
    fig = px.line(df_f,title=f + ' Over a Full Trajectory',color_discrete_map=cmap,
                 labels={'x':'Frame Number','y':f+' Value','color':'Replicant'})
    fig.update_layout(template='simple_white',
                     title={'font':{'color':title_clr}})
    return fig