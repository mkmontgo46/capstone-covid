#!/Users/harshitaarora/opt/anaconda3/bin/python

import os,sys,tarfile,re,glob
import time 
import numpy as np
import pandas as pd
import mdtraj as md
from biopandas.pdb import PandasPdb
import argparse
import psutil,multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import logging
from queue import Queue
from threading import Thread
from time import time


import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


def extract_glycan_residues_4m_pdb(dcdObj):
    dcdObj[0].save_pdb('.tmp.pdb')
    pdb_df = PandasPdb().read_pdb('.tmp.pdb')
       
    pdb_atom_df = pdb_df.df['ATOM']
    glycan_mask =  pdb_atom_df.segment_id.apply(lambda x : True if re.match('G\d+', x) else False)
    glycan_residues = pdb_atom_df[glycan_mask].residue_name.unique()
    if os.path.exists('.tmp.pdb'):
        os.remove('.tmp.pdb')
    del pdb_df
    return glycan_residues    



def get_atom_ids_for_feature(dcd_traj,feature='protein'):
    try:
        result = (i for i in dcd_traj.top.select(feature))
    except :
        print(f'[ERROR] {feature} not recognized for atom filtering')
        result = []
    else :
        #print(f'[INFO] # of atoms : {len(list(result))} filtered for {feature}')
        return list(result)

def build_atom_lup_4_common_features(dcd_traj,flist = ['protein', 'backbone','sidechain']):
    return {f: get_atom_ids_for_feature(dcd_traj,f) for f in flist}


def metric_4m_mDtraj(dcdObj):
    return {'Rofguration' : md.compute_rg(dcdObj),
            'density' : md.density(dcdObj),
            'compute_center_of_mass' : md.compute_center_of_mass(dcdObj)
            }



def rmsd_by_mdTraj(f1,f2):
    return md.rmsd(f1,f2)[0]
def gen_frame_tuples(dcdObj):
    return ((i+1,i) for i in range(dcdObj.n_frames -1 ))



def save_dcdMetric(addon_metrics,fileOut = f'./dcd_ExtractedMetrics.csv'):    
    dcd_metricOut_df = pd.DataFrame()
    for k in addon_metrics.keys():
        if (( type(addon_metrics[k]) == list) or ( len(addon_metrics[k].shape) == 1)):
            dcd_metricOut_df[k] = addon_metrics[k]
        else :
            for y in range(addon_metrics[k].shape[1]):

                dcd_metricOut_df[k + '_' + str(y)] = addon_metrics[k][:,y]
    dcd_metric_Out_df = (dcd_metricOut_df
     .assign(frame = lambda df : [ f'frame_{i}' for i in range(df.shape[0]) ])
     .rename(columns = {'Rofguration' : 'rofgyration' , 
                        'compute_center_of_mass_0' : 'COM_x',
                        'compute_center_of_mass_1' : 'COM_y',
                        'compute_center_of_mass_2' : 'COM_z'
                       }
            )
     .loc[:, ['frame','rofgyration','density',
     'rmsd', 
     'COM_x',
     'COM_y',
     'COM_z',
     ]]

    )
    if not os.path.exists(os.path.dirname(fileOut)):
        os.mkdir(os.path.dirname(fileOut))
        
    dcd_metric_Out_df.to_csv(fileOut)
    #print(f'[INFO] metric dumped ==> {fileOut}')
    return dcd_metric_Out_df



def extract_features_4m_filteredTraj(trajObj,trajName,outDir):
    addon_metrics = metric_4m_mDtraj(trajObj)
    addon_metrics['rmsd'] = [md.rmsd(trajObj[i+1],trajObj[0])[0] for i in range(trajObj.n_frames - 1)]
    addon_metrics['rmsd'].append(md.rmsd(trajObj[-1],trajObj[0])[0])
    df = save_dcdMetric(addon_metrics,fileOut=f'{outDir}/{trajName}/featureMetric.csv')
    return df.assign(tracjectory_KEY = trajName)



def extract_featuresPerChain_4m_filteredTraj(trajObj,trajName, outDir, chainIds = []):
    
    df = pd.DataFrame()
    if len(chainIds) == 0 :
        chainIds = range(trajObj.n_chains)
        
    
    chain_LUP = {f'chainID_{c}' : get_atom_ids_for_feature(trajObj,f'chainid == {c}') for c in range(trajObj.n_chains) if c in chainIds}
    print(f'[INFO] {trajName} {len(list(chain_LUP.keys()))} chains considered for feature extraction')
    for k in chain_LUP.keys():
        
        cur_time = time()
        print(f'[INFO] deriving Chain {k} tracjectory')
        chainObj = derive_trajectory(trajObj,frames=list(range(trajObj.n_frames)),atom_key= f'{k}',LUP = chain_LUP) 
        print(f'[INFO] Chain {k} tracjectory derivation completed in {round(time() - cur_time,2) } seconds')
        cur_time = time()
        addon_metrics = metric_4m_mDtraj(chainObj)
        addon_metrics['rmsd'] = [md.rmsd(chainObj[i+1],chainObj[0])[0] for i in range(chainObj.n_frames - 1)]
        addon_metrics['rmsd'].append(md.rmsd(chainObj[-1],chainObj[0])[0])
        del chainObj
        df = (df.append(
                save_dcdMetric(addon_metrics,fileOut=f'{outDir}/{trajName}/featureMetric__{k}.csv')
                .assign(chain_ID = k)
                .assign(tracjectory_KEY = trajName)
                )
             )
        print(f'[INFO] Feature extraction for chain {k} completed in {round(time() - cur_time,2) } seconds')
        
    return df



def derive_trajectory(traj_Full,frames=[0], atom_key = 'backbone', LUP = {}):
    return traj_Full[frames].atom_slice(LUP[atom_key])

def launchFeatureExract(dcdObj,gly_chains,outDir,LUP_dict={}):
    
    
    df_Traj_chains = pd.DataFrame()    
    start_0 = time()
    for k in LUP_dict.keys():
        if k == 'protein' or k == 'sidechain':
            continue
        cur_time = time()
        print(f'[INFO] deriving Tracjectory for {k}')
        curTraj = derive_trajectory(dcdObj,frames=list(range(dcdObj.n_frames)),atom_key=k ,LUP = LUP_dict)
        print(f'[INFO] Tracjectory derivation completed in {round(time() - cur_time,2) } seconds')
       
        if k == 'GLY' :
            print(f'[INFO] Starting Feature extraction for {gly_chains} of feature {k}')
            extract_featuresPerChain_4m_filteredTraj(curTraj,k, outDir,chainIds = gly_chains)

        else : 
            print(f'[INFO] Starting Feature extraction for {k}')
            extract_features_4m_filteredTraj(curTraj,k,outDir)

        print(f'[INFO] Feature extraction for  {k} completed in {round(time() - cur_time,2) } seconds')
        del curTraj

    print(f'[INFO] Time elapsed for Feature extraction {round(time() - start_0,2) } seconds')
    


def get_xyz_perFrame(traj,atom_ids):
    return pd.DataFrame(columns=['x','y','z'], data=traj.xyz[0,atom_ids])


def gen_xyz_Table_4_LUP(LUP = {}, keyNames =['sidechain','RBD_CA', 'CH_CA', 'GLY','backbone'] ):
    frame_0_coord_df = pd.DataFrame(columns=['type','typeID','x','y','z'])
    i = 0 
    for k in LUP.keys():
        if k in keyNames:
            frame_0_coord_df = (frame_0_coord_df
            .append(get_xyz_perFrame(dcd_traj,LUP[k]).assign(type = k).assign(typeID = i))
                               )
            i += 1
    return frame_0_coord_df

def extract_distance_metric(df1,df2):
   
    return (df1
     .merge(df2, left_on=['frame'], right_on=['frame'], how = 'inner', suffixes = ['_1','_2'])
     .assign(metric = lambda dfx : np.sqrt((np.square(dfx.COM_x_1 - dfx.COM_x_2) + np.square(dfx.COM_y_1 - dfx.COM_y_2) + np.square(dfx.COM_z_1 - dfx.COM_z_2) ).astype(float)))
     .metric.to_list()
    )

################## H-bond extraction code updates ############################
def report_inter_hbond(traj,hbonds,inter_structs = ['RBD_CA','GLY']):
    type_1_hbond = 0
    type_2_hbond = 0
    type_3_hbond = 0
    for hbond in hbonds :
        #print(set(traj[inter_structs[0]]).intersection([hbond[0]]))
        if len(set(traj[inter_structs[0]]).intersection([hbond[0]])) and \
        len((set(traj[inter_structs[1]]).intersection([hbond[1]]) or set(traj[inter_structs[1]]).intersection([hbond[2]]))):
            type_1_hbond += 1
        elif len(set(traj[inter_structs[1]]).intersection(list(hbonds[0]))) and \
        len((set(traj[inter_structs[0]]).intersection([hbond[1]]) or set(traj[inter_structs[0]]).intersection([hbond[2]]))):
            type_2_hbond += 1
        elif len(set(traj[inter_structs[1]]).intersection(list(hbonds[0]))) and \
        len((set(traj[inter_structs[1]]).intersection([hbond[1]]) or set(traj[inter_structs[1]]).intersection([hbond[2]]))):
            type_3_hbond += 1
            
    return [type_1_hbond,type_2_hbond,type_3_hbond]

def extract_hbonds_per_frame(dcd_traj,LUP):
    hbonds_per_frame = []
    start_0 = time()
    start = start_0
    for f in range(dcd_traj.n_frames):
        
        hbonds = md.baker_hubbard(dcd_traj[f])
        #print(f'INFO : frame # {f} , hbonds {hbonds.shape[0]}')
        hbonds_per_frame.append(report_inter_hbond(LUP,hbonds))
        if f % 100 == 0 :
            print(f'[INFO] Time elapsed frame # {f} ==> {round(time() - start,2) } seconds')
            start = time()
    print(f'[INFO] Total Time elapsed for {dcd_traj.n_frames} ==> {round(time() - start_0,2) } seconds')
    return hbonds_per_frame


def main():

    out_dir = args.out_dir
    dcdFile = args.dcd
    psfFile = args.psf
    hbondExtFlag = args.hbondOnly 
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    
    #dcdFile = './amarolab_covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab/spike_WE.dcd'
    #psfFile = './amarolab_covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab/spike_WE_renumbered.psf'
    trajDir = out_dir

    start_t = time()
    print(f'[INFO] loading dcd file[s] : {dcdFile}')
    dcd_traj = md.load(dcdFile, top = psfFile)
    print(f'[INFO] dcd loaded in {round(time() - start_t,2)} seconds')
    cur_time = time()
    atom_id_LUP = build_atom_lup_4_common_features(dcd_traj)
    atom_id_LUP['GLY'] =[]

    for gly in extract_glycan_residues_4m_pdb(dcd_traj):
        for gly_atom in get_atom_ids_for_feature(dcd_traj,f"resn =~ {gly}"):
            atom_id_LUP['GLY'].append(gly_atom)



    atom_id_LUP['RBD_CA'] = get_atom_ids_for_feature(dcd_traj,"resid >= 330 and resid <= 530 and name == CA")
    atom_id_LUP['CH_CA'] = get_atom_ids_for_feature(dcd_traj,"((resid >= 747 and resid <= 784) or (resid >= 946 and resid <= 967) or (resid >= 986 and resid <= 1034)) and (name == CA)")

    ######### h-bond extraction trigger ################
    hbond_feature_df = pd.DataFrame(columns=['RBD_GYC_interHbondTyp1','RBD_GYC_interHbondTyp2', 'GYC_IntraHbond'],data = extract_hbonds_per_frame(dcd_traj,LUP=atom_id_LUP))
    hbond_feature_df.to_csv(f'{out_dir}/Feature_hBond.csv')
    print(f'[INFO] H-Bond Feature Extraction converged successfully ==> {out_dir}/Feature_hBond.csv' )
    
    if hbondExtFlag:
        print(f'[INFO] Total Time elapsed  : {round(time() - start_t,2)} seconds')
        exit()

    print(f'[INFO] Filtering Atoms corresponding to RBD, CH, backbone, sidechain and GLY')
    ## build trajectore from filtered atoms of RBD and GLY, use only frame-0
    traj_GLY_F0 = derive_trajectory(dcd_traj,atom_key='GLY', LUP=atom_id_LUP)
    traj_RBD_F0 = derive_trajectory(dcd_traj,atom_key='RBD_CA', LUP=atom_id_LUP)


    ## get COM of all chains in GLY-trajectory of frame1
    GLY_RBD_proximity_df = pd.DataFrame(columns=['chain','x','y','z'])
    GLY_chain_LUP = {f'chainID_{c}' : get_atom_ids_for_feature(traj_GLY_F0,f'chainid == {c}')  for c in range(traj_GLY_F0.n_chains)}
    GLY_chain_COM = {}
    for k in GLY_chain_LUP.keys():
            #print(f'[INFO] deriving Chain {k} tracjectory')
            chainObj = derive_trajectory(traj_GLY_F0,frames=list(range(traj_GLY_F0.n_frames)),atom_key= f'{k}',LUP = GLY_chain_LUP)
            GLY_chain_COM[k] = md.compute_center_of_mass(chainObj)
            GLY_RBD_proximity_df = GLY_RBD_proximity_df.append(pd.DataFrame(columns =['x','y','z'], data =[GLY_chain_COM[k][0]]).assign(chain = k))


    print(f'[INFO] Time elapsed for Atom-Filtering :  {round(time() - cur_time,2)} seconds')
    cur_time =  time()
    RBD_COM = md.compute_center_of_mass(traj_RBD_F0)
    GLY_RBD_proximity_df['RBD_x'] = RBD_COM[0][0]
    GLY_RBD_proximity_df['RBD_y'] = RBD_COM[0][1]
    GLY_RBD_proximity_df['RBD_z'] = RBD_COM[0][2]

    print(f'[INFO] Filtering Glycans based on proximity to RBD')
    ### calculate the distance in center of mass of RBD vs GLY_chains and drop all GLY chains > 20A
    GLY_RBD_proximity_df = (GLY_RBD_proximity_df
        .assign(distance = lambda df : np.sqrt((np.square(df.x - df.RBD_x) + np.square(df.y - df.RBD_y) + np.square(df.z - df.RBD_z)).astype(float)))
        .sort_values(by = ['distance'],ascending=True)    
    )
    GLY_chain_ids_next_to_RBD =  [int(s_c[0]) for s_c in GLY_RBD_proximity_df[GLY_RBD_proximity_df.distance <= 4].chain.str.extract(r'chainID_(\d+)').values]
    #GLY_chain_ids_next_to_RBD, atom_id_LUP.keys()
    print(f'[INFO] Time elapsed for Filtering Glycans based on proximity to RBD :  {round(time() - cur_time,2)} seconds')

    # #### Extract feature matrix from backbone , RBD & CH without sub-fracturing into chains
    # #### Extract feature matrix from each shortlisted Glycan chain in GLY_chain_ids_next_to_RBD
    print(f'[INFO] Start Feature Extraction from derived Tracjectories')
    launchFeatureExract(dcd_traj,GLY_chain_ids_next_to_RBD,out_dir,LUP_dict=atom_id_LUP)


    # 
    # - RBD , CH  (com)
    # - sidechain + glycans -->  (G1-G70)  AI (Anand/Lorenzo)
    # - Monomer A/B/C  --> (Needs Info)  AI (Anand/Lorenzo)
    # - backbone (low prioroty) 
    # - Monomer A/B/C are comprised of group of chains. these chainIDs need to be provided by Lab/Data experts?
    # #### Read-in extracted feature per chain for RBD/backbone/CH


    featureFile_dict = {k : glob.glob(f'{out_dir}/{k}/*csv') for k in ['backbone','RBD_CA', 'CH_CA', 'GLY']}   
    #print(featureFile_dict)
    feature_df = pd.DataFrame()
    for k in featureFile_dict.keys():
        for f in  featureFile_dict[k]:
            if 'chain' in f:
                cid = int(os.path.basename(f).split('_')[-1].replace('.csv',''))
                if k == 'GLY':
                    if cid in GLY_chain_ids_next_to_RBD:
                        feature_df = feature_df.append(pd.read_csv(f).assign(feature = k).assign(chainID = cid )  )
                    else :
                        continue
            else :
                feature_df = feature_df.append(pd.read_csv(f).assign(feature = k).assign(chainID = 0)  )
    feature_df = feature_df.drop(['Unnamed: 0'],axis=1)
    

    feature_df = feature_df.assign(feature_chain = lambda df  : df.feature +  df.chainID.astype(str)).rename(columns={'rofgyration': 'ROF', 'density': 'DNS', 'rmsd' : 'RMSD'})
    #print(feature_df.head())

    common_features = ['ROF','DNS','RMSD']
    final_feature_df = pd.DataFrame(columns=['frame'], data = feature_df[feature_df.feature == 'RBD_CA'].frame.to_list())
    for c in common_features:
                final_feature_df[f'RBD_CA0:{c}'] = feature_df[feature_df.feature_chain == 'RBD_CA0'][c]
            
    for f in sorted(feature_df.feature_chain.unique()):
        if f != 'RBD_CA0' and not re.match(r'sidechain\d+',f) :
            match_object = re.match(r'GLY(\d+)',f)
            if match_object != None:
                if int(match_object.group(1)) not in GLY_chain_ids_next_to_RBD:
                    continue
            final_feature_df[f'RBD__2__{f}'] = extract_distance_metric(feature_df[feature_df.feature_chain == 'RBD_CA0'], feature_df[feature_df.feature_chain == f])
            for c in common_features:
                final_feature_df[f'{f}:{c}'] = feature_df[feature_df.feature_chain == f][c]
    final_feature_df.to_csv(f'{out_dir}/FinalExtractedFeature.csv')
    print(f'[INFO] Feature Extraction converged successfully ==> {out_dir}/FinalExtractedFeature.csv' )
    print(f'[INFO] Total Time elapsed  : {round(time() - start_t,2)} seconds')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="profile dcd to extract metrics from each frame")
    parser.add_argument("-dcd", type=str, help="dcd file",required=True, nargs='+')
    parser.add_argument("-psf", type=str, help="psf file",required=True)
    parser.add_argument("-hbondOnly", type=bool, help="extract H-bond Feature Only", required=False, default=True)
    parser.add_argument("-multiprocess", required=False, type=bool, default=False,help="multi-cpu for parallel processing")
    parser.add_argument("-out_dir", required=False, default= f'{os.getcwd()}/dcdMetricRunDir')

    args = parser.parse_args()
    print(f' CPU cores : {multiprocessing.cpu_count()} , avilable RAM on m/c {psutil.virtual_memory()[0]/pow(1024,3)} Gigs')
    main()