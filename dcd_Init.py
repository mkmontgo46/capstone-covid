#!/Users/harshitaarora/opt/anaconda3/bin/python


import os,sys
import numpy as np
import pandas as pd
import mdtraj as md
from biopandas.pdb import PandasPdb
import argparse

import logging
from queue import Queue
from threading import Thread
from time import time
import psutil,multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor



def download_n_extract_4m_url(filNames, url="https://amarolab.ucsd.edu/files/covid19", extract_dir = "amarolab_covid19"):
    outDir_dict = {}
    if not os.path.exists(extract_dir) :
        os.mkdir(extract_dir)
        
    for tarF in filNames :
        out_dir = extract_dir + '/' + str(os.path.basename(tarF)).split('.')[0]
        
        if not os.path.exists(out_dir) :
            os.mkdir(out_dir)
        out_file = out_dir + '/' + tarF
        if not os.path.exists(out_file) :
            print(f'Downloading {tarF} from {url} at  {out_file}')
            zip_path, _ = urllib.request.urlretrieve(url + '/' + tarF, out_file)
            with tarfile.open(name=zip_path, mode="r") as tf:
                print(f'Extracting {out_file} : {tf.getnames()}')
                outDir_dict[out_dir]  = tf.getnames()
                tf.extractall(out_dir)
        else :
            print(f' {out_file} already exists locally')
            outDir_dict[out_dir] = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if 
                                    os.path.isfile(os.path.join(out_dir, f))]
            
    return outDir_dict

def get_ext_4m_diskData(ext='pdb'):
    return {dir_key:glob.glob((dir_key + '/*' + ext)) for dir_key in  data_on_disk_dict.keys()}



def get_dcd_info(dcdObj,keys):
    return pd.DataFrame.from_dict(
    {k : [dcdObj.__getattribute__(k)] for k in keys}
    )



def metric_4m_mDtraj(dcdObj):
    return {'Rofguration' : md.compute_rg(dcdObj),
            'density' : md.density(dcdObj),
            'compute_center_of_mass' : md.compute_center_of_mass(dcdObj)
            }




# #### Start the RMSD extraction from pdb dump, redundant and not required. Incorrect instructions from Anand ####


def report_rmsd(p1,p2,delObj=True):
    s_tuple = {'main chain', 'hydrogen', 'c-alpha', 'heavy', 'carbon'}
    ret_dict = {sval : [PandasPdb.rmsd(p1.df['ATOM'], p2.df['ATOM'], s= sval)] for sval in s_tuple}
    del p1
    del p2
    return ret_dict
 

def extract_rmsd_metric_4m_dcd(dcdObj, save=True, out_dir = 'amarolab_covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab'):
    start_0 = time.time()
    start = start_0
    
    dcd_metric_df = pd.DataFrame(columns=['frame_1','frame_2','main chain', 'hydrogen', 'c-alpha', 'heavy', 'carbon'])
    if not os.path.exists (out_dir + '/PDB_DUMP'):
        os.mkdir(out_dir + '/PDB_DUMP')
        
    pdbDumpDir = out_dir + '/PDB_DUMP'
    for i in range(0,dcdObj.n_frames - 1 ,1):
        if not os.path.exists(f'{pdbDumpDir}/frame_{i}.pdb'):
            dcdObj[i].save_pdb(f'{pdbDumpDir}/frame_{i}.pdb')
        if not os.path.exists(f'{pdbDumpDir}/frame_{i+1}.pdb'):
            dcdObj[i+1].save_pdb(f'{pdbDumpDir}/frame_{i+1}.pdb')
        if i % 10 == 0:
            print(f'[INFO] extracting rmsd metric for {i} vs {i + 1} frames, time elapsed : {round((time.time() - start),2)}')
            start = time.time()
        dcd_metric_df = (dcd_metric_df
            .append(
                (pd.DataFrame
                .from_dict(report_rmsd(
                    PandasPdb().read_pdb(f'{pdbDumpDir}/frame_{i}.pdb'),
                    PandasPdb().read_pdb(f'{pdbDumpDir}/frame_{i+1}.pdb')
                ))
                .assign(frame_1 = f'frame_{i}')
                .assign(frame_2 = f'frame_{i + 1}')
                )
            )
        )
        if not save:
            os.remove(f'{pdbDumpDir}/frame_{i}.pdb')
            os.remove(f'{pdbDumpDir}/frame_{i + 1}.pdb')
   
    print(f'Total time elapsed = {round((time.time() - start_0),2)}')
    return dcd_metric_df.reset_index()
        


#rmsd_dcd_df = extract_rmsd_metric_4m_dcd(dcd_traj,save=False)
#rmsd_dcd_df.to_csv('./rmsd_metric.csv')
#####################################################################################################

def rmsd_by_mdTraj(f1,f2):
    #print(f'rmsd extraction for {f1}, {f2}')
    return md.rmsd(f1,f2)[0]

def gen_frame_tuples(dcdObj):
    return ((i+1,i) for i in range(dcdObj.n_frames -1 ))


def main() :
    
    out_dir = args.out_dir
    dcd_file = args.dcd
    psf_file = args.psf
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)  
    
    start_t = time()
    print(f'[INFO] loading dcd file : {dcd_file}')
    dcd_traj = md.load(dcd_file,top = psf_file)
    print(f'[INFO] dcd loaded in {round(time() - start_t,2)} seconds')

    dcd_info_df = get_dcd_info(dcd_traj,keys = [
     'n_frames',
     'n_atoms',
     'n_chains',
     'n_residues',
     'timestep'
    ])
    print(dcd_info_df)

    cur_time = time()

    addon_metrics = metric_4m_mDtraj(dcd_traj)
    print(f'[INFO] Extracted {list(addon_metrics.keys())} stats from dcd in {round(time() - cur_time,2)} seconds')
    print(f'[INFO] Starting rmsd metric extraction from dcd')
    
    cur_time = time()
        
    if args.multiprocess :
        addon_metrics['rmsd'] = np.zeros(shape(dcd_traj.n_frames,))
        print(f'[INFO] Initiate parallel processing, using {multiprocessing.cpu_count()} cores')
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_fxn = {executor.submit(rmsd_by_mdTraj, dcd_traj[tup[0]],dcd_traj[tup[1]]): tup 
                             for tup in list( gen_frame_tuples(dcd_traj) ) }
            for future in concurrent.futures.as_completed(future_to_fxn):
                tup = future_to_fxn[future]
                if tup[0] % 100 == 0 :
                    print(f'[INFO] completed extraction for {tup[0]} frames in {round(time() - cur_time,2)} seconds')
                data = future.result()
                addon_metrics['rmsd'][tup[0]] = data
    else :
    
        addon_metrics['rmsd'] = [rmsd_by_mdTraj(dcd_traj[i],dcd_traj[j]) for i,j in list(gen_frame_tuples(dcd_traj))]
        addon_metrics['rmsd'].append(rmsd_by_mdTraj(dcd_traj[-1],dcd_traj[0]))
    
    print(f'[INFO] rmsd metric extraction completed  {round(time() - cur_time,2)} seconds')
    
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
                        'compute_center_of_mass_0' : 'compute_center_of_mass_x',
                        'compute_center_of_mass_1' : 'compute_center_of_mass_y',
                        'compute_center_of_mass_2' : 'compute_center_of_mass_z'
                       }
            )
     .loc[:, ['frame','rofgyration','density',
     'rmsd', 
     'compute_center_of_mass_x',
     'compute_center_of_mass_y',
     'compute_center_of_mass_z',
     ]]

    )
    out_csv = f'{out_dir}/{os.path.splitext(os.path.basename(dcd_file))[0]}_ExtractedMetrics.csv'
    dcd_metric_Out_df.to_csv(out_csv)
    print(f'[INFO] Generated {out_csv}')
    print(f'[INFO] Metric extaction completed in  {round(time() - start_t,2)} seconds')

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="profile dcd to extract metrics from each frame")
    parser.add_argument("-dcd", type=str, help="dcd file")
    parser.add_argument("-psf", type=str, help="psf file")
    parser.add_argument("-multiprocess", nargs="?", type=bool, default=False,help="multi-cpu for parallel processing")
    parser.add_argument("-out_dir", nargs="?", default= f'{os.getcwd()}/dcdMetricRunDir')

    args = parser.parse_args()
    print(f' CPU cores : {multiprocessing.cpu_count()} , avilable RAM on m/c {psutil.virtual_memory()[0]/pow(1024,3)} Gigs')
    
    main()