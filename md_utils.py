import mdtraj as md
from biopandas.pdb import PandasPdb
import glob
import os
import sys, re
import pandas as pd
import plotly_express as px
import glycan_bionames

def extract_glycan_residues_4m_pdb(dcdObj):
    '''Extract glycans from dcd object. Glycans=atoms w/ segment_id == G1, G2, etc'''
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
    '''Get atom ids for top-level structures using mdtraj'''
    try:
        result = (i for i in dcd_traj.top.select(feature))
    except :
        print(f'[ERROR] {feature} not recognized for atom filtering')
        result = []
    else :
        #print(f'[INFO] # of atoms : {len(list(result))} filtered for {feature}')
        return list(result)

def build_atom_lup_4_common_features(dcd_traj,flist = ['protein', 'backbone','sidechain']):
    '''Pull atoms for all top-level structures from dcd'''
    return {f: get_atom_ids_for_feature(dcd_traj,f) for f in flist}

def get_xyz_perFrame(traj,atom_ids):
    return pd.DataFrame(columns=['x','y','z'], data=traj.xyz[0,atom_ids])

def gen_xyz_Table_4_LUP(traj, LUP , keyNames =['sidechain','RBD_CA', 'CH_CA', 'GLY','backbone'] ):
    frame_0_coord_df = pd.DataFrame(columns=['type','typeID','x','y','z'])
    i = 0; frames = []
    for k in LUP.keys():
        if k in keyNames:
            frames.append(get_xyz_perFrame(traj,LUP[k]).assign(type = k).assign(typeID = i))
            i += 1
#             frame_0_coord_df = (frame_0_coord_df
#             .append()
#                                )
    frame_0_coord_df = pd.concat(frames)
            
    return frame_0_coord_df


def gly_4m_featname(featname):
    return featname.replace(':ROF','').replace('RBD__2__','').replace(':RMSD','').replace('_x','').replace('_y','').replace('_z','').replace('GLY','')

def load_traj(trajDir):
    '''Parse dcd and psf files from input directory and load trajectory'''
    # Get files with .dcd and .psf extensions
    dcdFiles = glob.glob(os.path.join(trajDir,'*.dcd'))
    psfFiles = glob.glob(os.path.join(trajDir,'*.psf'))
    
    # Load first dcd file
    traj = md.load(dcdFiles[0], top = psfFiles[0])
    
    return traj

def parse_traj(traj):
    '''Get atom ids for all elements in input trajectory'''
    atom_id_LUP = build_atom_lup_4_common_features(traj)
    atom_id_LUP['GLY'] =[]
    for gly in extract_glycan_residues_4m_pdb(traj):
        for gly_atom in get_atom_ids_for_feature(traj,f"resn =~ {gly}"):
            atom_id_LUP['GLY'].append(gly_atom)

    atom_id_LUP['RBD_CA'] = get_atom_ids_for_feature(traj,"resid >= 330 and resid <= 530 and name == CA")
    atom_id_LUP['CH_CA'] = get_atom_ids_for_feature(traj,"((resid >= 747 and resid <= 784) or (resid >= 946 and resid <= 967) or (resid >= 986 and resid <= 1034)) and (name == CA)")
    
    return atom_id_LUP

def assign_marker_size(atom_type):
    '''Assign marker size and symbol for an atom of input type'''
    bkg = ['backbone','sidechain','GLY']
    if atom_type in bkg:
        return 1
    else:
        return 8

def viz_traj(traj,atom_id_LUP, dfFeats,title_str,title_clr):
    '''Display trajectory with top features highlighted'''
    # Get names of substructures
    dfFeats.sort_values(by='importance',axis=0,inplace=True,ascending=False)
    feats = []; 
    bionames = {'sidechain':'Sidechain','RBD_CA':'RBD','CH_CA':'Central Helix','GLY':'Glycans','backbone':'Backbone'}
    for i in dfFeats['feats']:
        try:
            featname = f'G{int(gly_4m_featname(i))+1}'
            feats.append(featname)
            glyname = f'GLY{int(gly_4m_featname(i))}'
            bionames[featname] = glycan_bionames.get_elem(glyname,'position') + '_' + glycan_bionames.get_elem(glyname,'chain')
        except:
            continue

    for j in feats[:5]:
        name = 'segname ' + j
        atom_id_LUP[j] = traj.top.select(name)
    
    
    # Prep most important features for visualization
    keyNames =['sidechain','RBD_CA', 'CH_CA', 'GLY','backbone']+feats[:5]
    coord_df = gen_xyz_Table_4_LUP(LUP=atom_id_LUP,traj=traj, keyNames =keyNames)
    
    # Rename features to use bionames
    coord_df['Chain'] = coord_df.apply(lambda row: bionames[row['type']],axis=1)
    
    # Add marker size info
    coord_df['marker_size'] = coord_df.apply(lambda row: assign_marker_size(row['type']),axis=1)
    
    # Create custom colormap
    cmap = {}; bkg = ['Backbone','Sidechain','Glycans']; ct = 2
    for s in coord_df['Chain'].unique():
        if s in bkg:
            cmap[s] = '#7f7f7f'
        elif s == 'RBD':
            cmap[s] = px.colors.qualitative.Plotly[0]
        elif s == 'Central Helix':
            cmap[s] = px.colors.qualitative.Plotly[1]
        else:
            cmap[s] = px.colors.qualitative.Plotly[ct]
            ct += 1
    
    # Display most important features
    fig1 = px.scatter_3d(coord_df, title=title_str, x='x', y='y', z='z',
              color='Chain',width=800,height=800,opacity=1, template='simple_white',
                         size = 'marker_size', color_discrete_map = cmap,
                )
    
    # Remove white border around markers
    fig1.update_traces(line=dict(width=0))
    
    # Remove tick labels on all 3 axes
    fig1.update_layout(scene=dict(xaxis=dict(showticklabels=False),
                             yaxis = dict(showticklabels=False),
                             zaxis = dict(showticklabels=False)),
                       title={'font':{'color':title_clr}})
    return fig1