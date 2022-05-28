def add_gly(names,idx,chain_idx,gly_idx,pos=None):
    positions = ['N17','N61','N74','N122','N149','N165','N234','N282','N331','N343','N603','N616','N657','N709','N717','N801','N1074','N1098','N1134','N1158','N1173','N1194','T323']
    chains = ['Monomer A','Monomer B','Monomer C']
    if pos is None:
        position = positions[gly_idx]
    else:
        position = pos
    
    segname = 'GLY' + str(idx)
    chainname = chains[chain_idx]
    
    # Add to dictionary
    names[segname] = {'position':position, 'chain':chainname}
    return names

def rename_feat(featname):
    featname = featname.replace('RBD__2__','RBD distance to ')
    if ':ROF' in featname:
        featname = 'Radius of Gyration for ' + featname.replace(':ROF','')
    if ':RMSD' in featname:
        featname = 'RMSD for ' + featname.replace(':RMSD','')
    if '_x' in featname:
        featname = 'x location for ' + featname.replace('_x','')
    if '_y' in featname:
        featname = 'y location for ' + featname.replace('_y','')
    if '_z' in featname:
        featname = 'z location for ' + featname.replace('_z','')
    featname = featname.replace('_', ' at ')
    return featname
        

def get_names():
    bionames = {}
    num_chains = 3
    num_gly = 23
    extra = 0
    for chain in range(num_chains):
        for gly in range(num_gly):
            idx = chain*num_gly+gly+1+extra
            bionames = add_gly(bionames, idx, chain, gly)
            if idx == 23:
                extra = 1
                bionames = add_gly(bionames,24,chain,None,pos='S325')
     
    bionames['CH_CA0'] = {'chain':'Core','position':'CH'}
    bionames['backbone0'] = {'chain':'Core','position':'Backbone'}
    bionames['RBD_CA0'] = {'chain':'RBD','position':'RBD'}
    return bionames

def gly_4m_featname(featname):
    return featname.replace(':ROF','').replace('RBD__2__','').replace(':RMSD','').replace('_x','').replace('_y','').replace('_z','')

def get_elem(featname,elem):
    gly = gly_4m_featname(featname)
    if 'GLY' in gly:
        gly1 = 'GLY' + str(int(gly.replace('GLY',''))+1) #zero indexing caused glycan segnames to be off by 1
    else:
        gly1 = gly
    bionames = get_names()
    if gly1 in bionames.keys():
        if elem == 'feat':
            return featname.replace(gly,bionames[gly1]['position']) + '_' + bionames[gly1]['chain']
        else:
            return bionames[gly1][elem]
    else:
        return featname
