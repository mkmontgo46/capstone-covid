import os

dirname = '/net/jam-amaro-shared/dse_project/Spike_Dataset/TRAJECTORIES_spike_mutant_prot_glyc_amarolab/'
num_dcds = 6
file_prefix = 'spike_mutant_prot_glyc_amarolab'
save_file = 'FinalExtractedFeature.csv'

for i in range(num_dcds):
    # Run feature extraction
    os.system('python dcd_featureExtract.py -dcd ' + os.path.join(dirname,file_prefix + '_' + str(i+1) + '.dcd') + ' -psf ' + os.path.join(dirname,file_prefix + '.psf') + ' -out_dir ' + os.path.join(dirname,'results'))
    # Rename results file
    os.rename(os.path.join(dirname,'results',save_file),os.path.join(dirname,'results',save_file.replace('.csv','_' + str(i+1) + '.csv')))
    
print('All files finished')