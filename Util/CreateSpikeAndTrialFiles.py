import importlib.machinery
import types
import pickle
import os
import pdb
import scipy
import shutil

def import_module(name,file):
    loader = importlib.machinery.SourceFileLoader(name,file)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

OE = import_module("OpenEphys",r"C:\Users\bsriram\Desktop\Code\analysis-tools\Python3\OpenEphys.py")
remote_loc = r'D:\PhysiologySession\Sessions';

def create_dat_file(src_base, des_base, name, dref, filename):
    src_folder = os.path.join(src_base,name)
    des_folder = os.path.join(des_base,name)
    OE.pack_2(src_folder,destination=des_folder,dref=dref,filename=filename)

# get the relevant modules
TRUtil = import_module("TrialRecordsUtil","C:\\Users\\bsriram\\Desktop\\Code\\V1PaperAnalysis\\Util\\TrialRecordsUtil.py")
base_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved'

for sess in os.listdir(base_loc): 
    #if not (('bas070' in sess) or ('bas072' in sess) or ('bas074' in sess) or ('bas079' in sess)): 
    #    continue
    #else:
    print("Running for ", sess)    
    tR_file = [f for f in os.listdir(os.path.join(base_loc,sess)) if f.startswith('trialRecords')] 
    print(sess,":",tR_file)
    #temp = scipy.io.loadmat(os.path.join(base_loc,sess,tR_file[0])) # verify I can load all of this
    
    full_loc = os.path.join(base_loc,sess)
    if 'spike_and_trials.pickle' in os.listdir(full_loc): 
        print('Already done for ',full_loc)
        continue
        
    # check if the .dat is available
    DATFile = [f for f in os.listdir(full_loc) if f.endswith('.dat')]
    if not DATFile: 
        print('DAT file not available. Need to make from external source')
        # find if the kwik had CAR or CMR
        kwik_file_name = [f for f in os.listdir(full_loc) if f.endswith('.kwik')]
        if 'CMR' in kwik_file_name[0]:
            dref='med'
        elif 'CAR' in kwik_file_name[0]:
            dref='ave'
        else:
            dref=None

        # now use the kwik_file_name to get the dat filename
        kwik_file_split = os.path.splitext(kwik_file_name[0])
        dat_file_name = kwik_file_split[0]+'.dat'
        
        create_dat_file(remote_loc, base_loc, sess,dref,dat_file_name)
        DATFile=[dat_file_name]
    else:
        print('DAT file found: ',DATFile)
        
    spike_and_trial_details = TRUtil.load_spike_and_trial_details(full_loc)
    pickle_filename = os.path.join(full_loc,'spike_and_trials.pickle')
    print("Saving pickle to file name ::",pickle_filename)
    with open(pickle_filename,'wb') as f:
        pickle.dump(spike_and_trial_details,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    os.remove(os.path.join(full_loc,DATFile[0]))