import importlib.machinery
import types
import pickle
import os
import pdb

def import_module(name,file):
    loader = importlib.machinery.SourceFileLoader(name,file)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

OE = import_module("OpenEphys",r"C:\Users\bsriram\Desktop\Code\analysis-tools\Python3\OpenEphys.py")
remote_loc = r'D:\PhysiologySession\Sessions';

def create_dat_file(src_base, des_base, name):
    src_folder = os.path.join(src_base,name)
    des_folder = os.path.join(des_base,name)
    dref = 'ave'
    OE.pack_2(src_folder,destination=des_folder,dref=dref,filename='100_raw_CAR.dat')

# get the relevant modules
TRUtil = import_module("TrialRecordsUtil","C:\\Users\\bsriram\\Desktop\\Code\\V1PaperAnalysis\\Util\\TrialRecordsUtil.py")
base_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\SessionsWithLED'
locs = ['bas081b_2017-07-28_20-31-03','m311_2017-07-31_16-15-43','m311_2017-07-31_17-08-20','m311_2017-08-01_12-05-12',
'm311_2017-08-01_13-08-00','m325_2017-08-10_13-14-51','m325_2017-08-10_14-21-22']

def create_spike_and_trial_details_pickle(locs,base_loc):
    for loc in locs:
        full_loc = os.path.join(base_loc,loc)
        if 'spike_and_trials.pickle' in os.listdir(full_loc): 
            print('Already done for ',full_loc)
            continue
        
        # check if the .dat is available
        DATFile = [f for f in os.listdir(full_loc) if f.endswith('.dat')]
        if not DATFile: 
            print('DAT file not available. Need to make from external source')
            create_dat_file(remote_loc, base_loc, loc)
        else:
            print('DAT file found: ',DATFile)
            
        spike_and_trial_details = TRUtil.load_spike_and_trial_details(full_loc)
        pickle_filename = os.path.join(full_loc,'spike_and_trials.pickle')
        print("Saving pickle to file name ::",pickle_filename)
        with open(pickle_filename,'wb') as f:
                pickle.dump(spike_and_trial_details,f,protocol=pickle.HIGHEST_PROTOCOL)    
                
if __name__=='__main__':
    