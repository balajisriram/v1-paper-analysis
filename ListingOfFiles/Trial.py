import importlib.machinery
import types
import pickle
import os
import pdb
import scipy

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
base_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly'

for sess in os.listdir(base_loc):
    tR_file = [f for f in os.listdir(os.path.join(base_loc,sess)) if f.startswith('trialRecords')] 
    print(sess,":",tR_file)
    #scipy.io.loadmat(os.path.join(bas_loc,sess,tR_file[0])) # verify I can load all of this