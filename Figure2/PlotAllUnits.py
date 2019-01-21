import os
from Figure2.util import get_unit_details, plot_ISI, plot_unit_waveform,plot_unit_stability, plot_unit_quality,plot_firing_rate,plot_or_tuning
from Figure2.pool_backend import process_session
from Util.util import get_unit_id, get_unit_depth, get_subject_from_session, get_session_date
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import pandas as pd

import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())
session_records = []

label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

session_type = ['phys', 'behaved']
base_locs = ['/home/bsriram/data/DetailsProcessedPhysOnly', '/home/bsriram/data/DetailsProcessedBehaved']
save_locs = ['/home/bsriram/data/Analysis/SummaryDetails/DetailsProcessedPhysOnly', '/home/bsriram/data/Analysis/SummaryDetails/DetailsProcessedBehaved']
neuron_save_loc = '/home/bsriram/data/Analysis/SummaryDetails'

def plot_unit(fig_ref, unit, loc, this_neuron_record):
    ax1 = plt.subplot2grid((5,3),(0,0),colspan=2)
    this_neuron_record = plot_ISI(unit, ax1, this_neuron_record)
    ax2 = plt.subplot2grid((5,3),(0,2),colspan=1)
    this_neuron_record = plot_unit_waveform(unit, ax2, this_neuron_record)
    ax3 = plt.subplot2grid((5,3),(1,0),colspan=2)
    this_neuron_record = plot_unit_stability(unit, loc, this_neuron_record, ax=ax3)
    ax4 = plt.subplot2grid((5,3),(1,2),colspan=1)
    this_neuron_record = plot_unit_quality(unit, loc, this_neuron_record, ax=ax4)
    ax5 = plt.subplot2grid((5,3),(2,0),colspan=2)
    this_neuron_record = plot_firing_rate(unit, loc, this_neuron_record, ax=ax5)
    ax6 = plt.subplot2grid((5,3),(2,2),colspan=2,polar=True)
    this_neuron_record = plot_or_tuning(unit, loc, this_neuron_record, ax=ax6)

    return this_neuron_record

def collect_result(result):
    global session_records
    session_records.append(result)

all_neuron_records = []

# for phys
base_loc = base_locs[0]
folder_list = os.listdir(base_loc)
folder_list.sort()
for session_folder in folder_list:
    pool.apply_async(process_session, args=(0,session_folder,base_loc,), callback=collect_result)

    
# for behaved
base_loc = base_locs[1]
folder_list = os.listdir(base_loc)
folder_list.sort()
for session_folder in folder_list:
    print('hello')
    pool.apply_async(process_session, args=(1,session_folder,base_loc,), callback=collect_result)

# pickle the results                
with open(os.path.join(neuron_save_loc,'NeuronData.pickle'),'wb') as f:
    pickle.dump(session_records, f, pickle.HIGHEST_PROTOCOL)