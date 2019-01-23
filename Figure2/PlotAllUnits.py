import os
from Figure2.util import get_unit_details, plot_ISI, plot_unit_waveform,plot_unit_stability, plot_unit_quality,plot_firing_rate,plot_or_tuning,plot_unit
from Util.util import get_unit_id, get_unit_depth, get_subject_from_session, get_session_date
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import pandas as pd
import matplotlib as mpl
import multiprocessing as mp
import h5py

session_records = []
session_type = ['phys', 'behaved']

# linux
base_locs = ['/home/bsriram/data/DetailsProcessedPhysOnly', '/home/bsriram/data/DetailsProcessedBehaved']
save_locs = ['/home/bsriram/data/Analysis/SummaryDetails/DetailsProcessedPhysOnly', '/home/bsriram/data/Analysis/SummaryDetails/DetailsProcessedBehaved']
neuron_save_loc = '/home/bsriram/data/Analysis/SummaryDetails'

# windows
# base_locs = [r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly', r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved']
# save_locs = [r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\SummaryDetails\DetailsProcessedPhysOnly', r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\SummaryDetails\DetailsProcessedBehaved']
# neuron_save_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\SummaryDetails'
    
def process_session(session_type_idx,session_folder,base_loc):
    # print('accessible')
    # neurons_that_session = []
    # summary_filename = 'UnitSummaryDetails_%s.pdf' % session_folder
    # with PdfPages(os.path.join(save_locs[session_type_idx],summary_filename)) as pdf:
        # with open(os.path.join(base_loc,session_folder,'spike_and_trials.pickle'),'rb') as f:
            # data = pickle.load(f)
        # for unit in data['spike_records']['units']:
            # print('session::{0},shank::{1},cluster::{2}'.format(session_folder,unit['shank_no'],unit['cluster_id']))
                 
            # if unit['manual_quality'] in ['good','mua']:
                # this_neuron_record = {}
                # this_neuron_record['session'] = session_folder
                # this_neuron_record['session_type'] = session_type[session_type_idx]
                # this_neuron_record['session_date'] = get_session_date(session_folder)
                # this_neuron_record['subject'] = get_subject_from_session(session_folder)
                # this_neuron_record['probe_name'] = data['spike_records']['probe_name']
                # this_neuron_record['sample_rate'] = data['spike_records']['sample_rate']
                # this_neuron_record['strong_threshold'] = data['spike_records']['strong_threshold']
                # this_neuron_record['weak_threshold'] = data['spike_records']['weak_threshold']
                # this_neuron_record['spikes_direction'] = data['spike_records']['spikes_direction']
                # this_neuron_record['high_pass_hi_f'] = data['spike_records']['high_pass_hi_f']
                # this_neuron_record['high_pass_lo_f'] = data['spike_records']['high_pass_lo_f']
                # this_neuron_record['session_duration'] = data['spike_records']['duration']
                
                # this_neuron_record['num_spikes'] = unit['num_spikes']
                # this_neuron_record['shank_no'] = unit['shank_no']
                # this_neuron_record['cluster_id'] = unit['cluster_id']
                # this_neuron_record['unit_id'] = get_unit_id(session_folder, unit['shank_no'], unit['cluster_id'])
                # this_neuron_record['x_loc'] = unit['x_loc']
                # this_neuron_record['depth'] = get_unit_depth(session_folder,unit['y_loc'])

                
                # fig = plt.figure(figsize=(8.5,11), dpi=300, frameon=False, facecolor=None)
                # this_neuron_record=plot_unit(fig, unit, os.path.join(base_loc,session_folder), this_neuron_record)
                # fig.subplots_adjust(wspace=0.25, hspace=0.25)
                # plt.suptitle('shank::%d, cluster::%d, quality::%s' % (unit['shank_no'], unit['cluster_id'], unit['manual_quality']))
                # pdf.savefig()  # saves the current figure into a pdf page
                # plt.close()
                # neurons_that_session.append(this_neuron_record)
        # fig = plt.figure(figsize=(8.5,11), dpi=300, frameon=False, facecolor=None)
        # ax = plt.subplot(1,1,1)
        # ax.text(0.5,0.5,'DONE!')
        # pdf.savefig()
        # plt.close()
    return session_folder

    
def collect_result(result):
    session_folder = result
    print('DONE::',session_folder)
    # with h5py.File(os.path.join(neuron_save_loc,'NeuronData.hdf'),'a') as f:
        # f.create_dataset(session_folder, data=neurons_that_session)
    with h5py.File(os.path.join(neuron_save_loc,'NeuronData.hdf'),'a') as f:
        f.create_dataset(session_folder, data=session_folder)
    with open(os.path.join(neuron_save_loc,'FinishedSession.txt'),'a') as f:
        f.write(session_folder+'\n')

def handle_error(er):
    print(er)
        
if __name__=='__main__':
    pool = mp.Pool(8) # 8 simulataneous processes
    print('Starting phys...')
    # for phys
    base_loc = base_locs[0]
    folder_list_phys = os.listdir(base_loc)
    folder_list_phys.sort()
    sess_idx_phys = [0 for x in folder_list_phys]
    base_locs_phys = [base_loc for x in folder_list_phys]
    collated_phys = [(x,y,z) for x,y,z in zip(sess_idx_phys,folder_list_phys,base_locs_phys)]
    
    
    # for behaved
    base_loc = base_locs[1]
    folder_list_beh = os.listdir(base_loc)
    folder_list_beh.sort()
    sess_idx_beh = [1 for x in folder_list_phys]
    base_locs_beh = [base_loc for x in folder_list_beh]
    collated_beh = [(x,y,z) for x,y,z in zip(sess_idx_phys,folder_list_phys,base_locs_phys)]
    
    # now collate them
    collated_all = collated_phys
    for x in collated_beh:
        collated_all.append(x)
    
    for job in collated_all:
        pool.apply_async(process_session, args=(job[0],job[1],[job3]), callback=collect_result, error_callback=handle_error)
    
    print('Done for phys')
    

