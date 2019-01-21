from Figure2.util import get_unit_details, plot_ISI, plot_unit_waveform,plot_unit_stability, plot_unit_quality,plot_firing_rate,plot_or_tuning
from Util.util import get_unit_id, get_unit_depth, get_subject_from_session, get_session_date
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import pandas as pd
def process_session(session_type_idx,session_folder,base_loc):
    neurons_that_session = []
    summary_filename = 'UnitSummaryDetails_%s.pdf' % session_folder
    with PdfPages(os.path.join(save_locs[session_type_idx],summary_filename)) as pdf:
        with open(os.path.join(base_loc,session_folder,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        for unit in data['spike_records']['units']:
            print('session::{0},shank::{1},cluster::{2}'.format(session_folder,unit['shank_no'],unit['cluster_id']))
                 
            if unit['manual_quality'] in ['good','mua']:
                this_neuron_record = {}
                this_neuron_record['session'] = session_folder
                this_neuron_record['session_type'] = session_type[session_type_idx]
                this_neuron_record['session_date'] = get_session_date(session_folder)
                this_neuron_record['subject'] = get_subject_from_session(session_folder)
                this_neuron_record['probe_name'] = data['spike_records']['probe_name']
                this_neuron_record['sample_rate'] = data['spike_records']['sample_rate']
                this_neuron_record['strong_threshold'] = data['spike_records']['strong_threshold']
                this_neuron_record['weak_threshold'] = data['spike_records']['weak_threshold']
                this_neuron_record['spikes_direction'] = data['spike_records']['spikes_direction']
                this_neuron_record['high_pass_hi_f'] = data['spike_records']['high_pass_hi_f']
                this_neuron_record['high_pass_lo_f'] = data['spike_records']['high_pass_lo_f']
                this_neuron_record['session_duration'] = data['spike_records']['duration']
                
                this_neuron_record['num_spikes'] = unit['num_spikes']
                this_neuron_record['shank_no'] = unit['shank_no']
                this_neuron_record['cluster_id'] = unit['cluster_id']
                this_neuron_record['unit_id'] = get_unit_id(session_folder, unit['shank_no'], unit['cluster_id'])
                this_neuron_record['x_loc'] = unit['x_loc']
                this_neuron_record['depth'] = get_unit_depth(session_folder,unit['y_loc'])

                
                fig = plt.figure(figsize=(8.5,11), dpi=300, frameon=False, facecolor=None)
                this_neuron_record=plot_unit(fig, unit, os.path.join(base_loc,session_folder), this_neuron_record)
                fig.subplots_adjust(wspace=0.25, hspace=0.25)
                plt.suptitle('shank::%d, cluster::%d, quality::%s' % (unit['shank_no'], unit['cluster_id'], unit['manual_quality']))
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                neurons_that_session.append(this_neuron_record)
    return neurons_that_session
