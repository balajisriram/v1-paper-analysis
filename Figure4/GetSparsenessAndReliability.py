import os
import sys
sys.path.append(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\Figure3')
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle
import pprint
import scipy
import pandas
from statsmodels.stats.outliers_influence import summary_table
from util import get_base_depth,get_unit_depth,get_subject_from_session,get_OSI, get_vector_sum,get_orientation_tuning_stepname,get_frame_channel,get_unit_id,get_short_response_stepname
ppr = pprint.PrettyPrinter(indent=2)


def get_all_short_duration_responses(location):
    unit_details_across_sessions = {}
    
    for sess in os.listdir(location):
        print(sess)    
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        try:
            stepnames,durations,contrasts = get_short_response_stepname(sess)
        except:
            pdb.set_trace()
        framechan,shift = get_frame_channel(sess)
        
        if not stepnames: 
            print(sess,'why no stepname???')
            continue
        
        trial_numbers = numpy.asarray(data['trial_records']['trial_number'])
        step_names = numpy.asarray(data['trial_records']['step_name'])
        contrasts = numpy.asarray(data['trial_records']['contrast'])
        max_durations = numpy.asarray(data['trial_records']['max_duration'])
        phases = numpy.asarray(data['trial_records']['phase'])
        orientations = numpy.asarray(data['trial_records']['orientation'])
        
        that_session_details = {}
        that_session_details['session'] = sess
        that_session_details['subject'] = get_subject_from_session(sess)
        
        which_step = numpy.full(step_names.shape,False,dtype=bool)
        for stepname in stepnames:
            which_step = which_step | (step_names==stepname)
            
        trial_numbers = trial_numbers[which_step]
        step_names = step_names[which_step]
        contrasts = contrasts[which_step]
        max_durations = max_durations[which_step]
        phases = phases[which_step]
        orientations = orientations[which_step]
        frame_start_index = []
        frame_end_index = []
        responses_that_session = {}
        stimuli_that_session = {}
        stimuli_that_session['trial_numbers'] = trial_numbers
        stimuli_that_session['step_names'] = step_names
        stimuli_that_session['contrasts'] = contrasts
        stimuli_that_session['max_durations'] = max_durations
        stimuli_that_session['phases'] = phases
        stimuli_that_session['orientations'] = orientations
        for trial_number in trial_numbers:
            event_for_trial = data['trial_records']['events']['ttl_events'][trial_number]
            if shift:
                try:
                    frame_start_index.append(event_for_trial[framechan]['falling'][1])
                except:
                    frame_start_index.append(None)
                try:
                    frame_end_index.append(event_for_trial[framechan]['falling'][-2])
                except:
                    frame_end_index.append(None)                
            else:
                try:
                    frame_start_index.append(event_for_trial[framechan]['falling'][0])
                except:
                    frame_start_index.append(None)
                try:
                    frame_end_index.append(event_for_trial[framechan]['falling'][-1])
                except:
                    frame_end_index.append(None)
            
        frame_start_index = numpy.asarray(frame_start_index)
        frame_end_index = numpy.asarray(frame_end_index)
        uids = []
        manual_qualities = []
        unit_depths = []
        for i,unit in enumerate(data['spike_records']['units']):
            if not unit['manual_quality'] in ['good','mua']: continue
            uid = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
            uids.append(uid)
            manual_qualities.append(unit['manual_quality'])
            unit_depths.append(get_unit_depth(sess,unit['y_loc']))
            spike_time = numpy.squeeze(numpy.asarray(unit['spike_time']))
            spike_raster_start_plus_1000 = []
            for trial_number in trial_numbers:
                idx = trial_numbers==trial_number
                if frame_start_index[idx] and frame_end_index[idx]:
                    start_time = frame_start_index[idx][0]/30000
                    end_time = frame_end_index[idx][0]/30000
                    spike_raster_start_plus_1000.append(spike_time[numpy.bitwise_and(spike_time>start_time,spike_time<(start_time+1))]-start_time)
                else:
                    print(trial_number)
                    spike_raster_start_plus_1000.append(None)
            responses_that_session[uid] = spike_raster_start_plus_1000
        unit_details = {'uid':uids,'manual_quality':manual_qualities,'unit_depths':unit_depths}
        that_session_details['unit_responses'] = pandas.DataFrame(data=responses_that_session)
        that_session_details['stimuli'] = pandas.DataFrame(data=stimuli_that_session)
        that_session_details['unit_details'] = unit_details
        unit_details_across_sessions[sess] = that_session_details
        print('done session')
    return unit_details_across_sessions

def get_sparseness(details,feature_name='spike_raster_start_plus_500'):
    sessions = details.keys()
    sessions = list(sessions)
    sparseness_total = pandas.DataFrame()
    current_ind_max = 0
    for sess in sessions:
        print(sess)
        session_keys = details[sess].keys()
        session_keys = list(session_keys)
        stimuli = details[sess]['stimuli']
        
        contrasts = stimuli['contrasts']
        orientations = stimuli['orientations']
        max_durations = stimuli['max_durations']
        phases = stimuli['phases']
        trial_numbers = stimuli['trial_numbers']
        
        unit_responses = details[sess]['unit_responses']
        
        def len_of_vals_less_than_500(x):
            try:
                temp = numpy.sum(numpy.asarray(x)<0.5)
            except:
                temp = None
            return temp
        
        def sparseness(x):
            return sum(x!=0)/len(x)
        temp = unit_responses.applymap(len_of_vals_less_than_500)
        temp2 = temp.apply(sparseness,axis=1)
        data = {'sparseness':temp2, 'contrasts':contrasts, 'orientations':orientations,'max_durations':max_durations,'phases':phases,'trial_numbers':trial_numbers, 'session': numpy.squeeze(numpy.matlib.repmat(sess,temp.shape[0],1))}
        current_sparseness = pandas.DataFrame(data=data)
        current_sparseness.index = current_sparseness.index+current_ind_max
        current_ind_max = current_sparseness.index[-1]+1       
        
        temp = [sparseness_total,current_sparseness]
        sparseness_total = pandas.concat(temp)
        print('session done')
    return sparseness_total

def plot_sparseness_ctr_dur(sparseness):
    print('total number of trials')
    
if __name__=="__main__":
    #ORIENTATION DETAILS FOR PHYS
    # location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly'
    # unit_details = get_all_short_duration_responses(location)
    
    # with open('ShortDurationResponses_DataFramed_PhysOnly_Falling.pkl','wb') as f:
       # pickle.dump(unit_details,f)
       
    #ORIENTATION DETAILS FOR BEHAVED
    # location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved'
    # unit_details = get_all_short_duration_responses(location)
    
    # with open('ShortDurationResponses_DataFramed_Behaved_Falling.pkl','wb') as f:
        # pickle.dump(unit_details,f)

    location = r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis'
    with open(os.path.join(location,'ShortDurationResponses_DataFramed_PhysOnly_Falling.pkl'),'rb') as f:
        unit_details = pickle.load(f)
        
    sparseness = get_sparseness(unit_details)
    
    plot_sparseness_ctr_dur(sparseness)
    
    
    
