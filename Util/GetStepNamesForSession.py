import os
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle
from scipy.interpolate import interp1d
from util import get_subject_from_session,get_frame_channel,get_orientation_tuning_stepname
          
def print_session_steps_and_details(location):
    # create lists
    for sess in os.listdir(location):
        print(sess)    
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        step_names = numpy.asarray(data['trial_records']['step_name'])
        durations = numpy.asarray(data['trial_records']['max_duration'])
        contrasts = numpy.asarray(data['trial_records']['contrast'])
        try:
            if not step_names: pdb.set_trace()
        except ValueError:
            pass
        
        for step_name in numpy.unique(step_names):
            durations_that_step = durations[step_names==step_name]
            contrasts_that_step = contrasts[step_names==step_name]
            print('\t',step_name,':',numpy.unique(durations_that_step),'\t\t',numpy.unique(contrasts_that_step))

def get_event_channels_in_session(location):
    for sess in os.listdir(location):
        print(sess)
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        trial_duations = (numpy.asarray(data['trial_records']['trial_end_index'])-numpy.asarray(data['trial_records']['trial_start_index']))/30000
        durations = numpy.asarray(data['trial_records']['max_duration'])*1./60.
        ttl_events = data['trial_records']['events']['ttl_events']
        
        ttl_iter = iter(ttl_events.items())
        k,val = next(ttl_iter)
        for k in val.keys():
            print('\tChannel:',k,'\t',numpy.size(numpy.unique(val[k]['falling'])))

def load_spike_and_trials_for_one_session(session):
    with open(os.path.join(session,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
    pdb.set_trace()

if __name__=="__main__":
    location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved'
    print_session_steps_and_details(location)
    # for sess in os.listdir(location):
        # print(sess,get_orientation_tuning_stepname(sess))
    # load_spike_and_trials_for_one_session(location)
    
    
    
