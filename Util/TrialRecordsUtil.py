import scipy
import scipy.io
import pdb
import numpy
import os
import importlib.machinery
import types
import pickle

np64 = numpy.float64


def import_module(name,file):
    loader = importlib.machinery.SourceFileLoader(name,file)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
    
def load_spike_and_trial_details(loc):
    spike_and_trial_details = {}
    print('Loading trial details')
    spike_and_trial_details['trial_records'] = load_trialrecs_to_dict(loc)
    print('Loading spike details')
    spike_and_trial_details['spike_records'] = load_spikerecs_to_dict(loc)
    return spike_and_trial_details
    
    
def load_trialrecs_to_dict(loc):
    file = [f for f in os.listdir(loc) if f.startswith('trialRecords')]
    if len(file) > 1 or len(file)==0:
        print(loc)
        error('too many or too few trial Records. how is this possible?')
    
    temp = scipy.io.loadmat(os.path.join(loc,file[0]))
    pdb.set_trace()
    tRs = temp['trialRecords'][0]
    numTrials = tRs.size
    
    trial_number = []
    refresh_rate = []
    step_name = []
    stim_manager_name = []
    trial_manager_name = []
    afc_grating_type = []
    trial_start_index = []
    trial_end_index = []
    
    pix_per_cycs = []
    driftfrequency = []
    orientation = []
    phase = []
    contrast = []
    max_duration = []
    radius = []
    annulus = []
    waveform = []
    radius_type = []
    location = []
    
    led_on = []
    led_intensity = []
    
    events = find_indices_for_all_trials(loc)
    events = get_channel_events(loc,events)
    for i,tR in enumerate(tRs):
        this_trial_number = np64(tR['trialNumber'][0][0])
        if this_trial_number in events['trial_number']:
            which_in_messages = [True if x==this_trial_number else False for x in events['trial_number']]
            this_start_index = [x for i,x in enumerate(events['start_index']) if which_in_messages[i]==True]
            this_end_index = [x for i,x in enumerate(events['end_index']) if which_in_messages[i]==True]
        else:
            continue
        
        trial_number.append(np64(tR['trialNumber'][0][0]))
        refresh_rate.append(np64(tR['refreshRate'][0][0]))
        
        step_name.append(tR['stepName'][0])
        stim_manager_name.append(tR['stimManagerClass'][0])
        trial_manager_name.append(tR['trialManagerClass'][0])
        afc_grating_type.append(tR['afcGratingType'][0])
        
        trial_start_index.append(this_start_index[0])
        trial_end_index.append(this_end_index[0])
        
        try: pix_per_cycs.append(np64(tR['stimulus'][0]['pixPerCyc'][0][0][0]))
        except: pix_per_cycs.append(np64(tR['stimulus'][0]['pixPerCycs'][0][0][0]))
        
        try: driftfrequency.append(np64(tR['stimulus'][0]['driftfrequency'][0][0][0]))
        except: driftfrequency.append(np64(tR['stimulus'][0]['driftfrequencies'][0][0][0]))
        
        try: orientation.append(np64(tR['stimulus'][0]['orientation'][0][0][0]))
        except: orientation.append(np64(tR['stimulus'][0]['orientations'][0][0][0]))
        
        try: phase.append(np64(tR['stimulus'][0]['phase'][0][0][0]))
        except: phase.append(np64(tR['stimulus'][0]['phases'][0][0][0]))
        
        try: contrast.append(np64(tR['stimulus'][0]['contrast'][0][0][0]))
        except: contrast.append(np64(tR['stimulus'][0]['contrasts'][0][0][0]))
        
        max_duration.append(np64(tR['stimulus'][0]['maxDuration'][0][0][0]))
        
        try: radius.append(np64(tR['stimulus'][0]['radius'][0][0][0]))
        except: radius.append(np64(tR['stimulus'][0]['radii'][0][0][0]))
        
        try: annulus.append(np64(tR['stimulus'][0]['annulus'][0][0][0]))
        except: annulus.append(np64(tR['stimulus'][0]['annuli'][0][0][0]))
        
        waveform.append(tR['stimulus'][0]['waveform'][0][0])
        radius_type.append(tR['stimulus'][0]['radiusType'][0][0])
        
        location.append(np64(tR['stimulus'][0]['location'][0][0]))
        
        led_on.append(np64(tR['LEDON'][0][0]))
        led_intensity.append(np64(tR['LEDIntensity'][0][0]))
        
    trial_records = dict([('trial_number',trial_number),\
                         ('refresh_rate',refresh_rate),\
                         ('step_name',step_name),\
                         ('stim_manager_name',stim_manager_name),\
                         ('trial_manager_name',trial_manager_name),\
                         ('afc_grating_type',afc_grating_type),\
                         ('trial_start_index',trial_start_index),\
                         ('trial_end_index',trial_end_index),\
                         ('pix_per_cycs',pix_per_cycs),\
                         ('driftfrequency',driftfrequency),\
                         ('orientation',orientation),\
                         ('phase',phase),\
                         ('contrast',contrast),\
                         ('max_duration',max_duration),\
                         ('radius',radius),\
                         ('annulus',annulus),\
                         ('waveform',waveform),\
                         ('radius_type',radius_type),\
                         ('location',location),\
                         ('led_on',led_on),\
                         ('led_intensity',led_intensity),\
                         ('events',events)])
                         
    return trial_records
    
def find_indices_for_all_trials(location):
    trial_number = []
    start_index = []
    end_index = []
    
    file = [f for f in os.listdir(location) if f.startswith('messages')]
    if len(file) > 1 or len(file)==0:
        print(location)
        print(file)
        print('too many or too few trial Records. how is this possible?')
    
    with open(os.path.join(location,file[0]),errors='ignore') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.rstrip('\x00') for x in content]
    
    for i,line in enumerate(content):
        if 'TrialStart' in line:
            # has TrialStart. Find trial number
            ind_start = line.find('TrialStart')
            that_trial_number = int(line[ind_start+12:])
            # now search for time index
            that_start_index = int(line[:ind_start]) 
            # look at next line for TrialEnd
            if i+1<len(content) and 'TrialEnd' in content[i+1]:
                # TrialEnd is available for that trial number
                stop_line = content[i+1] 
                ind_stop = stop_line.find('TrialEnd')
                that_stop_index = int(stop_line[:ind_stop])
                
                trial_number.append(that_trial_number)
                start_index.append(that_start_index);
                end_index.append(that_stop_index)
            elif i+2<len(content) and 'TrialEnd' in content[i+2]:
                # TrialEnd is available for that trial number. Some mistaken stuff out there
                stop_line = content[i+2]
                ind_stop = stop_line.find('TrialEnd')
                that_stop_index = int(stop_line[:ind_stop])
                
                trial_number.append(that_trial_number)
                start_index.append(that_start_index);
                end_index.append(that_stop_index)
            else: 
                print('Trial End not found for Trial number {}'.format(that_trial_number))
            
    messages = dict([('trial_number',trial_number),\
                         ('start_index',start_index),\
                         ('end_index',end_index)])
                         
    return messages
    
def get_channel_events(loc,events):
    OEMod = import_module("OpenEphys","C:\\Users\\bsriram\\Desktop\\Code\\analysis-tools\\Python3\\OpenEphys.py")
    file = [f for f in os.listdir(loc) if f.startswith('all_channels')]
    if len(file) > 1 or len(file)==0:
        print(loc)
        error('too many or too few trial Records. how is this possible?')
    
    data = OEMod.load(os.path.join(loc,file[0]))
    # filter TTL events
    fil = [True if x==3 else False for x in data['eventType']]
    timestamps = [x for i,x in enumerate(data['timestamps']) if fil[i]==True]
    channel = [x for i,x in enumerate(data['channel']) if fil[i]==True]
    rising_edge = [x for i,x in enumerate(data['eventId']) if fil[i]==True]
    unique_channels = numpy.unique(channel)
    
    ttl_events = {} # across all trials. referenced by trial_number
    for i,tr_num in enumerate(events['trial_number']):
        print(tr_num)
        ttl_event = {} # for that trial. referenced by channel
        
        start_ts = events['start_index'][i]
        end_ts = events['end_index'][i]
        fil = [True if (x>=start_ts and x<=end_ts) else False for x in timestamps]
        that_trial_timestamps = [x for j,x in enumerate(timestamps) if fil[j]==True]
        that_trial_channel = [x for j,x in enumerate(channel) if fil[j]==True]
        that_trial_redge = [x for j,x in enumerate(rising_edge) if fil[j]==True]
        
        # now split by channel
        for chan in unique_channels:
            chan_events = {} # for that chan. references rising or falling
            
            fil = [True if x==chan else False for x in that_trial_channel]
            that_trial_that_chan_timestamps = [x for j,x in enumerate(that_trial_timestamps) if fil[j]==True]
            that_trial_that_chan_redge = [x for j,x in enumerate(that_trial_redge) if fil[j]==True]
            
            # now split by rising edge
            fil = [True if x==1 else False for x in that_trial_that_chan_redge]
            chan_events['rising'] = [x for j,x in enumerate(that_trial_that_chan_timestamps) if fil[j]==True]
            chan_events['falling'] = [x for j,x in enumerate(that_trial_that_chan_timestamps) if fil[j]==False]
            
            ttl_event[chan] = chan_events
        ttl_events[tr_num] = ttl_event
    
    events['ttl_events'] = ttl_events
        
    return events
        
def load_spikerecs_to_dict(loc):
    CDMod = import_module('ClusterDetails','C:\\Users\\bsriram\\Desktop\\Code\\V1PaperAnalysis\\ClusterDetails.py')
    spike_details = CDMod.get_cluster_details(loc)
    return spike_details
    
if __name__=='__main__':
    loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\TEMP\m310_2017-05-16_13-51-53'
    
    with open(os.path.join(loc,'spike_and_trials.pickle'),'rb') as f:
        spike_and_trial_details = pickle.load(f)
    
    pdb.set_trace()