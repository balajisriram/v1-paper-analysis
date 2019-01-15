import types
import pickle
import os
import pdb
import numpy
from util import get_LED_stepname,get_unit_id,get_frame_channel,get_subject_from_session,get_unit_dist_to_LED,get_unit_depth,raster,get_LED_channel
import matplotlib.pyplot as plt


def fr_for_units_by_led(location):
    base,sess = os.path.split(location)
    with open(os.path.join(location,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
    stepname,durations = get_LED_stepname(sess)
    framechan,shift = get_frame_channel(sess)
    LEDchan = get_LED_channel(sess)
    
    trial_numbers = numpy.asarray(data['trial_records']['trial_number'])
    step_names = numpy.asarray(data['trial_records']['step_name'])
    contrasts = numpy.asarray(data['trial_records']['contrast'])
    max_durations = numpy.asarray(data['trial_records']['max_duration'])
    phases = numpy.asarray(data['trial_records']['phase'])
    orientations = numpy.asarray(data['trial_records']['orientation'])
    led_status = numpy.multiply(numpy.asarray(data['trial_records']['led_on']),numpy.nan_to_num(numpy.asarray(data['trial_records']['led_intensity'])))
    
    which_step = step_names==stepname
    trial_numbers = trial_numbers[which_step]
    step_names = step_names[which_step]
    contrasts = contrasts[which_step]
    max_durations = max_durations[which_step]
    phases = phases[which_step]
    orientations = orientations[which_step]
    led_status = led_status[which_step]
    frame_start_index = []
    frame_end_index = []
    for trial_number in trial_numbers:
        event_for_trial = data['trial_records']['events']['ttl_events'][trial_number]
        frame_start_index.append(event_for_trial[LEDchan]['rising'][0])
        frame_end_index.append(event_for_trial[LEDchan]['falling'][0])
            
    frame_start_index = numpy.asarray(frame_start_index)
    frame_end_index = numpy.asarray(frame_end_index)
    
    unit_responses_that_session = {}
    unit_responses_that_session['session'] = sess
    unit_responses_that_session['subject'] = get_subject_from_session(sess)
    
    for i,unit in enumerate(data['spike_records']['units']):
        if not unit['manual_quality'] in ['good','mua']: continue
        unit_details = {}
        unit_details['uid'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
        unit_details['manual_quality'] = unit['manual_quality']
        unit_details['unit_depth'] = get_unit_depth(sess,unit['y_loc'])
        unit_details['unit_depth'] = get_unit_dist_to_LED(sess,unit['y_loc'])
        spike_time = numpy.squeeze(numpy.asarray(unit['spike_time']))
        
        spike_raster = {}
        spike_raster_expanded = {} # t-100ms to t+500 ms
        for trial_number in trial_numbers:
            frame_start_time = frame_start_index[trial_numbers==trial_number][0]/30000
            frame_end_time = frame_end_index[trial_numbers==trial_number][0]/30000
            spike_raster[trial_number] = spike_time[numpy.bitwise_and(spike_time>frame_start_time,spike_time<frame_end_time)]-frame_start_time
            spike_raster_expanded[trial_number] = spike_time[numpy.bitwise_and(spike_time>(frame_start_time-0.5),spike_time<(frame_end_time+1))]-frame_start_time
        unit_details['spike_raster'] = spike_raster
        unit_details['spike_raster_expanded'] = spike_raster_expanded
        
        # plot by LED
        LED_OFFs = led_status==0
        trial_led_off = trial_numbers[LED_OFFs]
        trial_led_on = trial_numbers[numpy.bitwise_not(LED_OFFs)]
        
        events_led_off = []
        events_led_on = []
        for tr in trial_led_off:
            events_led_off.append(spike_raster_expanded[tr])
        for tr in trial_led_on:
            events_led_on.append(spike_raster_expanded[tr])
        
        # pdb.set_trace()
        # isi = numpy.diff(unit['spike_time'],axis=0)
        # histc, binedge = numpy.histogram(isi,range=(0,0.1),bins=100)
        # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
        # plt.clf()
        # plt.bar(binedge[0:100]*1000,histc,align='edge',edgecolor='none',color='k')
        # plt.show()
        
        
        fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
        plt.clf()
        ax_off = plt.subplot(211)
        raster(events_led_off)
        ax_on = plt.subplot(212)
        raster(events_led_on,color='b')
        plt.show()
        
        
        
        print('done_unit')
        unit_responses_that_session[i] = unit_details
    
    print('done session')
    
    
    
if __name__=='__main__':
    location = r'C:\Users\bsriram\Desktop\Data_V1Paper\SessionsWithLED'
    for sess in os.listdir(location):
        out = fr_for_units_by_led(os.path.join(location,sess))