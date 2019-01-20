import numpy
import scipy
import pdb
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pprint
ppr = pprint.PrettyPrinter(indent=2)
from datetime import datetime

def get_base_depth(sess):
    if 'bas070_' in sess:
        return 800
    elif 'bas072_' in sess:
        return 1000
    elif 'bas074_' in sess:
        return 500
    elif 'bas077_' in sess:
        return 500
    elif 'bas078_' in sess:
        return 800
    elif 'bas079_' in sess:
        return 500
    elif 'bas080_' in sess:
        if sess=='bas080_2017-06-07_16-36-59':
            return 800
        elif sess=='bas080_2017-06-08_16-39-59':
            return 800
    elif 'bas081a_' in sess:
        if sess=='bas081a_2017-06-23_12-55-28':
            return 1000
        elif sess=='bas081a_2017-06-24_11-57-43':
            return 800
        elif sess=='bas081a_2017-06-24_12-55-51':
            return 500
        elif sess=='bas081a_2017-06-25_12-44-45':
            return 950
    elif 'bas081b_' in sess:
        if sess=='bas081b_2017-07-28_20-31-03':
            return 850
    elif 'm281_' in sess:
        return 500
    elif 'm282_' in sess:
        return 500
    elif 'm284_' in sess:
        return 500
    elif 'm310_' in sess:
        return 500
    elif 'm311_' in sess:
        if sess=='m311_2017-07-31_16-15-43':
            return 800
        elif sess=='m311_2017-07-31_17-08-20':
            return 800
        elif sess=='m311_2017-08-01_12-05-12':
            return 895
        elif sess=='m311_2017-08-01_13-08-00':
            return 650
    elif 'm312_' in sess:
        return 800
    elif 'm317_' in sess:
        if sess=='m317_2017-08-24_15-52-35':
            return 950
        elif sess=='m317_2017-08-24_16-48-34':
            return 950
        elif sess=='m317_2017-08-25_14-21-08':
            return 950
    elif 'm318_' in sess:
        if sess=='m318_2017-08-26_18-14-22':
            return 950
        elif sess=='m318_2017-08-26_19-28-43':
            return 900
    elif 'm325_' in sess:
        if sess=='m325_2017-08-10_13-14-51':
            return 850
        elif sess=='m325_2017-08-10_14-21-22':
            return 950

def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]

def get_session_date(sess):
    splits = sess.split('_')
    date_time = splits[1]+'_'+splits[2]
    return datetime.strptime(date_time,"%Y-%m-%d_%H-%M-%S")
    
def get_unit_depth(sess,y_loc):
    return(get_base_depth(sess)-y_loc)

def get_best_waveform(m_waveform):
    idx = numpy.unravel_index(numpy.argmax(m_waveform),m_waveform.shape)
    return m_waveform[:,idx[1]]
    
def get_fwhm(wvform):
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    
    if abs(numpy.min(wvform))<numpy.max(wvform):
        wvform = -wvform
    
    # plt.clf()
    
    # interpolate for more precise form
    x = numpy.linspace(0,len(wvform)-1,len(wvform))
    total_time = len(wvform)/30. # in ms
    # plt.plot(numpy.linspace(0,total_time,len(wvform)),wvform,'k')
    y = wvform
    interp_wvform_fn = interp1d(x,y,kind='cubic')
    
    x_new = numpy.linspace(0,len(wvform)-1,1000)
    y_new = interp_wvform_fn(x_new)
    t_new = numpy.linspace(0,total_time,1000)
    # plt.plot(t_new,y_new,'k')
   
    # find the min value and index of interp functionm
    wvform_min = numpy.min(y_new)
    half_min = wvform_min/2
    wvform_min_idx = numpy.argmin(y_new)
    
    idx_left = 0
    for i in range(wvform_min_idx,-1,-1):
        if y_new[i]>half_min:
           idx_left = i
           break
    
    idx_right = 999
    for i in range(wvform_min_idx,999):
        if y_new[i]>half_min:
            idx_right = i
            break
    
    # plt.plot(t_new[idx_left],y_new[idx_left],'rx')    
    # plt.plot(t_new[idx_right],y_new[idx_right],'rx')
    # plt.plot([t_new[idx_left], t_new[idx_right]],[y_new[idx_left], y_new[idx_right]],'r')    
    
    spike_width = (idx_right-idx_left)/1000.*total_time
    # plt.show()
    return spike_width
    
def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]
    
def get_orientation_tuning_stepname(sess):
    subj = get_subject_from_session(sess)
    if subj in ['bas072','bas074','m284']:
        return 'gratings',500
    elif subj in ['bas077','bas078','bas079','bas080','bas081a']:
        return 'gratings',2000
    elif subj in ['bas070']:
        if sess in ['bas070_2015-08-05_12-31-50','bas070_2015-08-06_12-09-26','bas070_2015-08-11_12-51-37','bas070_2015-08-13_11-56-14_1']:
            return None, None
        else:
            return 'gratings',500
    elif subj in ['bas081b']:
        return 'LongDurationOR_LED',2000
    elif subj in ['m317','m311','m318','m325']:
        return 'OrSweep',2000
 
def get_short_response_stepname(sess):
    subj = get_subject_from_session(sess)
    if subj in ['bas072','bas074','m284']:
        return ['gratings_LED'],[[3,6,12]], [[0,0.15,1.]]
    elif subj in ['bas077','bas078','bas079','bas080','bas081a']:
        return  ['gratings_NOLED','gratings_baseLine'],[[1,3,6,12,30],[1,3,6,12,30]],[[0.15,1],[0]]
    elif subj in ['bas070']:
        if sess in ['bas070_2015-08-05_12-31-50','bas070_2015-08-06_12-09-26','bas070_2015-08-11_12-51-37']:
            return ['gratings'], [[1,2,3,6,12]], [[0.15,1.]]
        elif sess in ['bas070_2015-09-22_12-30-10','bas070_2015-09-23_15-23-36','bas070_2015-09-24_15-47-24','bas070_2015-09-25_14-18-57','bas070_2015-09-28_12-24-33','bas070_2015-09-29_14-51-50','bas070_2015-09-30_13-22-42','bas070_2015-10-01_11-27-45','bas070_2015-10-02_11-46-50']:
            return ['gratings_LED'],[[3,6,12]], [[0,0.15,1.]]
        else:
            return ['gratings_LED'],[[1,2,3,6,12]], [[0.15,1.]]
    elif subj in ['bas081b']:
        return ['ShortDurationOR'],[[3,6,9,12,30]],[[0.15,1]]
    elif subj in ['m317','m311','m318','m325']:
        return ['ShortDurationOR'],[[3,6,9,12,30]],[[0.15,1]]
  
def get_frame_channel(sess):
    if sess in ['bas080_2017-06-07_16-36-59','bas080_2017-06-08_16-39-59','bas081a_2017-06-23_12-55-28','bas081a_2017-06-24_11-57-43','bas081a_2017-06-24_12-55-51','bas081a_2017-06-25_12-44-45','bas081b_2017-07-28_20-31-03','m311_2017-07-31_16-15-43','m311_2017-07-31_17-08-20','m311_2017-08-01_12-05-12','m311_2017-08-01_13-08-00','m325_2017-08-10_13-14-51','m325_2017-08-10_14-21-22','m317_2017-08-24_15-52-35',
    'm317_2017-08-24_16-48-34','m317_2017-08-25_14-21-08','m318_2017-08-26_18-14-22','m318_2017-08-26_19-28-43']:
        return 0,0
    else:
        return 1,2
        
def get_LED_channel(sess):
    if sess in ['bas081b_2017-07-28_20-31-03','m311_2017-07-31_16-15-43','m311_2017-07-31_17-08-20','m311_2017-08-01_12-05-12','m311_2017-08-01_13-08-00','m325_2017-08-10_13-14-51','m325_2017-08-10_14-21-22']:
        return 3
    else:
        return None
        
def get_OSI(orn,m):
    m = numpy.asarray(m)
    orn = numpy.asarray(orn)
    
    assert numpy.size(m)==numpy.size(orn)
    
    # ensure ascending orientation
    order = numpy.argsort(orn)
    orn = orn[order]
    m = m[order]
    
    # get or with highest mean firing rate
    which_max = numpy.argmax(m)
    or_max = orn[which_max]
    if (which_max>(numpy.size(m)/2)):
        or_min = or_max-(numpy.pi/2)
    else:
        or_min = or_max+(numpy.pi/2)
    which_min = numpy.argmin(numpy.abs((orn-or_min)))
    
    m_max = m[which_max]
    m_min = m[which_min]
    numpy.seterr(all='raise')
    try:
        osi = ((m_max-m_min)/(m_max+m_min))
    except:
        osi = None
    return osi,or_max
    
def get_vector_sum(orn,m):
    m = numpy.asarray(m)
    orn = numpy.asarray(orn)
    orn = 2*(-orn + (numpy.pi/2)) 
    
    assert numpy.size(m)==numpy.size(orn)
    
    complex_sum = numpy.dot(m,numpy.exp(1j*orn))    
    strength = numpy.abs(complex_sum)
    angle = numpy.angle(complex_sum)
    if angle<0:
        angle = 2*numpy.pi+angle
    angle = numpy.pi/2-angle/2
    
    if angle > numpy.pi/2:
        angle = numpy.pi-angle
    elif angle<-numpy.pi/2:
        angle = numpy.pi+angle
    return strength,angle
  
def get_unit_id(sess,shank,cluster):
    return '{0}_s{1}_c{2}'.format(sess,shank,cluster)

def get_LED_stepname(sess):
    if sess in ['bas081b_2017-07-28_20-31-03','m311_2017-07-31_16-15-43','m311_2017-07-31_17-08-20','m311_2017-08-01_12-05-12','m311_2017-08-01_13-08-00','m325_2017-08-10_13-14-51','m325_2017-08-10_14-21-22']:
        return 'LongDurationOR_LED',2000
    else:
        return None, None
        
def get_unit_dist_to_LED(sess,y_loc):
    return(500-y_loc)
    
def raster(event_times_list, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax
    
if __name__=='__main__':
    orn = numpy.linspace(-numpy.pi/2,numpy.pi/2,5)
    m = numpy.asarray([0,0,1,1,1])
    str,ang = get_vector_sum(orn,m)
    