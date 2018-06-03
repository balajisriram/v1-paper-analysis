import os
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle
from scipy.interpolate import interp1d

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

def get_unit_isi(spike_times):
    isi = numpy.diff(numpy.unique(spike_times),axis=0)
    histc, binedge = numpy.histogram(isi,range=(0,0.1),bins=100)
    return histc    
    
if __name__=="__main__":
    location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly'
    
    # create lists
    unit_session = []
    unit_subject = []
    unit_quality = []
    unit_shank = []
    unit_clusterid = []
    unit_depth = []
    unit_waveform = []
    unit_widthFWHM = []
    unit_firingrate = []
    unit_isi  = []
    for sess in os.listdir(location): 
        print(sess)
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        session_duration = data['spike_records']['duration']
        for unit in data['spike_records']['units']:
            if unit['manual_quality'] in ['good','mua']:
                unit_session.append(sess)
                unit_subject.append(get_subject_from_session(sess))
                unit_quality.append(unit['manual_quality'])
                unit_shank.append(unit['shank_no'])
                unit_clusterid.append(unit['cluster_id'])
                unit_depth.append(get_unit_depth(sess,unit['y_loc']))
                unit_waveform.append(get_best_waveform(unit['mean_waveform']))
                unit_widthFWHM.append(get_fwhm(get_best_waveform(unit['mean_waveform'])))
                unit_firingrate.append(unit['num_spikes']/session_duration)
                unit_isi.append(get_unit_isi(unit['spike_time']))
            elif unit['manual_quality'] in ['unsorted']:
                pass
    
    goods = numpy.asarray(unit_quality)=='good'
    muas = numpy.asarray(unit_quality)=='mua'
    
    fr = numpy.asarray(unit_firingrate)
    fwhm = numpy.asarray(unit_widthFWHM)
    depth = numpy.asarray(unit_depth)
    
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    # plt.clf()
    # plt.plot(fr[goods],-depth[goods]+100,'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    # plt.plot(fr[muas],-depth[muas]+100,'o',markerfacecolor='none',markeredgecolor=(0.,0.,1.))
    # plt.show()
    
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    # plt.clf()
    # plt.plot(fwhm[goods],-depth[goods]+100,'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    # plt.plot(fwhm[muas],-depth[muas]+100,'o',markerfacecolor='none',markeredgecolor=(0.,0.,1.))
    # plt.show()
    
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    plt.clf()
    plt.plot(fwhm[goods],fr[goods],'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    plt.show()
    
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    # plt.clf()
    # histc_good,binedge = numpy.histogram(fr[goods],range=(0,65),bins=13)
    # histc_mua,binedge = numpy.histogram(fr[muas],range=(0,65),bins=13)
    # plt.bar(binedge[0:13],histc_good,align='edge',color='b',width=4)
    # plt.bar(binedge[0:13],histc_good,align='edge',bottom=histc_good,facecolor='none',edgecolor='b',width=4)
    # plt.show()
    
    pdb.set_trace()
        
        