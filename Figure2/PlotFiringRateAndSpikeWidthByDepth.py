import os
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle
from scipy.interpolate import interp1d
from util import get_base_depth,get_unit_depth,get_best_waveform,get_fwhm,get_unit_isi,get_subject_from_session,get_frame_channel,get_unit_id
import scipy.stats

# get_unit_details
def get_unit_details(location):
    # create lists
    unit_id=[]
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
                unit_id.append(get_unit_id(sess,unit['shank_no'],unit['cluster_id']))
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
    
    # goods = numpy.asarray(unit_quality)=='good'
    # muas = numpy.asarray(unit_quality)=='mua'
    
    # fr = numpy.asarray(unit_firingrate)
    # fwhm = numpy.asarray(unit_widthFWHM)
    # depth = numpy.asarray(unit_depth)
    
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
    
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    # plt.clf()
    # plt.plot(fwhm[goods],fr[goods],'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    # plt.show()
    
    # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    # plt.clf()
    # histc_good,binedge = numpy.histogram(fr[goods],range=(0,65),bins=13)
    # histc_mua,binedge = numpy.histogram(fr[muas],range=(0,65),bins=13)
    # plt.bar(binedge[0:13],histc_good,align='edge',color='b',width=4)
    # plt.bar(binedge[0:13],histc_good,align='edge',bottom=histc_good,facecolor='none',edgecolor='b',width=4)
    # plt.show()
    
    unit_details = {}
    unit_details['unit_session'] = unit_session
    unit_details['unit_id'] = unit_id
    unit_details['unit_subject'] = unit_subject 
    unit_details['unit_quality'] = unit_quality
    unit_details['unit_shank'] = unit_shank
    unit_details['unit_clusterid'] = unit_clusterid
    unit_details['unit_depth'] = unit_depth
    unit_details['unit_waveform'] = unit_waveform
    unit_details['unit_widthFWHM'] = unit_widthFWHM
    unit_details['unit_firingrate'] = unit_firingrate
    unit_details['unit_isi'] = unit_isi
    
    return unit_details

def get_is_replicated(unit_details):
    unit_id = unit_details['unit_id']
    unit_subject = unit_details['unit_subject']
    unit_shank = unit_details['unit_shank']
    unit_depth = unit_details['unit_depth']
    unit_waveform = unit_details['unit_waveform']
    unit_isi = unit_details['unit_isi']
    unit_quality = unit_details['unit_quality']
    unit_firingrate = unit_details['unit_firingrate']
    unit_is_replica = []
    unit_is_like = {}
    
    #corr_mat_wvform = numpy.zeros([len(unit_id),len(unit_id)])
    #corr_mat_unitisi = numpy.zeros([len(unit_id),len(unit_id)])
    
    for i in numpy.arange(0,len(unit_id)):
        is_replica = False
        is_like = None
        print(i)
        for j in numpy.arange(0,i):
            # get the correlation in waveform and unitisi
            try:
                corr_waveform = numpy.corrcoef(unit_waveform[i],unit_waveform[j])[0,1]
            except ValueError:
                corr_waveform = 0
            corr_unitisi = numpy.corrcoef(unit_isi[i],unit_isi[j])[0,1]
            abs_amplitude_ratio = numpy.max(numpy.abs(unit_waveform[i]))/numpy.max(numpy.abs(unit_waveform[j]))
            firing_rate_ratio = unit_firingrate[i]/unit_firingrate[j]
            if unit_quality[i]=='good' and unit_subject[i]==unit_subject[j] and unit_shank[i]==unit_shank[j] and numpy.abs(unit_depth[i]-unit_depth[j])<50 and corr_waveform > 0.94 and corr_unitisi >0.94 and (abs_amplitude_ratio<1.333 and abs_amplitude_ratio>0.75):                
                print('sub_id::',unit_subject[i],'shank_id::',unit_shank[i],'corr_waveform::',corr_waveform,'corr_unitisi::',corr_unitisi,'abs_amplitude_ratio::',abs_amplitude_ratio,'firing_rate_ratio::',firing_rate_ratio)
                is_replica = True
                is_like = unit_id[j]
                # fig = plt.figure(figsize=(3,3),dpi=200,facecolor='w',edgecolor='k')
                # ax1 = fig.add_subplot(211)
                # ax1.plot(unit_waveform[i],color='k')
                # ax1.plot(unit_waveform[j],color='r')
                
                # ax2 = fig.add_subplot(212)
                # ax2.plot(unit_isi[i]/numpy.max(unit_isi[i]),color='k')
                # ax2.plot(unit_isi[j]/numpy.max(unit_isi[j]),color='r')
                
                # plt.show()
                break
        if not is_replica:
            unit_is_replica.append(is_replica)
            unit_is_like[unit_id[i]] = None
        else:
            unit_is_replica.append(is_replica)
            if  unit_is_like[is_like] is None:
                unit_is_like[unit_id[i]] = is_like
            else:
                assert unit_is_like[unit_is_like[is_like]] is None, "Weird"
                unit_is_like[unit_id[i]] = unit_is_like[is_like]
    unit_details['unit_is_replica'] = unit_is_replica
    unit_details['unit_is_like'] = unit_is_like
    return unit_details
    
def split_unit_firing_rate_by_depth(location):
    with open(os.path.join(location,'UnitDetailsAfterDereplication_BehavedOnly.pkl'),'rb') as f:
        unit_details = pickle.load(f)
        
    unit_id = unit_details['unit_id']
    unit_subject = unit_details['unit_subject']
    unit_shank = unit_details['unit_shank']
    unit_depth = unit_details['unit_depth']
    unit_waveform = unit_details['unit_waveform']
    unit_isi = unit_details['unit_isi']
    unit_quality = unit_details['unit_quality']
    unit_firingrate = unit_details['unit_firingrate']
    unit_is_replica = unit_details['unit_is_replica']
    unit_widthFWHM = unit_details['unit_widthFWHM']
    
    
    fr = numpy.asarray(unit_firingrate)
    fwhm = numpy.asarray(unit_widthFWHM)
    replica = numpy.asarray(unit_is_replica)
    depth = numpy.asarray(unit_depth)
    goods = numpy.asarray(unit_quality)=='good'
    muas = numpy.asarray(unit_quality)=='mua'
    
    # pdb.set_trace()
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    plt.clf()
    plt.plot(fr[goods & ~replica],-depth[goods & ~replica]+100,'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    plt.plot(fr[muas & ~replica],-depth[muas & ~replica]+100,'o',markerfacecolor='none',markeredgecolor=(0.,0.,1.))
    plt.show()
    
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    plt.clf()
    plt.plot(fwhm[goods & ~replica],-depth[goods & ~replica]+100,'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    plt.plot(fwhm[muas & ~replica],-depth[muas & ~replica]+100,'o',markerfacecolor='none',markeredgecolor=(0.,0.,1.))
    plt.show()
    
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    plt.clf()
    plt.plot(fwhm[goods & ~replica],fr[goods & ~replica],'o',markerfacecolor=(0.,0.,1.,0.5),markeredgecolor='none')
    plt.show()
    
    filter23 = (depth-100)<450
    filter56 = (depth-100)>=450
    
    fr23 = fr[(goods) & ~replica & filter23]
    fr56 = fr[(goods) & ~replica & filter56]
    
    print('Layer2/3::',numpy.mean(fr23),numpy.std(fr23),' n=',len(fr23))
    print('Layer5/6::',numpy.mean(fr56),numpy.std(fr56),' n=',len(fr56))
    
    (h,p) = scipy.stats.ttest_ind(fr23,fr56)
    print('Layer2/3 vs Layer 5/6:: h = ',h,' p = ',p)
    
    (d,p) = scipy.stats.ks_2samp(fr23,fr56)
    print('Layer2/3 vs Layer 5/6:: h = ',d,' p = ',p)
    
    pdb.set_trace()
    


if __name__=="__main__":
    # location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved'
    # unit_details = get_unit_details(location)
    # unit_details = get_is_replicated(unit_details)
    # with open('UnitDetailsAfterDereplication_BehavedOnly.pkl','wb') as f:
        # pickle.dump(unit_details,f)
    saveloc = r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis'
    split_unit_firing_rate_by_depth(saveloc)
    
    
    
    pdb.set_trace()
        
        