import pprint
ppr = pprint.PrettyPrinter(indent=2)
from datetime import datetime
from klusta.kwik.model import KwikModel
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pdb
import pickle
import pandas as pd
from scipy.stats import kde
from ClusterQuality import cluster_quality_core,cluster_quality_all
from tqdm import tqdm


def get_base_depth(sess,type='OE'):
    if type=='OE': # these values are from surgical records
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
    elif type=='plexon': # I got these values from my notebook
        depth_by_session = dict(
        {
        'b8_01312019':300,
        'b8_02022019':320,
        'b8_02032019':340,
        'b8_02042019':360,
        'b8_02152019':380,
        'b8_02172019':400,
        'b8_02182019':420,
        'b8_02192019':440,
        'b8_02212019':460,
        'b8_02222019':480,
        'b8_02232019':500,
        'b8_02252019':500,
        'g2_2_01312019':330,
        'g2_2_02012019':335,
        'g2_2_02022019':340,
        'g2_2_02042019':362.5,
        'g2_2_02052019':380,
        'g2_2_02062019':402.5,
        'g2_2_02132019':412.5,
        'g2_2_02142019':422.5,
        'g2_2_02152019':442.5,
        'g2_2_02182019':450,
        'g2_2_02192019':458,
        'g2_2_02212019':487.5,
        'g2_2_02222019':495,
        'g2_2_02232019':500,
        'g2_2_02252019':532.5,
        'g2_2_02272019':570.5,
        'g2_2_02282019':585,
        'g2_4_02152019':300,
        'g2_4_02182019':375,
        'g2_4_02212019':400,
        'g2_4_02222019':562.5,
        'g2_4_02232019':525,
        'g2_4_02252019':563.8,
        'g2_4_02272019':560,
        'g2_4_02282019':600,
        'g2_5_02152019':350,
        }
        )
        return depth_by_session[sess]

def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]

def get_session_date(sess,type='OE'):
    splits = sess.split('_')
    if type=='OE':
        date_time = splits[1]+'_'+splits[2]
        return datetime.strptime(date_time,"%Y-%m-%d_%H-%M-%S")
    elif type=='plexon':
        date_time - splits[1]
        return datetime.strptime(date_time,"%m%d%Y")
    
def get_unit_depth(sess,y_loc):
    return(get_base_depth(sess)-y_loc)

def get_best_waveform(m_waveform):
    idx = np.unravel_index(np.argmax(m_waveform),m_waveform.shape)
    return m_waveform[:,idx[1]]

def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]
    
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
    
def get_model(loc,session_type='OE'):
    if session_type=='OE':
        kwik_file = [f for f in os.listdir(loc) if f.endswith('.kwik') and '100' in f]
    else:
        kwik_file = [f for f in os.listdir(loc) if f.endswith('.kwik')]
    
    if len(kwik_file)>1 or len(kwik_file)==0:
        RuntimeError('Too many or too few files. Whats happening')
    kwik_file_path = os.path.join(loc,kwik_file[0])
    # print("loading kwik file :", kwik_file_path)
    kwik_model = KwikModel(kwik_file_path)
    kwik_model._open_kwik_if_needed()
    return kwik_model
        
def get_unit_details(folder):
    
    def get_cluster_waveform_all (kwik_model,cluster_id): 
        clusters = kwik_model.spike_clusters
        try:
            if not cluster_id in clusters:
                raise ValueError       
        except ValueError:
                print ("Exception: cluster_id (%d) not found !! " % cluster_id)
                return
        
        idx=np.argwhere(clusters==cluster_id)
        return kwik_model.all_waveforms[idx]
    
    # get the .kwik file and make KwikModel
    kwik_model = get_model(folder)
    
    # all the stuff i want to extract
    output = {}
    output["n_samples_waveforms"] = kwik_model.n_samples_waveforms
    output["duration"] = kwik_model.duration
    output["num_channels"] = kwik_model.n_channels
    output["strong_threshold"] = kwik_model.metadata["threshold_strong_std_factor"]
    output["weak_threshold"] = kwik_model.metadata["threshold_weak_std_factor"]
    output["sample_rate"] = kwik_model.metadata["sample_rate"]
    output["high_pass_lo_f"] = kwik_model.metadata["filter_low"]
    output["high_pass_hi_f"] = kwik_model.metadata["filter_high_factor"]*output["sample_rate"]
    output["probe_name"] = kwik_model.metadata["prb_file"]
    output["data_type"] = kwik_model.metadata["dtype"]
    output["spikes_direction"] = kwik_model.metadata["detect_spikes"]
    output["klustakwik2_version"] = kwik_model.clustering_metadata["klustakwik2_version"]
    
    ch_groups = kwik_model.channel_groups
    num_channel_groups = len(ch_groups)
    
    units = []
    
    for j,ch_grp in enumerate(ch_groups):
        print('Shank ', j+1,' of ',len(ch_groups))
        
        kwik_model.channel_group = ch_grp # go to that channel group
        kwik_model.clustering = 'main'
        
        cluster_quality = kwik_model.cluster_groups
        spike_time = kwik_model.spike_samples.astype(np.float64)/kwik_model.sample_rate
        spike_id = kwik_model.spike_ids
        cluster_id = kwik_model.spike_clusters
        cluster_ids = kwik_model.cluster_ids
        
        clu = cluster_id
        fet_masks = kwik_model.all_features_masks
        fet = np.squeeze(fet_masks[:,:,0])
        mask = np.squeeze(fet_masks[:,:,1])
        fet_N = 12
        unique_clu, clu_quality, cont_rate = CQMod.cluster_quality_all(clu, fet, mask, fet_N)
        print("Finished getting quality for shank")
        
        for clust_num in cluster_ids:
            unit = {}
            unit["shank_no"] = ch_grp
            unit["cluster_id"] = clust_num
            unit["channels_in_shank"] = kwik_model.channels
            unit["manual_quality"] = cluster_quality[clust_num]
            # find the spike_ids corresponding to that cluster id
            spike_id_idx = np.argwhere(cluster_id==clust_num)
            spike_id_that_cluster = spike_id[spike_id_idx]
            unit["spike_ids"] = spike_id_that_cluster
            unit["spike_time"] = spike_time[spike_id_idx]
            
            
            waves = get_cluster_waveform_all(kwik_model,clust_num)
            mu_all = np.mean(waves[:,:,:],axis=0);
            std_all = np.std(waves[:,:,:],axis=0);
            unit["mean_waveform"] = np.mean(waves,axis=0)
            unit["std_waveform"] = np.std(waves,axis=0)
            
            unit["num_spikes"] = waves.shape[0]
            
            max_ind = np.unravel_index(np.argmin(mu_all),mu_all.shape)
            unit["loc_idx"] = max_ind
            max_ch = max_ind[1]
            
            unit["x_loc"] = kwik_model.channel_positions[max_ch,0]
            unit["y_loc"] = kwik_model.channel_positions[max_ch,1]
            unit["quality"] = clu_quality[np.argwhere(unique_clu==clust_num)]
            unit["contamination_rate"] =cont_rate[np.argwhere(unique_clu==clust_num)]            
            
            units.append(unit)
    output["units"] = units
    
    kwik_model.close()
    return output

def plot_ISI(unit, ax, record):
    spike_train = unit['spike_time']
    n_spikes = np.size(spike_train)
    isi = np.diff(spike_train, axis=0)
    histc, binedge = np.histogram(isi,range=(0,0.025),bins=100)
    
    #ax.bar(binedge[0:100]*1000,histc,align='edge',edgecolor='none',color='b')
    #ax.plot([1,1],ax.get_ylim(),'r')
    #ax.plot([2,2],ax.get_ylim(),'r--')
    
    
    num_violations = np.sum(isi<0.001)
    total_time = spike_train[-1]
    violation_rate = num_violations/total_time
    total_rate = n_spikes/total_time
    record['isi_violation_rate'] = violation_rate
    record['spike_rate'] = total_rate
    #ax.text(ax.get_xlim()[1],ax.get_ylim()[1],'viol_rate=%2.3fHz of %2.3fHz' % (violation_rate,total_rate), horizontalalignment='right',verticalalignment='top',fontsize=6)
    return record

def plot_unit_stability(unit, model_loc, record, ax=None):
    kwik_model = get_model(model_loc)
    
    kwik_model.channel_group = unit['shank_no']
    kwik_model.clustering = 'main'
    spike_time = kwik_model.spike_samples.astype(np.float64)/kwik_model.sample_rate
    spike_id = kwik_model.spike_ids
    cluster_id = kwik_model.spike_clusters
    fet_masks = kwik_model.all_features_masks
    fet = np.squeeze(fet_masks[:,:,0])
    
    that_cluster_idx = np.argwhere(cluster_id==unit['cluster_id'])
    other_cluster_idx = np.argwhere(cluster_id!=unit['cluster_id'])
    
    spike_time_that = spike_time[that_cluster_idx]
    spike_time_other = spike_time[other_cluster_idx]
    
    fet_that = np.squeeze(fet[that_cluster_idx,:])
    fet_other = np.squeeze(fet[other_cluster_idx,:])
    
    # find the feature dimensions with greatest mean values
    fet_that_mean = np.mean(fet_that,axis=0)
    fet_seq = np.argsort(-1*np.absolute(fet_that_mean))
    
    fet_x_that = fet_that[:,fet_seq[0]]
    fet_y_that = fet_that[:,fet_seq[2]]
    
    fet_x_other = fet_other[:,fet_seq[0]]
    fet_y_other = fet_other[:,fet_seq[2]]
    
    # select 2% from 'that' and 0.1% from other
    that_choice = np.random.choice([True,False],size=fet_x_that.shape,p=[0.02,0.98])
    other_choice = np.random.choice([True,False],size=fet_x_other.shape,p=[0.01,0.99])
    if ax:
        pass
        # ax.plot(spike_time_that[that_choice],fet_x_that[that_choice],'o',color='blue',markersize=5,markeredgecolor='none')
        # ax.plot(spike_time_other[other_choice],fet_x_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')

    return record
    
def plot_unit_quality(unit, model_loc, record, ax=None):
    kwik_model = get_model(model_loc)
    
    kwik_model.channel_group = unit['shank_no']
    kwik_model.clustering = 'main'
    spike_time = kwik_model.spike_samples.astype(np.float64)/kwik_model.sample_rate
    spike_id = kwik_model.spike_ids
    cluster_id = kwik_model.spike_clusters
    fet_masks = kwik_model.all_features_masks
    fet = np.squeeze(fet_masks[:,:,0])
    
    that_cluster_idx = np.argwhere(cluster_id==unit['cluster_id'])
    other_cluster_idx = np.argwhere(cluster_id!=unit['cluster_id'])
    
    fet_that = np.squeeze(fet[that_cluster_idx,:])
    fet_other = np.squeeze(fet[other_cluster_idx,:])
    
    # find the feature dimensions with greatest mean values
    # fet_that_mean = np.mean(fet_that,axis=0)
    # fet_seq = np.argsort(-1*np.absolute(fet_that_mean))
    
    # fet_x_that = fet_that[:,fet_seq[0]]
    # fet_y_that = fet_that[:,fet_seq[1]]
    # pdf_that,axes_that = fastKDE.pdf(fet_x_that,fet_y_that)
    # data_that = np.vstack((fet_x_that, fet_y_that))
    # data_that = data_that.T
    
    # fet_x_other = fet_other[:,fet_seq[0]]
    # fet_y_other = fet_other[:,fet_seq[1]]
    # pdf_other,axes_other = fastKDE.pdf(fet_x_other,fet_y_other)
    # data_other = np.vstack((fet_x_other, fet_y_other))
    # data_other = data_other.T
    
    # min_x = np.min([np.min(fet_x_that),np.min(fet_x_other)])
    # max_x = np.max([np.max(fet_x_that),np.max(fet_x_other)])
    # x = np.linspace(min_x,max_x,50)
    # min_y = np.min([np.min(fet_y_that),np.min(fet_y_other)])
    # max_y = np.max([np.max(fet_y_that),np.max (fet_y_other)])
    # y = np.linspace(min_y,max_y,50)
    
    # xx, yy = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
    
    # k_that = kde.gaussian_kde(data_that.T)
    # z_that = k_that(np.vstack([xx.flatten(), yy.flatten()]))

    # k_other = kde.gaussian_kde(data_other.T)
    # z_other = k_other(np.vstack([xx.flatten(), yy.flatten()]))
    
    uq,cr = cluster_quality_core(fet_that,fet_other)
    record['isolation_distance'] = uq
    record['contamination_rate_from_mahal'] = cr
    
    if ax:
        # ax.hist2d(fet_x_that,fet_y_that,bins=30,cmap='Blues',alpha=0.5)
        # ax.contour(axes_that[0], axes_that[1], pdf_that,cmap='Blues',linewidths=[1,1,2,2,3,3])
        
        # ax.hist2d(fet_x_other,fet_y_other,bins=30,cmap='Greys',alpha=0.5)
        # ax.contour(axes_other[0], axes_other[1], pdf_other,cmap='Greys',linewidths=[1,1,2,2,3,3])
        if cr is not None:
            pass
            # ax.text(ax.get_xlim()[1],ax.get_ylim()[1]-100,'uq=%2.3f;cr=%2.3f' %(uq,cr), horizontalalignment='right',verticalalignment='top',fontsize=6)
        else:
            pass
            # ax.text(ax.get_xlim()[1],ax.get_ylim()[1]-100,'uq={0};cr={1}'.format(uq,cr), horizontalalignment='right',verticalalignment='top',fontsize=6)
    
    return record

def plot_firing_rate(unit, loc, record, ax=None):
    with open(os.path.join(loc,'spike_and_trials.pickle'),'rb') as f:
        data = pickle.load(f)
    session_duration = data['spike_records']['duration']
    spike_time = unit['spike_time']
    times = []
    spike_rate = []
    for t in np.arange(np.floor(session_duration)):
        spike_rate.append(np.sum((spike_time>t) & (spike_time<t+1)))
        times.append(t)
    
    if ax:
        pass
        # ax.plot(times,spike_rate,'k')
    return record
        
def plot_or_tuning(unit, loc, record, ax=None):
    tuning_data,failed_why = get_or_tuning(os.path.dirname(loc), os.path.basename(loc),unit)
    if not failed_why:
        record['or_tuning_orientation'] = tuning_data['orientation']
        record['or_tuning_rate'] = tuning_data['m_rate']
        record['or_tuning_sd'] = tuning_data['std_rate']
        record['or_tuning_n_trials'] = tuning_data['n_trials']
        record['or_tuning_sem'] = np.divide(tuning_data['std_rate'],np.sqrt(tuning_data['n_trials']))
        
        record['osi'] = tuning_data['osi'][0]
        record['osi_angle'] = tuning_data['osi'][1]
        record['osi_jk_m'] = np.mean(np.array([x[0] for x in tuning_data['osi_jackknife']]))
        record['osi_jk_sd'] = np.std(np.array([x[0] for x in tuning_data['osi_jackknife']]))
        record['osi_jk_n'] = np.size(np.array([x[0] for x in tuning_data['osi_jackknife']]))
        record['osi_angle_jk_m'] = np.mean(np.array([x[1] for x in tuning_data['osi_jackknife']]))
        record['osi_angle_jk_sd'] = np.std(np.array([x[1] for x in tuning_data['osi_jackknife']]))
        record['osi_angle_jk_n'] = np.size(np.array([x[1] for x in tuning_data['osi_jackknife']]))
        
        record['vecsum'] = tuning_data['vector_sum'][0]
        record['vecsum_angle'] = tuning_data['vector_sum'][1]
        record['vecsum_jk_m'] = np.mean(np.array([x[0] for x in tuning_data['vector_sum_jackknife']]))
        record['vecsum_jk_sd'] = np.std(np.array([x[0] for x in tuning_data['vector_sum_jackknife']]))
        record['vecsum_jk_n'] = np.size(np.array([x[0] for x in tuning_data['vector_sum_jackknife']]))
        record['vecsum_angle_jk_m'] = np.mean(np.array([x[1] for x in tuning_data['vector_sum_jackknife']]))
        record['vecsum_angle_jk_sd'] = np.std(np.array([x[1] for x in tuning_data['vector_sum_jackknife']]))
        record['vecsum_angle_jk_n'] = np.size(np.array([x[1] for x in tuning_data['vector_sum_jackknife']]))
        
        # if ax:
            # ax.plot(np.pi/2-ori,m_rate,color='k')
            # ax.errorbar(np.pi/2-ori,m_rate,yerr=sem,capsize=0,color='k')

            # vec_sum_ang = tuning_data['vector_sum'][1]
            # vec_sum_val = tuning_data['vector_sum'][0]

            # ax.plot([np.pi/2-tuning_data['osi'][1],np.pi/2-tuning_data['osi'][1]],[0,vec_sum_val],'r',linewidth=5,alpha=0.5)

            # for jk_sam in tuning_data['vector_sum_jackknife']:
                # plt.plot([np.pi/2-jk_sam[1],np.pi/2-jk_sam[1]],[0,jk_sam[0]],'k',linewidth=2.5,alpha=0.1)
            # ax.plot([np.pi/2-vec_sum_ang,np.pi/2-vec_sum_ang],[0,vec_sum_val],'g',linewidth=5,alpha=0.5)
            # ax.text(ax.get_xlim()[1],ax.get_ylim()[0],'fwhm=%2.3f\np2tT=%2.3f\np2tR=%2.3f\nsnr=%2.3f'%(fwhm,p2t_time,p2t_ratio,record['peak_snr_wvform']),
                    # horizontalalignment='right',verticalalignment='bottom',fontsize=5)
    else:
        record['or_tuning_orientation'] = None
        record['or_tuning_rate'] = None
        record['or_tuning_sd'] = None
        record['or_tuning_n_trials'] = None
        record['or_tuning_sem'] = None
        
        record['osi'] = None
        record['osi_angle'] = None
        record['osi_jk_m'] = None
        record['osi_jk_sd'] = None
        record['osi_jk_n'] = None
        record['osi_angle_jk_m'] = None
        record['osi_angle_jk_sd'] = None
        record['osi_angle_jk_n'] = None
        
        record['vecsum'] = None
        record['vecsum_angle'] = None
        record['vecsum_jk_m'] = None
        record['vecsum_jk_sd'] = None
        record['vecsum_jk_n'] = None
        record['vecsum_angle_jk_m'] = None
        record['vecsum_angle_jk_sd'] = None
        record['vecsum_angle_jk_n'] = None
        # if ax:
            # ax.clear()
            # ax.grid(False)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.text(0,0,failed_why,fontsize=6,horizontalalignment='center',verticalalignment='center')
    return record

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

## STUFF
def get_features(what,base_loc =r'C:\Users\bsriram\Desktop\Data_V1Paper'):
    out = []
    sub_locs = ['DetailsProcessedPhysOnly','DetailsProcessedBehaved','EyeTracked']
    sub_locs = ['EyeTracked']
    types = ['OE','OE','plexon']
    types = ['plexon']
    for type,sub_loc in zip(types,sub_locs):
        for sess in tqdm(os.listdir(os.path.join(base_loc,sub_loc))):
            spike_and_trial_file = [f for f in os.listdir(os.path.join(base_loc,sub_loc,sess)) if f.startswith('spike_and_trials')]
            assert len(spike_and_trial_file)==1, 'too many or too few spike and trials pickle'
            spike_and_trial_file = spike_and_trial_file[0]
            with open(os.path.join(base_loc,sub_loc,sess,spike_and_trial_file),'rb') as f:
                sess_data = pickle.load(f)
            for unit in sess_data['spike_records']['units']:
                if unit['manual_quality'] in ['good','mua']:
                    if what=='waveform_details':
                        interp_t,interp_m,interp_sd,peak_snr,fwhm,p2t_ratio,p2t_time = get_waveform_details(unit,type=type)
                        uid = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp = {}
                        temp['unit_id'] = uid
                        temp[('waveform','mu')] = interp_m
                        temp[('waveform','sd')] = interp_sd
                        temp[('waveform','snr')] = peak_snr
                        temp[('waveform','fwhm')] = fwhm
                        temp[('waveform','p2t_ratio')] = p2t_ratio
                        temp[('waveform','p2t_time')] = p2t_time
                        out.append(temp)
                    elif what=='firing_rates':
                        temp = {}
                        sess_duration = sess_data['spike_records']['duration']
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp['firing_rate'] = unit['num_spikes']/sess_duration
                        out.append(temp)
                    elif what=='unit_quality':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        uq,cr,why_failed = get_unit_quality(unit,os.path.join(base_loc,sub_loc,sess),session_type=type)
                        temp['unit_quality'] = uq
                        temp['contamination_rate'] = cr
                        temp['failure_reason'] = why_failed
                        out.append(temp)
    return out
    
## waveforms
def get_waveform_details(unit,type='OE'):
    def get_peak_trough(wvform, t):
        # find the min value and index of interp functionm
        wvform_min = np.min(wvform)
        half_min = wvform_min/2
        wvform_min_idx = np.argmin(wvform)
        wvform_max_idx = np.argmax(wvform[wvform_min_idx:])+ wvform_min_idx
        
        p2t_ratio = -wvform[wvform_max_idx]/wvform[wvform_min_idx]
        p2t_time = t[wvform_max_idx]-t[wvform_min_idx]
        
        return p2t_ratio,p2t_time
        
    def get_fwhm(wvform,t_wvform):
        # invert if positive waveform
        if abs(np.min(wvform))<np.max(wvform):
            wvform = -wvform

        wvform_min = np.min(wvform)
        half_min = wvform_min/2
        wvform_min_idx = np.argmin(wvform)
        
        idx_left = 0
        for i in range(wvform_min_idx,-1,-1):
            if wvform[i]>half_min:
               idx_left = i
               break
        idx_right = len(t_wvform)-1
        for i in range(wvform_min_idx,len(t_wvform)-1):
            if wvform[i]>half_min:
                idx_right = i
                break
        spike_width = t_wvform[idx_right]-t_wvform[idx_left]
        return spike_width
    
    mu_all = unit['mean_waveform']
    std_all = unit['std_waveform']
    waveform_size = mu_all.shape[0]
    num_chans = mu_all.shape[1]
    
    # get the largest deviation in the negative direction
    max_ind = np.unravel_index(np.argmin(mu_all),mu_all.shape)
    max_chan = max_ind[1]
    
    mu_max = mu_all[:,max_chan]
    sd_max = std_all[:,max_chan]
    
    if type=='OE':sample_rate=30000.
    else: sample_rate=40000.
    
    interp_fn_m = interpolate.interp1d(np.linspace(0,(waveform_size-1)/sample_rate,waveform_size),mu_max,kind='cubic')
    interp_fn_sd = interpolate.interp1d(np.linspace(0,(waveform_size-1)/sample_rate,waveform_size),sd_max,kind='cubic')
    
    interp_t = np.linspace(0,(waveform_size-1)/sample_rate,waveform_size*125)
    interp_m = interp_fn_m(interp_t)
    interp_sd = interp_fn_sd(interp_t)
    
    max_idx = np.argmin(interp_m)
    peak_snr_wvform = interp_m[max_idx]/interp_sd[max_idx]
    
    fwhm = get_fwhm(interp_m,interp_t)
    fwhm = 1000.*fwhm # in ms
    p2t_ratio,p2t_time = get_peak_trough(interp_m,interp_t)
    p2t_time = 1000.* p2t_time # in ms
    
    return interp_t,interp_m,interp_sd,peak_snr_wvform,fwhm,p2t_ratio,p2t_time
     
## OR tuning    
def get_or_tuning(location, sess, unit):

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
            
    def get_OSI(orn,m):
        m = np.asarray(m)
        orn = np.asarray(orn)
        
        assert np.size(m)==np.size(orn)
        
        # ensure ascending orientation
        order = np.argsort(orn)
        orn = orn[order]
        m = m[order]
        
        # get or with highest mean firing rate
        which_max = np.argmax(m)
        or_max = orn[which_max]
        if (which_max>(np.size(m)/2)):
            or_min = or_max-(np.pi/2)
        else:
            or_min = or_max+(np.pi/2)
        which_min = np.argmin(np.abs((orn-or_min)))
        
        m_max = m[which_max]
        m_min = m[which_min]
        np.seterr(all='raise')
        try:
            osi = ((m_max-m_min)/(m_max+m_min))
        except:
            osi = None
        return osi,or_max
        
    def get_vector_sum(orn,m):
        m = np.asarray(m)
        orn = np.asarray(orn)
        orn = 2*(-orn + (np.pi/2)) 
        
        assert np.size(m)==np.size(orn)
        
        complex_sum = np.dot(m,np.exp(1j*orn))    
        strength = np.abs(complex_sum)
        angle = np.angle(complex_sum)
        if angle<0:
            angle = 2*np.pi+angle
        angle = np.pi/2-angle/2
        
        if angle > np.pi/2:
            angle = np.pi-angle
        elif angle<-np.pi/2:
            angle = np.pi+angle
        return strength,angle
    
    def get_or_tuning_dict(spike_raster,trial_numbers,orientations):
        or_tuning = {'orientation':[],'m_rate':[],'std_rate':[],'n_trials':[]}
        spikes_found = False
        for orn in np.unique(orientations):
            # find those trials and find mean and std 
            trs = trial_numbers[orientations==orn]
            spike_nums_that_or = [np.size(spike_raster[tr]) for tr in trs]
            sp_mean_that_or = np.mean(spike_nums_that_or)
            if sp_mean_that_or != 0:
                spikes_found = True
            sp_std_that_or = np.std(spike_nums_that_or)
            or_tuning['orientation'].append(orn)
            or_tuning['n_trials'].append(np.size(trs))
            or_tuning['m_rate'].append(sp_mean_that_or)
            or_tuning['std_rate'].append(sp_std_that_or)
        if not spikes_found:
            print('Non spikes in orientation tuning for that unit')
        return(or_tuning, spikes_found)
    
    unit_details = {}
    failed_why = None
    with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
        data = pickle.load(f)
    try:
        stepname,durations = get_orientation_tuning_stepname(sess)
    except:
        print('failed to get stepname')
        stepname = None
    framechan,shift = get_frame_channel(sess)
    
    if not stepname: 
        failed_why = 'no_or_tuning_stepname'
        return unit_details, failed_why
    
    trial_numbers = np.asarray(data['trial_records']['trial_number'])
    step_names = np.asarray(data['trial_records']['step_name'])
    contrasts = np.asarray(data['trial_records']['contrast'])
    max_durations = np.asarray(data['trial_records']['max_duration'])
    phases = np.asarray(data['trial_records']['phase'])
    orientations = np.asarray(data['trial_records']['orientation'])
    
    which_step = step_names==stepname
    trial_numbers = trial_numbers[which_step]
    step_names = step_names[which_step]
    contrasts = contrasts[which_step]
    max_durations = max_durations[which_step]
    phases = phases[which_step]
    orientations = orientations[which_step]
    frame_start_index = []
    frame_end_index = []    
    for trial_number in trial_numbers:
        event_for_trial = data['trial_records']['events']['ttl_events'][trial_number]
        if shift:
            frame_start_index.append(event_for_trial[framechan]['rising'][1])
            frame_end_index.append(event_for_trial[framechan]['rising'][-2])
        else:
            frame_start_index.append(event_for_trial[framechan]['rising'][0])
            frame_end_index.append(event_for_trial[framechan]['rising'][-1])
            
    frame_start_index = np.asarray(frame_start_index)
    frame_end_index = np.asarray(frame_end_index)
        
    spike_time = np.squeeze(np.asarray(unit['spike_time']))
    spike_raster = {}
    for trial_number in trial_numbers:
        frame_start_time = frame_start_index[trial_numbers==trial_number][0]/30000
        frame_end_time = frame_end_index[trial_numbers==trial_number][0]/30000
        
        spikes_that_trial = spike_time[np.bitwise_and(spike_time>frame_start_time,spike_time<frame_end_time)]-frame_start_time
        spike_raster[trial_number] = spikes_that_trial
    or_tuning, spikes_found = get_or_tuning_dict(spike_raster,trial_numbers,orientations)
    if spikes_found:
        unit_details['or_tuning'] = or_tuning
        unit_details['osi'] = get_OSI(or_tuning['orientation'],or_tuning['m_rate'])
        unit_details['vector_sum'] = get_vector_sum(or_tuning['orientation'],or_tuning['m_rate'])
        # jack_knife
        unit_details['trial_jackknife'] = []
        unit_details['osi_jackknife'] = []
        unit_details['vector_sum_jackknife'] = []
        for key in spike_raster:
            # prep the copies
            spike_raster_jack_knife = spike_raster.copy()
            which_tr = trial_numbers==key
            idx = np.where(np.logical_not(which_tr))[0]
            trial_number_jack_knife = trial_numbers[idx]
            orientation_jack_knife = orientations[idx]
            del spike_raster_jack_knife[key]
            
            or_tuning_jack_knife,spikes_found_jackknife=get_or_tuning_dict(spike_raster_jack_knife,trial_number_jack_knife,orientation_jack_knife)
            unit_details['trial_jackknife'].append(key)
            unit_details['osi_jackknife'].append(get_OSI(or_tuning_jack_knife['orientation'],or_tuning_jack_knife['m_rate']))
            unit_details['vector_sum_jackknife'].append(get_vector_sum(or_tuning_jack_knife['orientation'],or_tuning_jack_knife['m_rate']))
        print('done_unit')
    else:
        unit_details = {}
        failed_why = 'no_spikes'
    return unit_details, failed_why
    
## Quality
def get_unit_quality(unit, model_loc,session_type='OE'):
    kwik_model = get_model(model_loc,session_type=session_type)
    
    kwik_model.channel_group = unit['shank_no']
    kwik_model.clustering = 'main'
    spike_time = kwik_model.spike_samples.astype(np.float64)/kwik_model.sample_rate
    spike_id = kwik_model.spike_ids
    cluster_id = kwik_model.spike_clusters
    fet_masks = kwik_model.all_features_masks
    fet = np.squeeze(fet_masks[:,:,0])
    
    that_cluster_idx = np.argwhere(cluster_id==unit['cluster_id'])
    other_cluster_idx = np.argwhere(cluster_id!=unit['cluster_id'])
    
    fet_that = np.squeeze(fet[that_cluster_idx,:])
    fet_other = np.squeeze(fet[other_cluster_idx,:])
    
    uq,cr,failure_mode = cluster_quality_core(fet_that,fet_other)

    return uq,cr,failure_mode

     
if __name__=='__main__':
    quality = get_features('unit_quality')
    with open('details_quality.pickle','wb') as f:
        pickle.dump(quality,f)
    
    waveform_details = get_features('waveform_details')
    with open('details_waveform.pickle','wb') as f:
        pickle.dump(waveform_details,f)
    
    firing_rates = get_features('firing_rates')
    with open('details_fr.pickle','wb') as f:
        pickle.dump(firing_rates,f)
        