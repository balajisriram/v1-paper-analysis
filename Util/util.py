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

def get_session_date(sess,type='OE'):
    splits = sess.split('_')
    if type=='OE':
        date_time = splits[1]+'_'+splits[2]
        return datetime.strptime(date_time,"%Y-%m-%d_%H-%M-%S")
    elif type=='plexon':
        date_time = splits[-1]
        try:dt_obj = datetime.strptime(date_time,"%m%d%Y")
        except:print(sess)        
        return 
    
def get_unit_depth(sess,y_loc,type='OE'):
    return(get_base_depth(sess,type=type)-y_loc)

def get_best_waveform(m_waveform):
    idx = np.unravel_index(np.argmax(m_waveform),m_waveform.shape)
    return m_waveform[:,idx[1]]

def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]
 
## FRAME AND LED CHANS 
def get_frame_channel(sess,type='OE'):
    if type=='OE':
        if sess in ['bas080_2017-06-07_16-36-59','bas080_2017-06-08_16-39-59','bas081a_2017-06-23_12-55-28','bas081a_2017-06-24_11-57-43','bas081a_2017-06-24_12-55-51','bas081a_2017-06-25_12-44-45','bas081b_2017-07-28_20-31-03','m311_2017-07-31_16-15-43','m311_2017-07-31_17-08-20','m311_2017-08-01_12-05-12','m311_2017-08-01_13-08-00','m325_2017-08-10_13-14-51','m325_2017-08-10_14-21-22','m317_2017-08-24_15-52-35',
        'm317_2017-08-24_16-48-34','m317_2017-08-25_14-21-08','m318_2017-08-26_18-14-22','m318_2017-08-26_19-28-43']:
            return 0,0
        else:
            return 1,2
    else:
        return None,None
        
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

def flatten_wvform_dict(inp):
    
    out = []
    for unit in inp:
        temp = {}
        temp['unit_id'] = unit['unit_id']
        temp['wv_snr'] = unit[('waveform', 'snr')]
        temp['wv_mu'] = unit[('waveform', 'mu')]
        temp['wv_sd'] = unit[('waveform', 'sd')]
        temp['wv_p2tr'] = unit[('waveform', 'p2t_ratio')]
        temp['wv_p2tt'] = unit[('waveform', 'p2t_time')]
        temp['wv_fwhm'] = unit[('waveform', 'fwhm')]
        out.append(temp)
    return out
        
## OR tuning    
def get_or_tuning_details(sess, trial_records, unit, type='OE'):

    def get_orientation_tuning_stepname(sess,type='OE'):
        subj = get_subject_from_session(sess)
        if type=='OE':
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
        else:
            if sess not in ['b8_02022019','g2_2_01312019','g2_2_02012019','g2_2_02022019']:
                return 'or_tuning_ts_8_ors_drift_2Hz_2s_fullC',2000
            else: return None,None
          
    def get_OSI(orn,m,type='OE'):
        m = np.asarray(m)
        if type=='OE':orn = np.asarray(orn)
        else:orn = np.deg2rad(orn)
        
        assert np.size(m)==np.size(orn)
        
        # ensure ascending orientation
        order = np.argsort(orn)
        orn = orn[order]
        m = m[order]
        
        # get or with highest mean firing rate
        try:which_max = np.argmax(m)
        except:pdb.set_trace()
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
        except FloatingPointError as e:
            print(e)
            osi = None
        except error as e:
            print(e)
            pdb.set_trace()
        return osi,or_max
        
    def get_vector_sum(orn,m,type='OE'):
        m = np.asarray(m)
        if type=='OE':orn = np.asarray(orn)
        else:orn = np.deg2rad(orn)
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
    
    def get_start_and_end_time_by_trial(trial_numbers,ttl_events,framechan_deets,type='OE'):
        if type=='OE':
            
            frame_chan,shift = framechan_deets
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                if shift:
                    frame_start_time.append(event_for_trial[framechan]['rising'][1])
                    frame_end_time.append(event_for_trial[framechan]['rising'][-2])
                else:
                    frame_start_time.append(event_for_trial[framechan]['rising'][0])
                    frame_end_time.append(event_for_trial[framechan]['rising'][-1])
            frame_start_time = np.asarray(frame_start_time)/30000.
            frame_end_time = np.asarray(frame_end_time)/30000.
            frame_end_time = frame_end_time+1/60. # goes on till the end of the last frame...
        else:
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                frame_start_time.append(event_for_trial['frame'][0])
                frame_end_time.append(event_for_trial['frame'][-1])
            frame_start_time = np.asarray(frame_start_time)
            frame_end_time = np.asarray(frame_end_time)
            frame_end_time = frame_end_time+1/60. # goes on till the end of the last frame...
        return frame_start_time,frame_end_time
        
    # get_or_tuning
    unit_details = {}
    failed_why = None

    try:
        stepname,durations = get_orientation_tuning_stepname(sess,type=type)
    except:
        print('failed to get stepname')
        stepname = None
    framechan,shift = get_frame_channel(sess,type=type)
    
    if not stepname: 
        failed_why = 'no_or_tuning_stepname'
        return unit_details, failed_why
    
    # get the data and filter for stepname
    trial_numbers = np.asarray(trial_records['trial_number'])
    step_names = np.asarray(trial_records['step_name'])
    contrasts = np.asarray(trial_records['contrast'])
    max_durations = np.asarray(trial_records['max_duration'])
    phases = np.asarray(trial_records['phase'])
    orientations = np.asarray(trial_records['orientation'])    
    which_step = step_names==stepname
    trial_numbers = trial_numbers[which_step]
    step_names = step_names[which_step]
    contrasts = contrasts[which_step]
    max_durations = max_durations[which_step]
    phases = phases[which_step]
    orientations = orientations[which_step]
    
    # get the spike raster
    frame_start_time,frame_end_time = get_start_and_end_time_by_trial(trial_numbers, trial_records['events']['ttl_events'], (framechan,shift), type=type)    
    spike_time = np.squeeze(np.asarray(unit['spike_time']))
    spike_raster = {}
    for ii,trial_number in enumerate(trial_numbers):
        try:spike_raster[trial_number] = spike_time[np.bitwise_and(spike_time>frame_start_time[ii],spike_time<frame_end_time[ii])]-frame_start_time[ii]
        except:pdb.set_trace()
    or_tuning, spikes_found = get_or_tuning_dict(spike_raster,trial_numbers,orientations)
    
    if spikes_found:
        unit_details['or_tuning'] = or_tuning
        unit_details['osi'] = get_OSI(or_tuning['orientation'],or_tuning['m_rate'],type=type)
        unit_details['vector_sum'] = get_vector_sum(or_tuning['orientation'],or_tuning['m_rate'],type=type)
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
    else:
        unit_details = {}
        failed_why = 'no_spikes'
    return unit_details, failed_why

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

def flatten_ortune_dict(inp):
    out = []
    for unit in inp:
        temp = {}
        temp['unit_id'] = unit['unit_id']
        if unit['failed_why'] in ['no_or_tuning_stepname','no_spikes']:
            temp['has_or_tuning'] = False
            temp['osi'] = np.nan
            temp['osi_ang'] = np.nan
            temp['vecsum_amp'] = np.nan
            temp['vecsum_ang'] = np.nan
            temp['osi_jk_mu'] = np.nan
            temp['osi_jk_sd'] = np.nan
            temp['osi_jk_n'] = np.nan
            temp['vecsum_jk_mu'] = np.nan
            temp['vecsum_jk_sd'] = np.nan
            temp['vecsum_jk_n'] = np.nan
        elif not unit['failed_why']:
            try:
                temp['has_or_tuning'] = True
                temp['osi'] = unit['or_tuning_details']['osi'][0]
                temp['osi_ang'] = unit['or_tuning_details']['osi'][1]
                temp['vecsum_amp'] = unit['or_tuning_details']['vector_sum'][0]
                temp['vecsum_ang'] = unit['or_tuning_details']['vector_sum'][1]
                osi_jk_vals = np.array([jk[0] for jk in unit['or_tuning_details']['osi_jackknife']],dtype=np.float)
                temp['osi_jk_mu'] = np.nanmean(osi_jk_vals)
                temp['osi_jk_sd'] = np.nanstd(osi_jk_vals)
                temp['osi_jk_n'] = np.sum(~np.isnan(osi_jk_vals))
                vecsum_jk_vals = np.array([jk[1] for jk in unit['or_tuning_details']['vector_sum_jackknife']],dtype=np.float)
                temp['vecsum_jk_mu'] = np.nanmean(vecsum_jk_vals)
                temp['vecsum_jk_sd'] = np.nanstd(vecsum_jk_vals)
                temp['vecsum_jk_n'] = np.sum(~np.isnan(vecsum_jk_vals))
            except Exception as e:
                print(e)
                pdb.set_trace()
        else:
            pdb.set_trace()
        
        out.append(temp)
    return out
## Long_dur_decoding
def get_longdur_tuning_details(sess, trial_records, unit, type='OE'):

    def get_long_dur_stepname(sess,type='OE'):
        subj = get_subject_from_session(sess)
        if type=='OE':
            if subj in ['bas080','bas081a']: return 'gratings_LongDuration'
            elif subj in ['bas081b','m311','m325']: return 'LongDurationOR_LED'
        else:
            return 'or_decoding_pm45deg_8phases_2Hz_2s_full_and_lo_C'
    
    def get_start_and_end_time_by_trial(trial_numbers,ttl_events,framechan_deets,type='OE'):
        if type=='OE':
            frame_chan,shift = framechan_deets
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                if shift:
                    frame_start_time.append(event_for_trial[framechan]['rising'][1])
                    frame_end_time.append(event_for_trial[framechan]['rising'][-2])
                else:
                    frame_start_time.append(event_for_trial[framechan]['rising'][0])
                    frame_end_time.append(event_for_trial[framechan]['rising'][-1])
            frame_start_time = np.asarray(frame_start_time)/30000.
            frame_end_time = np.asarray(frame_end_time)/30000.
            frame_end_time = frame_end_time+1/60. # goes on till the end of the last frame...
        else:
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                frame_start_time.append(event_for_trial['frame'][0])
                frame_end_time.append(event_for_trial['frame'][-1])
            frame_start_time = np.asarray(frame_start_time)
            frame_end_time = np.asarray(frame_end_time)
            frame_end_time = frame_end_time+1/60. # goes on till the end of the last frame...
        return frame_start_time,frame_end_time
    
    # get_or_tuning
    unit_details = {}
    failed_why = 'n/a'
    stepname = get_long_dur_stepname(sess,type=type)
    if not stepname: 
        failed_why = 'no_longdur_decoding_stepname'
        return unit_details, failed_why
    framechan,shift = get_frame_channel(sess,type=type)
    
    # get the data and filter for stepname
    trial_numbers = np.asarray(trial_records['trial_number'])
    step_names = np.asarray(trial_records['step_name'])
    contrasts = np.asarray(trial_records['contrast'])
    max_durations = np.asarray(trial_records['max_duration'])
    phases = np.asarray(trial_records['phase'])
    orientations = np.asarray(trial_records['orientation'])    
    which_step = step_names==stepname
    trial_numbers = trial_numbers[which_step]
    step_names = step_names[which_step]
    contrasts = contrasts[which_step]
    max_durations = max_durations[which_step]
    phases = phases[which_step]
    orientations = orientations[which_step]
    
    # get the spike raster
    frame_start_time,frame_end_time = get_start_and_end_time_by_trial(trial_numbers, trial_records['events']['ttl_events'],(framechan,shift),type=type)    
    spike_time = np.squeeze(np.asarray(unit['spike_time']))
    spike_raster = {}
    for ii,trial_number in enumerate(trial_numbers):
        try:spike_raster[trial_number] = spike_time[np.bitwise_and(spike_time>frame_start_time[ii],spike_time<frame_end_time[ii])]-frame_start_time[ii]
        except:pdb.set_trace()
        
    unit_details['orientations'] = orientations
    unit_details['phases'] = phases
    unit_details['max_durations'] = max_durations
    unit_details['contrasts'] = contrasts
    unit_details['trial_numbers'] = trial_numbers
    unit_details['spike_raster'] = spike_raster
    
    return unit_details,failed_why
    
## Short_dur_decoding
def get_shortdur_tuning_details(sess, trial_records, unit, type='OE'):

    def get_short_response_stepnames(sess,type='OE'):
        subj = get_subject_from_session(sess)
        if type=='OE':
            if subj in ['bas072','bas074','m284']:
                return ['gratings_LED']
            elif subj in ['bas077','bas078','bas079','bas080','bas081a']:
                return  ['gratings_NOLED','gratings_baseLine']
            elif subj in ['bas070']:
                if sess in ['bas070_2015-08-05_12-31-50','bas070_2015-08-06_12-09-26','bas070_2015-08-11_12-51-37']:
                    return ['gratings']
                elif sess in ['bas070_2015-09-22_12-30-10','bas070_2015-09-23_15-23-36','bas070_2015-09-24_15-47-24','bas070_2015-09-25_14-18-57','bas070_2015-09-28_12-24-33','bas070_2015-09-29_14-51-50','bas070_2015-09-30_13-22-42','bas070_2015-10-01_11-27-45','bas070_2015-10-02_11-46-50']:
                    return ['gratings_LED']
                else:
                    return ['gratings_LED']
            elif subj in ['bas081b']:
                return ['ShortDurationOR']
            elif subj in ['m317','m311','m318','m325']:
                return ['ShortDurationOR']
        else:
            if sess not in ['b8_01312019','b8_02022019','b8_02032019','b8_02042019','g2_2_01312019','g2_2_02012019','g2_2_02022019']: return ['short_duration_pm45deg_8phases']            
 
    def get_start_and_end_time_by_trial(trial_numbers,ttl_events,framechan_deets,type='OE'):
        if type=='OE':
            frame_chan,shift = framechan_deets
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                try:
                    if shift:
                        frame_start_time.append(event_for_trial[framechan]['rising'][1])
                        frame_end_time.append(event_for_trial[framechan]['rising'][1]+150000) # all the way to +5 seconds
                    else:
                        frame_start_time.append(event_for_trial[framechan]['rising'][0])
                        frame_end_time.append(event_for_trial[framechan]['rising'][0]+150000) # all the way to +5 seconds
                except Exception as e: 
                    print(e)
                    frame_start_time.append(np.nan)
                    frame_end_time.append(np.nan)
            frame_start_time = np.asarray(frame_start_time)/30000.
            frame_end_time = np.asarray(frame_end_time)/30000.
        else:
            frame_start_time = []
            frame_end_time = []    
            for trial_number in trial_numbers:
                event_for_trial = ttl_events[trial_number]
                frame_start_time.append(event_for_trial['frame'][0])
                frame_end_time.append(event_for_trial['frame'][0]+5.0) # all the way to +5 seconds
            frame_start_time = np.asarray(frame_start_time)
            frame_end_time = np.asarray(frame_end_time)
        return frame_start_time,frame_end_time
    
    # get_short_duration_stepname
    unit_details = {}
    failed_why = 'n/a'
    stepname = get_short_response_stepnames(sess,type=type)
    if not stepname: 
        failed_why = 'no_shortdur_decoding_stepname'
        return unit_details, failed_why
    framechan,shift = get_frame_channel(sess,type=type)
    
    
    # get the data and filter for stepname
    trial_numbers = np.asarray(trial_records['trial_number'])
    step_names = np.asarray(trial_records['step_name'])
    contrasts = np.asarray(trial_records['contrast'])
    max_durations = np.asarray(trial_records['max_duration'])
    phases = np.asarray(trial_records['phase'])
    orientations = np.asarray(trial_records['orientation']) 
    # get the step filter    
    which_step = step_names==stepname[0]
    for st in stepname:
        which_step = np.bitwise_or(which_step,step_names==st)
    trial_numbers = trial_numbers[which_step]
    step_names = step_names[which_step]
    contrasts = contrasts[which_step]
    max_durations = max_durations[which_step]
    phases = phases[which_step]
    orientations = orientations[which_step]
    
    # get the spike raster
    frame_start_time,frame_end_time = get_start_and_end_time_by_trial(trial_numbers, trial_records['events']['ttl_events'],(framechan,shift), type=type)    
    spike_time = np.squeeze(np.asarray(unit['spike_time']))
    spike_raster = {}
    assert len(frame_start_time)==len(trial_numbers),'something weird is happenineg'
    for ii,trial_number in enumerate(trial_numbers):
        if np.isnan(frame_start_time[ii]): 
            # print('found a weird trial')
            spike_raster[trial_number] = np.nan
            continue
        try:spike_raster[trial_number] = spike_time[np.bitwise_and(spike_time>(frame_start_time[ii]-0.5),spike_time<frame_end_time[ii])]-frame_start_time[ii]
        except:pdb.set_trace()
        
    unit_details['orientations'] = orientations
    unit_details['phases'] = phases
    unit_details['max_durations'] = max_durations
    unit_details['contrasts'] = contrasts
    unit_details['trial_numbers'] = trial_numbers
    unit_details['spike_raster'] = spike_raster
    return unit_details,failed_why
    
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

def flatten_unitqual_dict(inp):
    out = []
    for unit in inp:
        temp = {}
        temp['unit_id'] = unit['unit_id']
        
        if unit['failure_reason']=='nominal':
            temp['has_cluster_quality'] = False
            temp['contamination_rate'] = np.nan
            temp['unit_quality'] = np.nan
        elif unit['failure_reason']=='n/a':
            temp['has_cluster_quality'] = True
            temp['contamination_rate'] = unit['contamination_rate']
            temp['unit_quality'] = unit['unit_quality']
        else:
            pdb.set_trace()
        # elif npt unit['failure_reason']=='nominal':
            # temp['has_cluster_quality'] = False
            # temp['contamination_rate'] = np.nan
            # temp['unit_quality'] = np.nan

        out.append(temp)
    return out
## STUFF
def get_features(what,base_loc =r'C:\Users\bsriram\Desktop\Data_V1Paper'):
    out = []
    sub_locs = ['DetailsProcessedPhysOnly','DetailsProcessedBehaved','EyeTracked']
    # sub_locs = ['EyeTracked']
    types = ['OE','OE','plexon']
    # types = ['plexon']
    session_types = ['PhysOnly','Behaved','EyeTracked']
    for type,session_type,sub_loc in zip(types,session_types,sub_locs):
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
                    elif what=='manual_quality':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp['manual_quality'] = unit['manual_quality']
                        out.append(temp)
                    elif what=='or_tuning_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        trial_records = sess_data['trial_records']
                        otd,fw = get_or_tuning_details(sess,trial_records,unit,type)
                        temp['or_tuning_details'] = otd
                        temp['failed_why'] = fw
                        out.append(temp)
                    elif what=='session_type':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp['session_type'] = session_type
                        out.append(temp)
                    elif what=='long_dur_decoding_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        trial_records = sess_data['trial_records']
                        lddd,fw = get_longdur_tuning_details(sess,trial_records,unit,type)
                        temp['long_dur_decoding_details'] = lddd
                        temp['failed_why'] = fw
                        out.append(temp)
                    elif what=='short_dur_decoding_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        trial_records = sess_data['trial_records']
                        sddd,fw = get_shortdur_tuning_details(sess,trial_records,unit,type)
                        temp['short_dur_decoding_details'] = sddd
                        temp['failed_why'] = fw
                        out.append(temp)
                    elif what=='spike_location_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp['x_relative'] = unit['x_loc']
                        temp['y_relative'] = unit['y_loc']
                        temp['y_absolute'] = get_unit_depth(sess,unit['y_loc'],type=type)
                        out.append(temp)
                    elif what=='session_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        temp['session'] = sess
                        temp['session_date'] = get_session_date(sess,type=type)
                        temp['subject_id'] = get_subject_from_session(sess)
                        out.append(temp)
                    elif what=='isi_details':
                        temp = {}
                        temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                        spike_train = unit['spike_time']
                        n_spikes = np.size(spike_train)
                        isi = np.diff(spike_train, axis=0)
                        num_violations = np.sum(isi<0.001)
                        total_time = spike_train[-1]
                        temp['n_spikes'] = n_spikes
                        temp['num_isi_violations'] = num_violations
                        temp['isi_violation_rate'] = num_violations/total_time
                        temp['isi_violation_fraction'] = num_violations/n_spikes
                        out.append(temp)
                        
                        
    return out
    
def make_and_merge_dfs():
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\session_types_details.pickle','rb') as f:
        sess_type = pickle.load(f)
    sess_type_df = pd.DataFrame(sess_type)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\details_fr.pickle','rb') as f:
        fr = pickle.load(f)
    fr_df = pd.DataFrame(fr)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\details_manual_quality.pickle','rb') as f:
        mq = pickle.load(f)
    mq_df = pd.DataFrame(mq)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\details_quality.pickle','rb') as f:
        cq = pickle.load(f)
    cq = flatten_unitqual_dict(cq)
    cq_df = pd.DataFrame(cq)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\details_waveform.pickle','rb') as f:
        wv = pickle.load(f)
    wv = flatten_wvform_dict(wv)
    wv_df = pd.DataFrame(wv)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\details_or_tuning.pickle','rb') as f:
        or_tune = pickle.load(f)
    or_tune = flatten_ortune_dict(or_tune)
    or_tune_df = pd.DataFrame(or_tune)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\spike_loc_details.pickle','rb') as f:
        loc = pickle.load(f)
    loc_df = pd.DataFrame(loc)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\session_details.pickle','rb') as f:
        sess_deets = pickle.load(f)
    sess_deets_df = pd.DataFrame(sess_deets)
    
    with open(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\AnalysisRuns\isi_details.pickle','rb') as f:
        isi_deets = pickle.load(f)
    isi_deets_df = pd.DataFrame(isi_deets)
    
    
    neuron_df = sess_type_df
    neuron_df = neuron_df.merge(fr_df)
    neuron_df = neuron_df.merge(mq_df)
    neuron_df = neuron_df.merge(cq_df)
    neuron_df = neuron_df.merge(wv_df)
    neuron_df = neuron_df.merge(or_tune_df)
    neuron_df = neuron_df.merge(loc_df)
    neuron_df = neuron_df.merge(sess_deets_df)
    neuron_df = neuron_df.merge(isi_deets_df)
    # neuron_df = neuron_df.set_index('unit_id')
    neuron_df.to_pickle("AllNeurons.df")
        
def make_long_dur_session_dfs(base_loc =r'C:\Users\bsriram\Desktop\Data_V1Paper',out_loc=r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\LongDurSessionDFs'):
    sub_locs = ['DetailsProcessedPhysOnly','DetailsProcessedBehaved','EyeTracked']
    types = ['OE','OE','plexon']
    session_types = ['PhysOnly','Behaved','EyeTracked']
    for type,session_type,sub_loc in zip(types,session_types,sub_locs):
        for sess in tqdm(os.listdir(os.path.join(base_loc,sub_loc))):
            subj = get_subject_from_session(sess)
            out = []
            spike_and_trial_file = [f for f in os.listdir(os.path.join(base_loc,sub_loc,sess)) if f.startswith('spike_and_trials')]
            assert len(spike_and_trial_file)==1, 'too many or too few spike and trials pickle'
            spike_and_trial_file = spike_and_trial_file[0]
            with open(os.path.join(base_loc,sub_loc,sess,spike_and_trial_file),'rb') as f:
                sess_data = pickle.load(f)
            for unit in sess_data['spike_records']['units']:
                if unit['manual_quality'] in ['good','mua']:
                    temp = {}
                    temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                    trial_records = sess_data['trial_records']
                    lddd,fw = get_longdur_tuning_details(sess,trial_records,unit,type)
                    temp['long_dur_decoding_details'] = lddd
                    temp['failed_why'] = fw
                    out.append(temp)
            if out:
                reason, out = flatten_long_dur_dict(out,type=type)
                if reason=='has_data':
                    df = pd.DataFrame(out)
                    df_name = sess+'.df'
                    df.to_pickle(os.path.join(out_loc,df_name))
    return out
    
def make_short_dur_session_dfs(base_loc =r'C:\Users\bsriram\Desktop\Data_V1Paper',out_loc=r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\ShortDurSessionDFs'):
    sub_locs = ['DetailsProcessedPhysOnly','DetailsProcessedBehaved','EyeTracked']
    types = ['OE','OE','plexon']
    session_types = ['PhysOnly','Behaved','EyeTracked']
    for type,session_type,sub_loc in zip(types,session_types,sub_locs):
        for sess in tqdm(os.listdir(os.path.join(base_loc,sub_loc))):
            subj = get_subject_from_session(sess)
            out = []
            spike_and_trial_file = [f for f in os.listdir(os.path.join(base_loc,sub_loc,sess)) if f.startswith('spike_and_trials')]
            assert len(spike_and_trial_file)==1, 'too many or too few spike and trials pickle'
            spike_and_trial_file = spike_and_trial_file[0]
            with open(os.path.join(base_loc,sub_loc,sess,spike_and_trial_file),'rb') as f:
                sess_data = pickle.load(f)
            for unit in sess_data['spike_records']['units']:
                if unit['manual_quality'] in ['good','mua']:
                    temp = {}
                    temp['unit_id'] = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
                    trial_records = sess_data['trial_records']
                    sddd,fw = get_shortdur_tuning_details(sess,trial_records,unit,type)
                    temp['short_dur_decoding_details'] = sddd
                    temp['failed_why'] = fw
                    out.append(temp)
            if out:
                reason, out = flatten_short_dur_dict(out,type=type)
                if reason=='has_data':
                    df = pd.DataFrame(out)
                    df_name = sess+'.df'
                    df.to_pickle(os.path.join(out_loc,df_name))
    return out

def verify_uint_qualities(base_loc =r'C:\Users\bsriram\Desktop\Data_V1Paper'):
    sub_locs = ['DetailsProcessedPhysOnly','DetailsProcessedBehaved','EyeTracked']
    types = ['OE','OE','plexon']
    session_types = ['PhysOnly','Behaved','EyeTracked']
    for type,session_type,sub_loc in zip(types,session_types,sub_locs):
        for sess in os.listdir(os.path.join(base_loc,sub_loc)):
            subj = get_subject_from_session(sess)
            out = []
            spike_and_trial_file = [f for f in os.listdir(os.path.join(base_loc,sub_loc,sess)) if f.startswith('spike_and_trials')]
            assert len(spike_and_trial_file)==1, 'too many or too few spike and trials pickle'
            spike_and_trial_file = spike_and_trial_file[0]
            with open(os.path.join(base_loc,sub_loc,sess,spike_and_trial_file),'rb') as f:
                sess_data = pickle.load(f)
            quals = np.asarray([unit['manual_quality'] for unit in sess_data['spike_records']['units']])
            print('{0}:{1}'.format(sess,np.unique(quals)))

def flatten_long_dur_dict(inp,type='OE'):
    out = {}
    # get the failed_why in the session
    failed_whys = np.asarray([f['failed_why'] for f in inp])
    if np.all(failed_whys=='no_longdur_decoding_stepname'):
        what_happened = 'no_data'
    elif np.all(failed_whys=='n/a'):
        # get the units in the session
        units = [u['unit_id'] for u in inp]
        # get the trial_numbers
        ldd = 'long_dur_decoding_details'
        t_nums = inp[0][ldd]['trial_numbers']
        ors = inp[0][ldd]['orientations']
        ctrs = inp[0][ldd]['contrasts']
        phis = inp[0][ldd]['phases']
        durs = inp[0][ldd]['max_durations']
        
        # check if all units have the same trial numbers, len(orientations),len(contrasts),len(phases),len(max_durations)
        for u in inp:
            assert np.all(u[ldd]['trial_numbers']==t_nums), 'different trial numbers found for {0}'.format(u['unit_id'])
            assert np.all(u[ldd]['orientations']==ors), 'different orientations found for {0}'.format(u['unit_id'])
            assert np.all(u[ldd]['contrasts']==ctrs), 'different contrasts found for {0}'.format(u['unit_id'])
            assert np.all(u[ldd]['phases']==phis), 'different phases found for {0}'.format(u['unit_id'])
            assert np.all(u[ldd]['max_durations']==durs), 'different max_durations found for {0}'.format(u['unit_id'])

            assert len(u[ldd]['spike_raster'])==len(t_nums), 'different len(spike_raster) found for {0}'.format(u['unit_id'])
            
        # now we are back to business
        out['trial_number'] = t_nums
        if type=='OE':
            out['orientations'] = np.rad2deg(ors)
        else:
            out['orientations'] = ors
        out['contrasts'] = ctrs
        out['phases'] = phis
        if type=='OE':
            out['durations'] = durs/60.
        else:
            out['durations'] = durs
        
        for u in inp:
            current_unit = []
            uid = u['unit_id']
            for t in t_nums:
                current_unit.append(u[ldd]['spike_raster'][t])
            out[uid] = current_unit
        what_happened = 'has_data'
    else:
        pdb.set_trace()
        print('trial')
    return what_happened, out

def flatten_short_dur_dict(inp, type='OE'):
    out = {}
    # get the failed_why in the session
    failed_whys = np.asarray([f['failed_why'] for f in inp])
    if np.all(failed_whys=='no_shortdur_decoding_stepname'):
        what_happened = 'no_data'
    elif np.all(failed_whys=='n/a'):
        # get the units in the session
        units = [u['unit_id'] for u in inp]
        # get the trial_numbers
        sdd = 'short_dur_decoding_details'
        t_nums = inp[0][sdd]['trial_numbers']
        ors = inp[0][sdd]['orientations']
        ctrs = inp[0][sdd]['contrasts']
        phis = inp[0][sdd]['phases']
        durs = inp[0][sdd]['max_durations']
        
        # check if all units have the same trial numbers, len(orientations),len(contrasts),len(phases),len(max_durations)
        for unit in inp:
            assert np.all(unit[sdd]['trial_numbers']==t_nums), 'different trial numbers found for {0}'.format(u['unit_id'])
            assert np.all(unit[sdd]['orientations']==ors), 'different orientations found for {0}'.format(u['unit_id'])
            assert np.all(unit[sdd]['contrasts']==ctrs), 'different contrasts found for {0}'.format(u['unit_id'])
            assert np.all(unit[sdd]['phases']==phis), 'different phases found for {0}'.format(u['unit_id'])
            assert np.all(unit[sdd]['max_durations']==durs), 'different max_durations found for {0}'.format(u['unit_id'])

            assert len(unit[sdd]['spike_raster'])==len(t_nums), 'different len(spike_raster) found for {0}'.format(u['unit_id'])
            
        # remove the ones from the end and then remove the nans
        sorted_tnums = np.sort(t_nums)
        sorted_tnums = sorted_tnums[::-1]
        bad_tnums = []
        unit = inp[0]
        for t in sorted_tnums:
            temp = unit[sdd]['spike_raster'][t]
            try:
                if temp.size==0: bad_tnums.append(t)
                else: break
            except ValueError:
                break
        bad_tnums = np.array(bad_tnums)
        
        for u in inp:
            bad_tnums_that_unit = []
            for t in sorted_tnums:
                temp = u[sdd]['spike_raster'][t]
                try:
                    if temp.size==0: bad_tnums_that_unit.append(t)
                    else: break
                except ValueError:
                    break
            bad_tnums_that_unit = np.array(bad_tnums_that_unit)
            bad_tnums = np.intersect1d(bad_tnums,bad_tnums_that_unit)
            nan_tnums_that_unit = []
            for t in t_nums:
                temp=u[sdd]['spike_raster'][t]
                try: 
                    if np.isnan(temp):nan_tnums_that_unit.append(t)
                except ValueError: continue
            nan_tnums_that_unit = np.array(nan_tnums_that_unit)
            bad_tnums = np.union1d(bad_tnums,nan_tnums_that_unit)
                
        which = np.isin(t_nums,bad_tnums,assume_unique =True,invert=True)
        # now we are back to business
        out['trial_number'] = t_nums[which]
        if type=='OE':
            out['orientations'] = np.rad2deg(ors[which])
        else:
            out['orientations'] = ors[which]
        out['contrasts'] = ctrs[which]
        out['phases'] = phis[which]
        if type=='OE':
            out['durations'] = durs[which]/60.
        else:
            out['durations'] = durs[which]
            
        good_tnums = t_nums[which]
        for u in inp:
            current_unit = []
            uid = u['unit_id']
            for t in good_tnums:
                current_unit.append(u[sdd]['spike_raster'][t])
            out[uid] = current_unit
            
        what_happened = 'has_data'
    else:
        pdb.set_trace()
        print('trial')
    return what_happened, out

if __name__=='__main__':
    # quality = get_features('unit_quality')
    # with open('details_quality.pickle','wb') as f:
        # pickle.dump(quality,f)
    
    # waveform_details = get_features('waveform_details')
    # with open('details_waveform.pickle','wb') as f:
        # pickle.dump(waveform_details,f)
    
    # firing_rates = get_features('firing_rates')
    # with open('details_fr.pickle','wb') as f:
        # pickle.dump(firing_rates,f)
        
    # manual_quality = get_features('manual_quality')
    # with open('details_manual_quality.pickle','wb') as f:
        # pickle.dump(manual_quality,f)
    
    # or_tuning_details = get_features('or_tuning_details')
    # with open('details_or_tuning.pickle','wb') as f:
        # pickle.dump(or_tuning_details,f)  
    
    # session_types = get_features('session_type')
    # with open('session_types_details.pickle','wb') as f:
        # pickle.dump(session_types,f) 

    # long_dur_decoding_details = get_features('long_dur_decoding_details')
    # with open('long_dur_decoding_details.pickle','wb') as f:
        # pickle.dump(long_dur_decoding_details,f)

    # short_dur_decoding_details = get_features('short_dur_decoding_details')
    # with open('short_dur_decoding_details.pickle','wb') as f:
        # pickle.dump(short_dur_decoding_details,f)        
    
    # spike_loc_details = get_features('spike_location_details')
    # with open('spike_loc_details.pickle','wb') as f:
        # pickle.dump(spike_loc_details,f)        
    
    # sess_details = get_features('session_details')
    # with open('session_details.pickle','wb') as f:
        # pickle.dump(sess_details,f)        
    
    # isi_details = get_features('isi_details')
    # with open('isi_details.pickle','wb') as f:
        # pickle.dump(isi_details,f)        
    
        
    # make_and_merge_dfs()
    # verify_uint_qualities()
    make_long_dur_session_dfs()
    make_short_dur_session_dfs()

    
        