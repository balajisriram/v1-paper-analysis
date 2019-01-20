from klusta.kwik.model import KwikModel
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import types
import pdb
import pickle
import pandas

def get_model(loc):
    kwik_file = [f for f in os.listdir(loc) if f.endswith('.kwik') and '100' in f]
    if len(kwik_file)>1 or len(kwik_file)==0:
        RuntimeError('Too many or too few files. Whats happening')
    kwik_file_path = os.path.join(loc,kwik_file[0])
    print("loading kwik file :", kwik_file_path)
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
        #unique_clu, clu_quality, cont_rate = CQMod.cluster_quality_all(clu, fet, mask, fet_N)
        #print("Finished getting quality for shank")
        
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
            #unit["quality"] = clu_quality[np.argwhere(unique_clu==clust_num)]
            #unit["contamination_rate"] =cont_rate[np.argwhere(unique_clu==clust_num)]            
            
            units.append(unit)
    output["units"] = units
    
    kwik_model.close()
    return output

def plot_ISI(unit, ax, record):
    spike_train = unit['spike_time']
    n_spikes = np.size(spike_train)
    isi = np.diff(spike_train, axis=0)
    histc, binedge = np.histogram(isi,range=(0,0.025),bins=100)
    
    ax.bar(binedge[0:100]*1000,histc,align='edge',edgecolor='none',color='b')
    ax.plot([1,1],ax.get_ylim(),'r')
    ax.plot([2,2],ax.get_ylim(),'r--')
    
    
    num_violations = np.sum(isi<0.001)
    total_time = spike_train[-1]
    violation_rate = num_violations/total_time
    total_rate = n_spikes/total_time
    record['isi_violation_rate'] = violation_rate
    record['spike_rate'] = total_rate
    ax.text(ax.get_xlim()[1],ax.get_ylim()[1],'viol_rate=%2.3fHz of %2.3fHz' % (violation_rate,total_rate), horizontalalignment='right',verticalalignment='top',fontsize=6)
    return record

def get_fwhm(wvform, t, ax = None, plot = False):    
    # find the min value and index of interp functionm
    wvform_min = np.min(wvform)
    half_min = wvform_min/2
    wvform_min_idx = np.argmin(wvform)
    
    idx_left = 0
    for i in range(wvform_min_idx,-1,-1):
        if wvform[i]>half_min:
           idx_left = i
           break
    
    idx_right = wvform.size
    for i in range(wvform_min_idx,wvform.size):
        if wvform[i]>half_min:
            idx_right = i
            break
    # print('min_idx::%d, left:: %d, right %d' %(wvform_min_idx, idx_left, idx_right))
    spike_width = t[idx_right]-t[idx_left]
    if plot:
        ax.plot([t[idx_left],t[idx_right]],[half_min,half_min],'-',color='blue')
    return spike_width

def get_peak_trough(wvform, t, ax=None, plot=False):
    # find the min value and index of interp functionm
    wvform_min = np.min(wvform)
    half_min = wvform_min/2
    wvform_min_idx = np.argmin(wvform)
    wvform_max_idx = np.argmax(wvform[wvform_min_idx:])+ wvform_min_idx
    
    p2t_ratio = -wvform[wvform_max_idx]/wvform[wvform_min_idx]
    p2t_time = t[wvform_max_idx]-t[wvform_min_idx]
    
    if plot:
        ax.plot(t[wvform_max_idx],wvform[wvform_max_idx],'gx')
        ax.plot(t[wvform_min_idx],wvform[wvform_min_idx],'rx')
    
    return p2t_ratio,p2t_time
    
def plot_unit_waveform(unit,ax, record):

    mu_all = unit['mean_waveform']
    std_all = unit['std_waveform']
    waveform_size = mu_all.shape[0]
    num_chans = mu_all.shape[1]
    
    # get the largest deviation in the negative direction
    max_ind = np.unravel_index(np.argmin(mu_all),mu_all.shape)
    max_chan = max_ind[1]
    
    mu_max = mu_all[:,max_chan]
    sd_max = std_all[:,max_chan]
    
    interp_fn_m = interpolate.interp1d(np.linspace(0,(waveform_size-1)/30000.,waveform_size),mu_max,kind='cubic')
    interp_fn_sd = interpolate.interp1d(np.linspace(0,(waveform_size-1)/30000.,waveform_size),sd_max,kind='cubic')
    
    interp_x_vals = np.linspace(0,(waveform_size-1)/30000.,waveform_size*125)
    interp_m = interp_fn_m(interp_x_vals)
    interp_sd = interp_fn_sd(interp_x_vals)
    record['t_wvform'] = interp_x_vals
    record['m_wvform'] = interp_m
    record['sd_wvform'] = interp_sd
    
    ax.plot(interp_x_vals,interp_m,color="black",linewidth=2)
    ax.plot(interp_x_vals,interp_m+interp_sd,"-",color="black",linewidth=1)
    ax.plot(interp_x_vals,interp_m-interp_sd,"-",color="black",linewidth=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    fwhm = get_fwhm(interp_m,interp_x_vals, ax=ax, plot=True)
    p2t_ratio,p2t_time = get_peak_trough(interp_m,interp_x_vals, ax=ax, plot=True)
    record['fwhm'] = fwhm
    record['p2t_ratio'] = p2t_ratio
    record['p2t_time'] = p2t_time
    ax.plot(ax.get_xlim(),[0,0,],'--',color='black')
    return record
    
def plot_unit_stability(unit, model_loc, ax1, ax2, record):
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
    
    ax1.plot(spike_time_that[that_choice],fet_x_that[that_choice],'o',color='blue',markersize=5,markeredgecolor='none')
    ax1.plot(spike_time_other[other_choice],fet_x_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')

    
    ax2.plot(fet_y_that[that_choice],fet_x_that[that_choice],'o',color='blue',markersize=5,markeredgecolor='none')
    ax2.plot(fet_y_other[other_choice],fet_x_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')
    
    return record

