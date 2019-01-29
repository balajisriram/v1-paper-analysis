from klusta.kwik.model import KwikModel
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import types
import pdb
import pickle
import pandas
from scipy.stats import kde
from Util.ClusterQuality import cluster_quality_core
from Util.util import get_frame_channel
# from fastkde import fastKDE
import matplotlib as mpl



label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

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

def get_fwhm(wvform, t, ax = None):    
    # find the min value and index of interp functionm
    wvform_min = np.min(wvform)
    half_min = wvform_min/2
    wvform_min_idx = np.argmin(wvform)
    
    idx_left = 0
    for i in range(wvform_min_idx,-1,-1):
        if wvform[i]>half_min:
           idx_left = i
           break
    
    idx_right = wvform.size-1
    for i in range(wvform_min_idx,wvform.size):
        if wvform[i]>half_min:
            idx_right = i
            break
    # print('min_idx::%d, left:: %d, right %d' %(wvform_min_idx, idx_left, idx_right))
    spike_width = t[idx_right]-t[idx_left]
    if ax:
        pass
        #ax.plot([t[idx_left],t[idx_right]],[half_min,half_min],'-',color='blue')
    return spike_width

def get_peak_trough(wvform, t, ax=None):
    # find the min value and index of interp functionm
    wvform_min = np.min(wvform)
    half_min = wvform_min/2
    wvform_min_idx = np.argmin(wvform)
    wvform_max_idx = np.argmax(wvform[wvform_min_idx:])+ wvform_min_idx
    
    p2t_ratio = -wvform[wvform_max_idx]/wvform[wvform_min_idx]
    p2t_time = t[wvform_max_idx]-t[wvform_min_idx]
    
    if ax:
        pass
        # ax.plot(t[wvform_max_idx],wvform[wvform_max_idx],'gx')
        # ax.plot(t[wvform_min_idx],wvform[wvform_min_idx],'rx')
    
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
    
    max_idx = np.argmin(interp_m)
    record['peak_snr_wvform'] = interp_m[max_idx]/interp_sd[max_idx]
    
    # ax.plot(interp_x_vals,interp_m,color="black",linewidth=2)
    # ax.plot(interp_x_vals,interp_m+interp_sd,"-",color="black",linewidth=1)
    # ax.plot(interp_x_vals,interp_m-interp_sd,"-",color="black",linewidth=1)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    fwhm = get_fwhm(interp_m,interp_x_vals, ax=ax)
    fwhm = 1000.*fwhm # in ms
    p2t_ratio,p2t_time = get_peak_trough(interp_m,interp_x_vals, ax=ax)
    p2t_time = 1000.* p2t_time # in ms
    record['fwhm'] = fwhm
    record['p2t_ratio'] = p2t_ratio
    record['p2t_time'] = p2t_time
    # ax.plot(ax.get_xlim(),[0,0,],'--',color='black')
    # ax.text(ax.get_xlim()[1],ax.get_ylim()[0],'fwhm=%2.3f\np2tT=%2.3f\np2tR=%2.3f\nsnr=%2.3f'%(fwhm,p2t_time,p2t_ratio,record['peak_snr_wvform']),
            # horizontalalignment='right',verticalalignment='bottom',fontsize=5)
    
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
            
def get_or_tuning_dict(spike_raster,trial_numbers,orientations):
    or_tuning = {'orientation':[],'m_rate':[],'std_rate':[],'n_trials':[]}
    spikes_found = False
    for orn in numpy.unique(orientations):
        # find those trials and find mean and std 
        trs = trial_numbers[orientations==orn]
        spike_nums_that_or = [numpy.size(spike_raster[tr]) for tr in trs]
        sp_mean_that_or = numpy.mean(spike_nums_that_or)
        if sp_mean_that_or != 0:
            spikes_found = True
        sp_std_that_or = numpy.std(spike_nums_that_or)
        or_tuning['orientation'].append(orn)
        or_tuning['n_trials'].append(numpy.size(trs))
        or_tuning['m_rate'].append(sp_mean_that_or)
        or_tuning['std_rate'].append(sp_std_that_or)
    if not spikes_found:
        print('Non spikes in orientation tuning for that unit')
    return(or_tuning, spikes_found)        
    
def get_or_tuning(location, sess, unit):
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
    
    trial_numbers = numpy.asarray(data['trial_records']['trial_number'])
    step_names = numpy.asarray(data['trial_records']['step_name'])
    contrasts = numpy.asarray(data['trial_records']['contrast'])
    max_durations = numpy.asarray(data['trial_records']['max_duration'])
    phases = numpy.asarray(data['trial_records']['phase'])
    orientations = numpy.asarray(data['trial_records']['orientation'])
    
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
            
    frame_start_index = numpy.asarray(frame_start_index)
    frame_end_index = numpy.asarray(frame_end_index)
        
    spike_time = numpy.squeeze(numpy.asarray(unit['spike_time']))
    spike_raster = {}
    for trial_number in trial_numbers:
        frame_start_time = frame_start_index[trial_numbers==trial_number][0]/30000
        frame_end_time = frame_end_index[trial_numbers==trial_number][0]/30000
        
        spikes_that_trial = spike_time[numpy.bitwise_and(spike_time>frame_start_time,spike_time<frame_end_time)]-frame_start_time
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
            idx = numpy.where(numpy.logical_not(which_tr))[0]
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