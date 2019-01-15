from klusta.kwik.model import KwikModel
import os
import numpy
import matplotlib.pyplot as plt
import types
import pdb
import pickle


def get_cluster_waveforms (kwik_model,cluster_id):
        
    clusters = kwik_model.spike_clusters
    try:
        if not cluster_id in clusters:
            raise ValueError       
    except ValueError:
            print ("Exception: cluster_id (%d) not found !! " % cluster_id)
            return
    
    idx=numpy.argwhere(clusters==cluster_id)
    return kwik_model.all_waveforms[idx]

def isi_violations(spike_train, min_isi, ref_dur):
    # See matlab version for commentary
    isis = numpy.diff(spike_train)
    n_spikes = numpy.sizelen(spike_train)
    num_violations = numpy.sum(isis<ref_dur)
    violation_time = 2*n_spikes*(n_spikes-min_isi)
    total_rate = n_spikes/spike_train[-1]
    violation_rate = num_violations/violation_time
    fp_rate = violation_rate/total_rate
    return fp_rate, num_violations
    # get the .kwik file and make KwikModel
    kwik_file = [f for f in os.listdir(folder) if f.endswith('.kwik')]
    if len(kwik_file)>1 or len(kwik_file)==0:
        RuntimeError('Too many or too few files. Whats happening')
    kwik_file_path = os.path.join(folder,kwik_file[0])
    print("loading kwik file :", kwik_file_path)
    kwik_model = KwikModel(kwik_file_path)
    kwik_model._open_kwik_if_needed()
    
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
        spike_time = kwik_model.spike_samples.astype(numpy.float64)/kwik_model.sample_rate
        spike_id = kwik_model.spike_ids
        cluster_id = kwik_model.spike_clusters
        cluster_ids = kwik_model.cluster_ids
        
        clu = cluster_id
        fet_masks = kwik_model.all_features_masks
        fet = numpy.squeeze(fet_masks[:,:,0])
        mask = numpy.squeeze(fet_masks[:,:,1])
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
            spike_id_idx = numpy.argwhere(cluster_id==clust_num)
            spike_id_that_cluster = spike_id[spike_id_idx]
            unit["spike_ids"] = spike_id_that_cluster
            unit["spike_time"] = spike_time[spike_id_idx]
            
            
            waves = get_cluster_waveforms(kwik_model,clust_num)
            mu_all = numpy.mean(waves[:,:,:],axis=0);
            std_all = numpy.std(waves[:,:,:],axis=0);
            unit["mean_waveform"] = numpy.mean(waves,axis=0)
            unit["std_waveform"] = numpy.std(waves,axis=0)
            
            unit["num_spikes"] = waves.shape[0]
            
            max_ind = numpy.unravel_index(numpy.argmin(mu_all),mu_all.shape)
            max_ch = max_ind[1]
            
            unit["x_loc"] = kwik_model.channel_positions[max_ch,0]
            unit["y_loc"] = kwik_model.channel_positions[max_ch,1]
            unit["quality"] = clu_quality[numpy.argwhere(unique_clu==clust_num)]
            unit["contamination_rate"] =cont_rate[numpy.argwhere(unique_clu==clust_num)]            
                
            plot = False
            if plot:
                waveform_size = waves.shape[1]
                x_scale = 2
                y_scale = 0.5
                fig = plt.figure(figsize=(3,3),dpi=200,facecolor='w',edgecolor='k')
                plt.clf()
                for ch in range (0,waves.shape[2]):                        
                    mu_spike = mu_all[:,ch]
                    std_spike = std_all[:,ch]
                    x_offset = kwik_model.channel_positions[ch,0]
                    y_loc = kwik_model.channel_positions[ch,1]
                    y_offset = y_loc*y_scale
                    x = x_scale*x_offset+range(0,waveform_size)
                    plt.plot(x,0.05*mu_spike+y_offset,color="black",linewidth=2)
                    plt.plot(x,0.05*(mu_spike+std_spike)+y_offset,"--",color="black",linewidth=1)
                    plt.plot(x,0.05*(mu_spike-std_spike)+y_offset,"--",color="black",linewidth=1)
                    if ch==max_ch:
                        plt.text(numpy.mean(x),0.05*numpy.mean(mu_spike)+y_offset,"%r,%r" % (x_offset,y_loc),horizontalalignment='center',verticalalignment='center',color='red')
                    
                plt.show()
                input("Press <ENTER> to continue")
            
            units.append(unit)
    output["units"] = units
    
    kwik_model.close()
    return output

def plot_unit_waveform(data,whichs):
    units = data['spike_records']['units']
    num_units = len(units)
    shank_nos = [x['shank_no'] for x in units]
    cluster_nos = [x['cluster_id'] for x in units]
    for which in whichs:
        #unit = [True if (x==which[0] and y==which[1]) else False for (x,y) in zip(shank_nos,cluster_nos)]
        which_unit = [i for (x,y,i) in zip(shank_nos,cluster_nos,range(0,num_units)) if (x==which[0] and y==which[1])]
        
        if len(which_unit)==1: unit = units[which_unit[0]]
        else: RuntimeError('No way the same unit was detected multiple times...')
        
        mu_all = unit['mean_waveform']
        std_all = unit['std_waveform']
        waveform_size = mu_all.shape[0]
        num_chans = mu_all.shape[1]
        fig = plt.figure(figsize=(3,3),dpi=200,facecolor='w',edgecolor='k')
        plt.clf()
        for ch in range(0,num_chans):
            mu_spike = mu_all[:,ch]
            print(len(mu_spike))
            std_spike = std_all[:,ch]
            x_offset = 40*ch
            x = [40*ch+i for i in range(0,waveform_size)]
            print(len(x))
            plt.plot(x,mu_spike,color="black",linewidth=2)
            plt.plot(x,(mu_spike+std_spike),"-",color="black",linewidth=1)
            plt.plot(x,(mu_spike-std_spike),"-",color="black",linewidth=1)
            
        plt.show()
        input("Press <ENTER> to continue")
        
        pdb.set_trace() 

def plot_unit_ISI(data,whichs):
    units = data['spike_records']['units']
    num_units = len(units)
    shank_nos = [x['shank_no'] for x in units]
    cluster_nos = [x['cluster_id'] for x in units]
    for which in whichs:
        which_unit = [i for (x,y,i) in zip(shank_nos,cluster_nos,range(0,num_units)) if (x==which[0] and y==which[1])]
        if len(which_unit)==1: unit = units[which_unit[0]]
        else: RuntimeError('No way the same unit was detected multiple times...')
        
        fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
        isi = numpy.diff(unit['spike_time'],axis=0)
        histc, binedge = numpy.histogram(isi,range=(0,0.1),bins=100)
        
        plt.clf()
        plt.bar(binedge[0:100]*1000,histc,align='edge',edgecolor='none',color=which[2])
        
        plt.show()
        input("Press <ENTER> to continue")
        
def plot_num_sessions_histogram():
    base = [0,2,4,6,8,10,40]
    behaved = [4,1,4,3,2,2]
    phys = [36,2,7,4,2,7,4,4,1]
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    histc1 = numpy.histogram(phys,bins=base)
    histc2 = numpy.histogram(behaved,bins=base)
    plt.clf()
    plt.bar(base[0:6],histc1[0],align='edge',edgecolor='none',color=(0.,0.,0.),width=1.8)
    plt.bar(base[0:6],histc2[0],bottom=histc1[0],align='edge',edgecolor='none',color=(0.2,0.2,0.2),width=1.8)
    
    plt.ylim((0,6))
    plt.xlim((0,12))
    plt.show()

def get_model(loc):
    kwik_file = [f for f in os.listdir(loc) if f.endswith('.kwik')]
    if len(kwik_file)>1 or len(kwik_file)==0:
        RuntimeError('Too many or too few files. Whats happening')
    kwik_file_path = os.path.join(loc,kwik_file[0])
    print("loading kwik file :", kwik_file_path)
    kwik_model = KwikModel(kwik_file_path)
    kwik_model._open_kwik_if_needed()
    return kwik_model
    
def plot_unit_stability(loc,data,whichs):
    kwik_model = get_model(loc)
    for which in whichs:
        kwik_model.channel_group = which[0]
        kwik_model.clustering = 'main'
        spike_time = kwik_model.spike_samples.astype(numpy.float64)/kwik_model.sample_rate
        spike_id = kwik_model.spike_ids
        cluster_id = kwik_model.spike_clusters
        fet_masks = kwik_model.all_features_masks
        fet = numpy.squeeze(fet_masks[:,:,0])
        
        that_cluster_idx = numpy.argwhere(cluster_id==which[1])
        other_cluster_idx = numpy.argwhere(cluster_id!=which[1])
        
        spike_time_that = spike_time[that_cluster_idx]
        spike_time_other = spike_time[other_cluster_idx]
        
        fet_that = numpy.squeeze(fet[that_cluster_idx,:])
        fet_other = numpy.squeeze(fet[other_cluster_idx,:])
        
        # find the feature dimensions with greatest mean values
        fet_that_mean = numpy.mean(fet_that,axis=0)
        fet_seq = numpy.argsort(-1*numpy.absolute(fet_that_mean))
        
        fet_x_that = fet_that[:,fet_seq[0]]
        fet_y_that = fet_that[:,fet_seq[2]]
        
        fet_x_other = fet_other[:,fet_seq[0]]
        fet_y_other = fet_other[:,fet_seq[2]]
        
        fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
        
        # select 2% from 'that' and 0.1% from other
        that_choice = numpy.random.choice([True,False],size=fet_x_that.shape,p=[0.01,0.99])
        other_choice = numpy.random.choice([True,False],size=fet_x_other.shape,p=[0.01,0.99])
        
        ax1 = fig.add_subplot(221)
        ax1.plot(spike_time_that[that_choice],fet_x_that[that_choice],'o',color=which[2],markersize=5,markeredgecolor='none')
        ax1.plot(spike_time_other[other_choice],fet_x_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')
        
        ax2 = fig.add_subplot(224)
        ax2.plot(fet_y_that[that_choice],spike_time_that[that_choice],'o',color=which[2],markersize=5,markeredgecolor='none')
        ax2.plot(fet_y_other[other_choice],spike_time_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')
        
        ax3 = fig.add_subplot(222)
        ax3.plot(fet_y_that[that_choice],fet_x_that[that_choice],'o',color=which[2],markersize=5,markeredgecolor='none')
        ax3.plot(fet_y_other[other_choice],fet_x_other[other_choice],'o',color=(0.7,0.7,0.7),markersize=2,markeredgecolor='none')
        
        
        plt.savefig('trial.png')
        
        
if __name__=="__main__":
    location = '/home/bsriram/data/DetailsProcessedPhysOnly/bas070_2015-08-17_14-30-11'

    with open(os.path.join(location,'spike_and_trials.pickle'),'rb') as f:
        data = pickle.load(f)
    
    # whichs units as [shank_no,cluster_id,color]
    whichs = []
    whichs.append([0,68,(252./255,176./255,41./255)])
    whichs.append([0,70,(66./255,87./255,40./255)])
    #plot_unit_waveform(data,whichs)
    #plot_unit_ISI(data,whichs)
    #plot_num_sessions_histogram()
    plot_unit_stability(location,data,whichs)