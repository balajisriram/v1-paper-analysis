from klusta.kwik.model import KwikModel
#from .ClusterQuality import cluster_quality_all
import os
import numpy
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.io
import importlib.machinery
import types
import pdb

loader = importlib.machinery.SourceFileLoader('ClusterQuality',r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\Util\ClusterQuality.py')
# loader = importlib.machinery.SourceFileLoader('ClusterQuality','/camhpc/home/bsriram/v1paper/v1-paper-analysis/Util/ClusterQuality.py')
CQMod = types.ModuleType(loader.name)
loader.exec_module(CQMod)

def get_cluster_waveforms (kwik_model,cluster_id):
        
    clusters = kwik_model.spike_clusters
    try:
        if not cluster_id in clusters:
            raise ValueError       
    except ValueError:
            print ("Exception: cluster_id (%d) not found !! " % cluster_id)
            return
    
    idx=numpy.argwhere(clusters==cluster_id)
    #print(kwik_model.all_waveforms[idx])
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


def get_cluster_details(folder):
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
        try:
            unique_clu, clu_quality, cont_rate = CQMod.cluster_quality_all(clu, fet, mask, fet_N)
            quality_success = True
        except:
            quality_success = False
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
            if quality_success:
                unit["quality"] = clu_quality[numpy.argwhere(unique_clu==clust_num)]
                unit["contamination_rate"] =cont_rate[numpy.argwhere(unique_clu==clust_num)]            
            else:
                unit["quality"] = numpy.nan
                unit["contamination_rate"] = numpy.nan
                
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

    
if __name__=="__main__":
    folders = ["m325_2017-08-10_13-14-51"]
    basefolder = "C:\\Users\\bsriram\\Desktop\\Inspected"
    savefolder = "C:\\Users\\bsriram\\Desktop\\LocalSave"
    if True:        
        for folder in folders:
            req_folder = os.path.join(basefolder,folder)
            deets = get_cluster_details(req_folder)
            print(deets)
            save_loc = os.path.join(savefolder,folder+'.mat')
            scipy.io.savemat(save_loc,dict(details=deets))       
    elif False:
        x = numpy.random.normal(0,1,[100,12])
        y = numpy.random.normal(17,1,[1000,12])
        
        uq,cr = cluster_quality_core(x,y)
        print(uq)
        print(cr)
    else:
        x = numpy.random.normal(0,1,[100,12])
        y = numpy.random.normal(17,1,[1000,12])
        
        cov_x_inv = numpy.linalg.inv(numpy.cov(x,rowvar=False))
        cdist = scipy.spatial.distance.cdist
        md = cdist(y,x,'mahalanobis',cov_x_inv)
        md_lin = numpy.mean(md,axis=1)
        
        scipy.io.savemat('D:\\sample_feature.mat',dict(x=x,y=y,md=md,md_lin=md_lin))
        
        plt.plot(md_lin)
        plt.show()
        