import os
import sys
sys.path.append(r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\Figure3')
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle
import pprint
import scipy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from util import get_base_depth,get_unit_depth,get_subject_from_session,get_OSI, get_vector_sum,get_orientation_tuning_stepname,get_frame_channel,get_unit_id
ppr = pprint.PrettyPrinter(indent=2)
        
def get_all_orientation_tuning(location):
    unit_details_across_sessions = {}
    
    for sess in os.listdir(location):
        print(sess)    
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        try:
            stepname,durations = get_orientation_tuning_stepname(sess)
        except:
            pdb.set_trace()
        framechan,shift = get_frame_channel(sess)
        
        if not stepname: 
            print(sess,'why no stepname???')
            continue
        
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
        
        unit_responses_that_session = {}
        unit_responses_that_session['session'] = sess
        unit_responses_that_session['subject'] = get_subject_from_session(sess)
        
        for i,unit in enumerate(data['spike_records']['units']):
            if not unit['manual_quality'] in ['good','mua']: continue
            unit_details = {}
            uid = get_unit_id(sess,unit['shank_no'],unit['cluster_id'])
            unit_details['uid'] = uid
            unit_details['manual_quality'] = unit['manual_quality']
            unit_details['unit_depth'] = get_unit_depth(sess,unit['y_loc'])
            spike_time = numpy.squeeze(numpy.asarray(unit['spike_time']))
            
            
            spike_raster = {}
            for trial_number in trial_numbers:
                frame_start_time = frame_start_index[trial_numbers==trial_number][0]/30000
                frame_end_time = frame_end_index[trial_numbers==trial_number][0]/30000
                
                spikes_that_trial = spike_time[numpy.bitwise_and(spike_time>frame_start_time,spike_time<frame_end_time)]-frame_start_time
                spike_raster[trial_number] = spikes_that_trial
            unit_details['spike_raster'] = spike_raster
            
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
                unit_details['or_tuning'] = None
                unit_details['osi'] = None
                unit_details['vector_sum'] = None
                unit_details['trial_jackknife'] = None
                unit_details['osi_jackknife'] = None
                unit_details['vector_sum_jackknife'] = None
            
            
            unit_details_across_sessions[uid] = unit_details
        
        print('done session')
    return unit_details_across_sessions

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

def get_all_OSIs(o_details,u_details):
    uid = numpy.asarray(u_details['unit_id'])
    uq = numpy.asarray(u_details['unit_quality'])
    depth = numpy.asarray(u_details['unit_depth'])
    fr = numpy.asarray(u_details['unit_firingrate'])
    is_replica = numpy.asarray(u_details['unit_is_replica'])
    
    uid = uid[~is_replica]
    uq = uq[~is_replica]
    depth = depth[~is_replica]
    fr = fr[~is_replica]
    
    ids = []
    osi = []
    osi_ang = []
    osi_jk_m = []
    osi_jk_sd = []
    osi_jk_ang_m = []
    osi_jk_ang_sd = []
    osi_jk_n = []
    vs_ang = []
    vs_jk_ang_m = []
    vs_jk_ang_sd = []
    depths = []
    frs = []
    qualities = []
    
    for i,id in enumerate(uid):
        try:
            o_det = o_details[id]
        except KeyError:
            continue
        
        if not o_det['osi']:
            continue
        ids.append(id)
        depths.append(depth[i])
        frs.append(fr[i])
        qualities.append(uq[i])
        
        # osi
        osi.append(o_det['osi'][0])
        osi_ang.append(o_det['osi'][1])
        
        # osi jackknife
        osi_jk_vals = [jk_val[0] for jk_val in o_det['osi_jackknife'] if jk_val[0] is not None] 
        osi_jk_ang_vals = [jk_val[1] for jk_val in o_det['osi_jackknife'] if jk_val[1] is not None]     
        osi_jk_m.append(numpy.mean(osi_jk_vals))
        osi_jk_sd.append(numpy.std(osi_jk_vals))
        osi_jk_ang_m.append(numpy.mean(osi_jk_ang_vals))
        osi_jk_ang_sd.append(numpy.std(osi_jk_ang_vals))
        osi_jk_n.append(len(osi_jk_vals))
        
        # vs
        vs_ang.append(o_det['vector_sum'][1])
        
        # vs_jk
        vs_jk_ang_vals = [jk_val[1] for jk_val in o_det['vector_sum_jackknife'] if jk_val[1] is not None]
        vs_jk_ang_m.append(numpy.mean(vs_jk_ang_vals))
        vs_jk_ang_sd.append(numpy.std(vs_jk_ang_vals))
        
    osi_details = {}
    osi_details['ids'] = ids
    osi_details['osi'] = osi
    osi_details['osi_ang'] = osi_ang
    osi_details['osi_jk_m'] = osi_jk_m
    osi_details['osi_jk_sd'] = osi_jk_sd
    osi_details['osi_jk_ang_m'] = osi_jk_ang_m
    osi_details['osi_jk_ang_sd'] = osi_jk_ang_sd
    osi_details['osi_jk_n'] = osi_jk_n
    osi_details['vs_ang'] = vs_ang
    osi_details['vs_jk_ang_m'] = vs_jk_ang_m
    osi_details['vs_jk_ang_sd'] = vs_jk_ang_sd
    osi_details['depths'] = depths
    osi_details['frs'] = frs
    osi_details['qualities'] = qualities
    
    return osi_details
    
if __name__=="__main__":
    # ##ORIENTATION DETAILS FOR PHYS
    # location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly'
    # unit_details = get_all_orientation_tuning(location)
    
    # with open('Orientation_Details_PhysOnly.pkl','wb') as f:
       # pickle.dump(unit_details,f)
       
    # ##ORIENTATION DETAILS FOR BEHAVED
    # location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedBehaved'
    # unit_details = get_all_orientation_tuning(location)
    
    # with open('Orientation_Details_Behaved.pkl','wb') as f:
        # pickle.dump(unit_details,f)   
    
    ## PLOT THE ORIENTATION PREFERENCE
    loc = r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis'
    with open(os.path.join(loc,'Orientation_Details_PhysOnly.pkl'),'rb') as f:
        orientation_details = pickle.load(f)
        
    with open(os.path.join(loc,'UnitDetailsAfterDereplication_PhysOnly.pkl'),'rb') as f:
        unit_details = pickle.load(f)
    
    osi_details = get_all_OSIs(orientation_details,unit_details)
    
    with open(os.path.join(loc,'CompiledOrientationDetails_PhysOnly.pkl'),'wb') as f:
        pickle.dump(osi_details,f)
        
    
    osi_jk_m = numpy.asarray(osi_details['osi_jk_m'])
    osi_jk_sd = numpy.asarray(osi_details['osi_jk_sd'])
    vs_jk_ang_m = numpy.asarray(osi_details['vs_jk_ang_m'])
    vs_jk_ang_sd = numpy.asarray(osi_details['vs_jk_ang_sd'])
    frs = numpy.asarray(osi_details['frs'])
    depths = numpy.asarray(osi_details['depths'])
    qualities = numpy.asarray(osi_details['qualities'])
    
    pdb.set_trace()
    goods = ~(numpy.isnan(osi_jk_m) | numpy.isnan(vs_jk_ang_m) | (vs_jk_ang_sd>22.5))
    sus = qualities=='good'
    mua = qualities=='mua'
    
    fig = plt.figure(facecolor='None',edgecolor='None')
    fig.suptitle('Orientation Tuning of V1 neurons')
    
    # ax1::orientation tuning
    ax1 = plt.subplot(321)
    ax1.scatter(osi_jk_m[goods & sus],100-depths[goods & sus],marker='o',s=30,alpha=0.5,facecolor='b',edgecolor='none')
    ax1.scatter(osi_jk_m[goods & mua],100-depths[goods & mua],marker='o',s=10,facecolor='none',edgecolor='b')
    plt.title('Distribution of OSIs by depth')
    plt.xlabel('OSI')
    plt.ylabel('depth (um)')
    plt.xlim((-0.01,1.01))
    
    # ax2
    ax2 = plt.subplot(325)
    ax2.scatter(osi_jk_m[goods & sus],frs[goods & sus],marker='o',s=30,alpha=0.5,facecolor='b',edgecolor='none')
    ax2.scatter(osi_jk_m[goods & mua],frs[goods & mua],marker='o',s=10,facecolor='none',edgecolor='b')
    plt.title('OSI vs Firing rate')
    plt.xlabel('OSI')
    plt.ylabel('Firing Rate (Hz)')
    
    x = sm.add_constant(osi_jk_m[goods])
    y = frs[goods]
    
    model = sm.OLS(y,x)
    fitted = model.fit()
    print(fitted.summary())
    st, data, ss2 = summary_table(fitted, alpha=0.05)
    fittedvalues = data[:,2]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    ax2.plot(x, fittedvalues, '-', lw=2)
    ax2.plot(x[:-10], predict_mean_ci_low[:-10], 'r--', lw=1)
    ax2.plot(x[:-100], predict_mean_ci_upp[:-100], 'g--', lw=1)
    plt.ylim((0,30));plt.xlim((-0.01,1.1))
    
    # ax3 :  orientation preference
    ax3 = plt.subplot(322)
    degree_vals = numpy.degrees(vs_jk_ang_m[goods])
    hist_prob, hist_edge, hist_patches = ax3.hist(degree_vals,bins=30,normed=True)
    (d,p) = scipy.stats.ks_2samp(degree_vals,(numpy.random.rand(*degree_vals.shape)*180)-90)
    print('KS-test on orientation preference::',p)
    print(p)
    print(d)
    plt.xticks([-90,-45,0,45,90],['$-\pi/2$','$-\pi/4$','$0$','$\pi/4$','$\pi/2$'])
    plt.yticks([0.01],['0.01'])
    plt.xlim((-91,91))
    plt.title('Distribution of preferred angle')
    plt.xlabel('Preferred angle')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    # ax3.scatter(numpy.degrees(vs_jk_ang_m[goods & sus]),100-depths[goods & sus],marker='o',s=30,alpha=0.5,facecolor='b',edgecolor='none')
    # ax3.scatter(numpy.degrees(vs_jk_ang_m[goods & mua]),100-depths[goods & mua],marker='o',s=10,facecolor='none',edgecolor='b')
    
    # ax4 boxplots
    ax4 = plt.subplot(323)
    superficial = (100-depths)>-400
    deep = (100-depths)<-400
    ax4.boxplot([osi_jk_m[goods & deep],osi_jk_m[goods & superficial],],1,'rs',0)
    plt.xlim((-0.01,1.01))
    plt.title('Deep vs Superficial OSI')
    plt.xlabel('OSI')
    plt.yticks([1,2],['Deep','Superficial']) 
    
    ## STATISTICAL TESTS
    
    print('Deep OSI::',numpy.mean(osi_jk_m[goods & deep]))
    print('Superf OSI::',numpy.mean(osi_jk_m[goods& superficial]))
    (t,p) = scipy.stats.ttest_ind(osi_jk_m[goods & deep],osi_jk_m[goods& superficial],equal_var=False)
    print('prob ttest=',p)
    
    (d,p) = scipy.stats.ks_2samp(osi_jk_m[goods & deep],osi_jk_m[goods& superficial])
    print('prob kstest=',p)
    
    # ax5 high OSI neurons
    which_hi_osi = goods & (osi_jk_m>0.95)
    degree_vals_hi_osi = numpy.degrees(vs_jk_ang_m[which_hi_osi])
    
    # ks test between orientation preference of the high osi and the others
    (d,p) = scipy.stats.ks_2samp(degree_vals,degree_vals_hi_osi)
    print('KS-test on orientation preference for high OSI::',p)
    print(p)
    print(d)
    
    
    (t,p) = scipy.stats.mannwhitneyu(frs[which_hi_osi],frs[~which_hi_osi])
    print('mean_fr_highOSI:',numpy.mean(frs[which_hi_osi]))
    print('mean_fr_nothighOSI:',numpy.mean(frs[~which_hi_osi]))
    print('mean_fr:',numpy.mean(frs))
    print('prob high_osi=',p)
    n = 0
    o_pref_sd = []
    for unit_num in numpy.argwhere(which_hi_osi):
        n = n+1
        o_pref_sd.append(vs_jk_ang_sd[unit_num])
        # print('OSI_m:',osi_jk_m[unit_num],'\t OSI_sd:',osi_jk_sd[unit_num],'\t ang_m:',vs_jk_ang_m[unit_num],'\t ang_sd:',vs_jk_ang_sd[unit_num])
        # print('ang_m:',numpy.degrees(vs_jk_ang_m[unit_num]),'\tang_sd:',numpy.degrees(vs_jk_ang_sd[unit_num]))
    print('n with high OSI::',n)
    print('median SD on or preference',numpy.degrees(numpy.median(numpy.asarray(o_pref_sd))))
    plt.show()
    
