import pickle
import pandas as pd     
import numpy as np
import pprint
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt, resample

ppr = pprint.PrettyPrinter(indent=2).pprint


def plot_eye_tracking_and_running(loc):
    # fig=plt.figure(facecolor='w', edgecolor='k')
    
    with open(os.path.join(loc,'event_data.pickle'),'rb') as f:
        event_data = pickle.load(f, encoding='latin1')
        
    ## eye tracking
    print('getting pupil')
    eye_trk_file = [f for f in os.listdir(os.path.join(loc,'eye_tracking')) if f.endswith('.h5')]
    data = pd.read_hdf(os.path.join(loc,'eye_tracking',eye_trk_file[0]))
    #ppr(data.head())
    # get the positions
    h1_x = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'h1', 'x')]
    h1_y = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'h1', 'y')]
    h2_x = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'h2', 'x')]
    h2_y = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'h2', 'y')]
    h_d = np.power(np.power((h1_x-h2_x),2)+np.power((h1_y-h2_y),2),0.5)

    v1_x = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'v1', 'x')]
    v1_y = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'v1', 'y')]
    v2_x = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'v2', 'x')]
    v2_y = data[('DeepCut_resnet50_Eye-TrackingFeb17shuffle1_135000', 'v2', 'y')]
    v_d = np.power(np.power((v1_x-v2_x),2)+np.power((v1_y-v2_y),2),0.5) 
    
    pupil_size = h_d*v_d
    pupil_size = pupil_size.values
    pupil_t = np.arange(len(pupil_size))*1/30.
    pupil_t_max = np.max(pupil_t)
    t_filter = np.squeeze(np.argwhere(np.bitwise_and(pupil_t>180,pupil_t<pupil_t_max-180)))
    pupil_t = pupil_t[t_filter]
    pupil_size = pupil_size[t_filter]
    
    pupil_size_smooth = medfilt(pupil_size,31)
    pupil_z = (pupil_size_smooth-np.mean(pupil_size_smooth))/np.std(pupil_size_smooth)
    #pupil_z = pupil_z[0:1000]
    pupil_z[pupil_z>2] = 2
    pupil_z_hi = pupil_z
    pupil_z_hi[pupil_z_hi<0] = np.nan
    
    # running
    # print('getting running')
    # running_speed = np.fromfile(os.path.join(loc,'running_speed.np'))
    # running_speed = np.abs(running_speed-2.5)
    # running_speed = resample(running_speed,np.int(len(running_speed)/100))
    # running_rate = np.fromfile(os.path.join(loc,'running_samplerate.np'))
    # running_ts = np.fromfile(os.path.join(loc,'running_ts.np'))
    # running_t = running_ts+np.arange(len(running_speed))*100/running_rate
    # running_speed_hi = running_speed
    # running_speed_hi[running_speed_hi<0.2] = np.nan
    # running_speed_smooth = running_speed
    
    
    # import pdb
    # pdb.set_trace()
    
    
    plt.subplot(3,1,1)
    plt.plot(pupil_t,pupil_z,'k',alpha=0.5)
    plt.plot(pupil_t,pupil_z_hi,'g',alpha=0.5)
    plt.savefig('Running.svg',transparent=True,facecolor=None,edgecolor=None)
    # plt.subplot(3,1,2)
    # plt.plot(running_t,running_speed,'k')
    #plt.plot(running_t,running_speed_hi,'g')
    # plt.plot(pupil_t[locs_hi],pupil_z[locs_hi],'r.')
    
    
    plt.show()



if __name__=='__main__':
    loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\EyeTracked\g2_2_02062019'
    plot_eye_tracking_and_running(loc)
