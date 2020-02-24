import os
import time
import numpy as np
import ffmpy


def convert_video(video_file,save_path):
    new_filename = 'eye_trk.mp4'
    new_file_path = os.path.join(save_path,new_filename)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    job = ffmpy.FFmpeg(inputs={video_file:None},outputs={new_file_path:'-vf scale=320:-1 -crf 10'})
    job.run()
    
    return new_file_path


temp = [('g2_3','02172019','02172019_2','R1','RunningNotNamed','plexon','NNX_Poly3'),
 ('g2_3','02182019','02182019_1','R1','RunningNotNamed','plexon','NNX_Poly3'),
 ('g2_3','02152019','02152019_2','R1','RunningNotNamed','plexon','NNX_Poly3'),
 ('g2_2','01302019','01302019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
 ('g2_1','02012019','02012019_1','R1','RunningNotNamed','plexon','NNX_Poly3'),]
videos_path = '/home/bsriram/videos'
sessions_path = '/home/bsriram/data_biogen1'
sessions = [
    ('g2_1','01302019','01302019_1','R1','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','01312019','01312019_3','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_1','01312019','01312019_2','R1','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_2','01312019','01312019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02012019','02012019_2','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02012019','02012019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02022019','02022019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02022019','02022019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_1','02032019','02032019_1','R1','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_2','02032019','02032019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02032019','02032019_2','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('b8','02042019','02042019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02042019','02042019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02052019','02052019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02052019','02052019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02062019','02062019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02062019','02062019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02132019','02132019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02132019','02132019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02142019','02142019_2','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02142019','02142019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_2','02152019','02152019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_5','02152019','02152019_3','R1','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_4','02152019','02152019_3','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02152019','02152019_4','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('b8','02172019','02172019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02182019','02182019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02182019','02182019_2','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_4','02182019','02182019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('b8','02192019','02192019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('b8','02212019','02212019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('b8','02222019','02222019_1','R1','RunningNotNamed','plexon','NNX_4sh_tetX2'),
    ('g2_2','02192019','02192019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_2','02212019','02212019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_2','02222019','02222019_1','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_4','02192019','02192019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_4','02212019','02212019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ('g2_4','02222019','02222019_2','R2','RunningNotNamed','plexon','NNX_Poly3'),
    ]

def move_eye_trk():
    subjs = np.asarray([sess[0] for sess in sessions])
    dates = np.asarray([sess[1] for sess in sessions])
    sess_folders = np.asarray([sess[2] for sess in sessions])
    rigs = np.asarray([sess[3] for sess in sessions])
    runnings = np.asarray([sess[4] for sess in sessions])
    hstages = np.asarray([sess[5] for sess in sessions])
    elecs = np.asarray([sess[6] for sess in sessions])
    for ii,sess in enumerate(sessions):
        subj,date,sess_folder,rig,running,hs,elec = sess
        sess_folder_path = os.path.join(sessions_path,subj+'_'+date)
     
        # any other sibject in that session?
        num_subj_in_session = np.sum(sess_folders==sess_folder)
        if rig=='R1':avi_ending='_1.AVI'
        if rig=='R2':avi_ending='_2.AVI'
        avi_file = []
        if num_subj_in_session==1:
            # try searching for startswith(sess_folder) and endswith(_1.AVI)
            avi_file = [f for f in os.listdir(videos_path) if f.startswith(sess_folder+'_') and f.endswith(avi_ending)]
            # if not found, then some dont have the _rig.AVI
            if not avi_file:
                avi_file = [f for f in os.listdir(videos_path) if f.startswith(sess_folder+'_') and f.endswith('1.AVI')]
            # print(subj,'\t:',sess_folder,'\t:',avi_file)
        else:
            if rig=='R1':avi_ending='_1.AVI'
            if rig=='R2':avi_ending='_2.AVI'
            avi_file = [f for f in os.listdir(videos_path) if f.startswith(sess_folder+'_') and f.endswith(avi_ending)]
            # print(subj,'\t:',sess_folder,'\t:',avi_file)
            
        if len(avi_file)==0:
            print(subj,'\t:',sess_folder,'\t:','no eye tracking found. continuing')
            continue
        elif len(avi_file)>1:
            print(subj,'\t:',sess_folder,'\t:','too many tracking found')
        else:
            video_file_path = os.path.join(videos_path,avi_file[0])
            save_path = os.path.join(sess_folder_path,'eye_tracking')
            print(subj,'\t:',sess_folder,'\t:','save '+video_file_path+' in '+save_path)
            convert_video(video_file_path,save_path)

def make_analyze_script():
    for sess in sessions:
        subj,date,sess_folder,rig,running,hs,elec = sess
        sess_eyetrk_folder_path = os.path.join(sessions_path,subj+'_'+date,'eye_tracking')
        with open(os.path.join(sess_eyetrk_folder_path,'analyze_videos.py'),'w') as f:
            f.write("""import deeplabcut
import tensorflow as tf
import os
import time
path_config_file = '/home/bsriram/code/eye-trk/examples/Eye-Tracking-BAS-2019-02-17/config.yaml'
os.environ["DLClight"]="True"
videos = ['{0}']
t_start = time.time()
print('analyzing')
deeplabcut.analyze_videos(path_config_file,videos)
print('creating labeled_video')
deeplabcut.create_labeled_video(path_config_file,videos)
print("It took ",time.time()-t_start)""".format(os.path.join(sess_eyetrk_folder_path,'eye_trk.mp4')))
        print('wrote into '+os.path.join(sess_eyetrk_folder_path,'analyze_videos.py'))

def make_qsub_script():
    for sess in sessions:
        subj,date,sess_folder,rig,running,hs,elec = sess
        sess_eyetrk_folder_path = os.path.join(sessions_path,subj+'_'+date,'eye_tracking')
        with open(os.path.join(sess_eyetrk_folder_path,'submit.qsub'),'w') as f:
            f.write("""#!/bin/bash
#$ -N {0}
#$ -l h_rt=24:00:00 
#$ -q long.q 
#$ -wd {1}
#$ -j no 
#$ -M balaji.sriram@biogen.com
#$ -m be 
#$ -e error.log 
#$ -o output.log 
#$ -pe openmpi-fillup 1

##########################################################################
# Start your script here #
##########################################################################
# Load the modules you need.
source /home/bsriram/miniconda3/bin/activate dlc_cpu
# Run some commands.
python analyze_videos.py

# Exit successfully.
exit 0
""".format(subj+'_'+date,sess_eyetrk_folder_path))
        print('wrote into '+os.path.join(sess_eyetrk_folder_path,'submit.qsub'))


if __name__=='__main__':
    # move_eye_trk()
    make_analyze_script()
    make_qsub_script()