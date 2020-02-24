import deeplabcut
import tensorflow as tf
import os
import time

path_config_file = '/home/bsriram/code/eye-trk/examples/Eye-Tracking-BAS-2019-02-17/config.yaml'
os.environ["DLClight"]="True"
videos = ['/home/bsriram/dlc_cpu_test_1/test.mp4']
t_start = time.time()
print('analyzing')
deeplabcut.analyze_videos(path_config_file,videos)
print('creating labeled_video')
deeplabcut.create_labeled_video(path_config_file,videos)
print("It took ",time.time()-t_start)
