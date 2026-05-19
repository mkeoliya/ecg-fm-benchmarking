import resampy
import pandas as pd
import numpy as np

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}
channel_stoi_canonical = {"i": 0, "ii": 1, "iii":2, "avr":3, "avl":4, "avf":5, "v1":6, "v2":7, "v3":8, "v4":9, "v5":10, "v6":11, "vx":12, "vy":13, "vz":14}

def fix_nans_and_clip(signal,clip_amp=3):
    for i in range(signal.shape[1]):
        tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
        signal[:,i]= np.clip(tmp,a_max=clip_amp, a_min=-clip_amp) if clip_amp>0 else tmp
    return signal

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                #if(skimage_transform):
                #    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                #else:
                #    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
                data[:,channel_stoi[cl]] = resampy.resample(sigbufs[:,i], fs, target_fs).astype(np.float32)
    else:
        #if(skimage_transform):
        #    data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        #else:
        #    data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data