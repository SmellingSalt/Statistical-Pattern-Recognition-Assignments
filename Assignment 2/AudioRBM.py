#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:21:50 2020

@author: sa1
"""

import sounddevice as sd
import numpy as np
import my_functions as mf
#%% AUDIO
import os
from glob import glob
import librosa
#%%
def train_song(path_to_file):
    """This functions yeilds each wav file in the dataset
    if sample_rate_only==True, then it only gives a scalar sample rate"""
    
    #Go to the path where the dataset is stored
    wav_data_path=os.path.relpath(os.path.join(path_to_file))
    wav_data_files_list=glob(os.path.join(wav_data_path, "*.wav"))
    #Get the audio in terms of frames of length 1000

    for wav_file in wav_data_files_list:    
        data, sampling_rate = librosa.load(wav_file)
        frame_size=sampling_rate*5     # roughtly 45ms frames at 22kHz sampling rate
        hop_length=int(frame_size/2)  
        print("Getting "+wav_file+"\n")                
        frames = librosa.util.frame(data, frame_length=frame_size, hop_length=hop_length)
        
        #Normalizing to value between 0 and 1
        frames=frames+1
        frames=frames/frames.max()
        yield frames.T

#%% SEED A SONG
def seed_a_song(sample_length,path_to_file):
    """This functions returns a portion of music from the dataset to seed into
    reconstructing whatever the RBM has learned"""
    
    #Go to the path where the dataset is stored
    # wav_data_path=os.path.relpath(os.path.join('MusicDatasets/ShortPiano/WAVFILES'))
    wav_data_path=os.path.relpath(os.path.join(path_to_file))    
    wav_data_files_list=glob(os.path.join(wav_data_path, "*.wav"))
    #Get the audio in terms of frames of length 1000
    how_many_files=len(wav_data_files_list)
    file_index=np.random.randint(0,high=how_many_files)
    wav_file=wav_data_files_list[file_index]
    
    print("Seeding file "+wav_file+"\n")
    data, sampling_rate = librosa.load(wav_file)
    
    start_point=np.random.randint(0,(len(data)-sample_length))
    end_point=start_point+sample_length    
    
    return data[start_point:end_point]
        
    
        
    
#%% RBM TRAINING
num_hidden_units=500
learn_rate=0.01
epochs=15
k=1
batchsize=1
i=0

#%% TRAINING RBM
path_to_piano='MusicDatasets/ShortPiano/WAVFILES/Em_120bpm'
path_to_violin='MusicDatasets/ShortViolin'
w_histp=[]

print("Training for Piano \n")
for song in train_song(path_to_piano):
    #song is an iterator, coming thanks to the yeild instruction
    i=i+1
    if i==1:
        [wp,ap,bp]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize)
    else:
        [wp,ap,bp]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize,w=wp,a=ap,b=bp)
        
    w_histp.append([wp,ap,bp])

w_histv=[]
i=0
print("Training for Violin \n")
for song in train_song(path_to_violin):
    #song is an iterator, coming thanks to the yeild instruction
    i=i+1
    if i==1:
        [wv,av,bv]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize)
    else:
        [wv,av,bv]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize,w=wv,a=av,b=bv)
        
    w_histv.append([wv,av,bv])
#%% CLASSIFICATION
sample_rate=22050
sample_length=sample_rate*5

test_sample=[]

test_set_length=100
test_label=np.zeros(test_set_length)
for i in range(test_set_length):
    if np.random.randint(0,2)==0:#Choose piano sample
        path_to_file=path_to_piano
    else: #Choose a violin sample
        path_to_file=path_to_violin
        test_label[i]=1
   
    temp=seed_a_song(sample_length,path_to_file)
    # temp=temp*2-1 #Scale to -1 to 1
    test_sample.append(temp)

#%% TRY DIFFERENT WEIGHTS
accuracy=[]
for v in range(len(w_histv)): #Which weight to use to test
    for p in range(len(w_histp)):
        print("testing violin weight " ,v," and piano weight ", p)
        wv=w_histv[v][0]
        wp=w_histp[p][0]
        
        av=w_histv[v][1]
        ap=w_histp[p][1]
        
        bv=w_histv[v][2]
        bp=w_histp[p][2]
        prediction=[]
        i=0
        #1 for violin, 0 for piano
        for sample in test_sample:   
            i+=1
                # seed=np.random.uniform(-1,1,size=sample_rate*5)            
            tempv=mf.reconstruct(sample, wv, av, bv)
            tempp=mf.reconstruct(sample, wp, ap, bp)
            if (np.linalg.norm(tempv-sample)>=np.linalg.norm(tempp-sample)):
                prediction.append(1)
            else:
                prediction.append(0)
        
        prediction=np.asarray(prediction)
        prediction=np.squeeze(prediction)
        misses=prediction[prediction!=test_label]
        
        temp_accuracy=1-len(misses)/len(prediction)
        accuracy.append([temp_accuracy,p,v])
        
accuracy=np.asarray(accuracy)

print(max(accuracy[:,0]))
#%% AUDIO PLAYBACK 
# import time
# n=0 #Which weight to use to test
# for sample in test_sample:
#     for i in range(2):
#         seed=sample
#         # seed=np.random.uniform(-1,1,size=sample_rate*5)
#         tempv=mf.reconstruct(seed, wv, av, bv)
#         audio_playback=np.asarray(tempv)
#         audio_playback=np.squeeze(audio_playback)
#         audio_playback=np.reshape(audio_playback,audio_playback.shape[0]*audio_playback.shape[1])   
#     #Rescaling it between -1 and 1
#         audio_playback=audio_playback*2-1
#         sd.play(audio_playback,samplerate=sample_rate)  
#         time.sleep(5.1)


