#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:21:50 2020

@author: sa1
"""

import sounddevice as sd
import numpy as np
import my_functions as mf
        
#%%
num_hidden_units=100
learn_rate=0.01
epochs=10
k=1
batchsize=30
i=0

#%% TRAINING RBM
w_hist=[]
for song in mf.train_song():
    #mf.song is an iterator, coming thanks to the yeild instruction
    i=i+1
    print("\n Got audio file",i," \n")
    if i==1:
        [w,a,b]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize)
    else:
        [w,a,b]=mf.rbm(song, num_hidden_units, learn_rate, epochs, k,batchsize,w=w,a=a,b=b)
        
    w_hist.append([w,a,b])
#%%
sample_rate=22050
frame_length=sample_rate*5
# audio_playback=np.zeros(sample_rate*10)
audio_playback=[]

sample_length=sample_rate*5
for i in range(10):
    seed=mf.seed_a_song(sample_length)
    temp=mf.reconstruct(seed, w, a, b)
    audio_playback.append(temp)
audio_playback=np.asarray(audio_playback)
audio_playback=np.squeeze(audio_playback)
audio_playback=np.reshape(audio_playback,audio_playback.shape[0]*audio_playback.shape[1])    
sd.play(audio_playback,samplerate=sample_rate)

#%% TRY DIFFERENT WEIGHTS
for i in w_hist:
    
    sd.play(seed,samplerate=sample_rate)
    w=i[0]
    a=i[1]    
    b=i[2]
    sd.play(audio_playback,samplerate=sample_rate)
    print("Next")
#%%

# data1 = mf.reconstruct(song[0,:], w, a, b)
# data2 = mf.sample_hidden(song[0,:], w, b)    