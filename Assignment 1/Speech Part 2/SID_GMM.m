% clc;
% clear all;
% close all;
addpath(genpath('vlfeat'))
datapath='/home/sysad/Desktop/akshay/Gitrepos/Statistical-Pattern-Recognition-Assignments/Assignment 1/Speech Part 2/Data';
%% switch

sw=2;



Nomfccs=14;Nbands=22;Framesize=20;FrameShift=10;Del=1;DelDel=1;NDL=2;





%% Data acqusition for training



if sw==0
    
    disp('you have to give three sample files for training  \n');
    name = input('Enter your name: ', 's');
    mkdir(strcat(datapath,'/',name));
    fs = 8000;
    myVoice = audiorecorder(fs,16,1);
    
    for i=1:3
        
        myVoice.StartFcn = 'disp(''Say: Allow me to login my system.'')';
        recordblocking(myVoice, 5);
        y = getaudiodata(myVoice);
        plot(y)
        savedata=strcat(datapath,'/',name,'/','S0',num2str(i),'.wav');
        audiowrite(savedata,y,fs);
       % pause(1)
       
    end
    sw=1;
end
%% training
d=dir(datapath);
folders=char(d.name);
folders=folders(3:end,:);



if sw==1
    for i=1:size(folders,1)
        dd=dir(strcat(datapath,'/',folders(i,:)));
        files_1=char(dd.name);
        files_1=files_1(3:end,:);
        for j=1:3
            filepath=strcat(datapath,'/',folders(i,:),'/',files_1(j,:));
            [d,fs]=audioread(filepath);
            
            d=d-mean(d);
            d=d./(1.01*max(abs(d)));
            if fs~=8000
                d=resample(d,8000,fs);fs=8000;
            end
            
            [MFCC, DMFCC, DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,Nomfccs,Nbands,Framesize,FrameShift,Del,DelDel,NDL);
            mfcc{j}=[MFCC(:,2:end),DMFCC(:,2:end), DDMFCC(:,2:end)]';
            
            
            
            
        end
        
        [m{i},var{i} ,weights{i} ] = vl_gmm([mfcc{:}], 6, 'verbose', 'MaxNumIterations',100);
        
        
    end
    sw=2;
end
%%



%testing
if sw==2
    fs = 8000;
    myVoice = audiorecorder(fs,16,1);;
    %
    disp('you have to give one sample for testing  \n');
    name = input('Enter your name: ', 's');
    
    myVoice.StartFcn = 'disp(''Say: Allow me to login my system.'')';
    recordblocking(myVoice, 5);
    myVoice.StopFcn = 'disp(''End of recording.'')';
    y = getaudiodata(myVoice);
    plot(y)
    pause(1)
    
    dd=dir(strcat(datapath,'/',folders(i,:)));
    files_1=char(dd.name);
    files_1=files_1(3:end,:);
    
    d=y;
    d=d-mean(d);
    d=d./(1.01*max(abs(d)));
    
    if fs~=8000
        d=resample(d,8000,fs);fs=8000;
    end
    
    [MFCC, DMFCC, DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,Nomfccs,Nbands,Framesize,FrameShift,Del,DelDel,NDL);
    xmfcc=[MFCC(:,2:end),DMFCC(:,2:end), DDMFCC(:,2:end)]';
    
    
    for k=1:size(folders,1)
        llk(k) = mean(compute_llk(xmfcc,m{k},var{k},weights{k}));
    end
    score{i}=llk;
    
    id=find(llk==max(llk));
    %folders(id,:)
    
    if strcmp(name,deblank((folders(id,:))))
        disp('Accepted')
    else
        disp('rejected')
    end
    
end


