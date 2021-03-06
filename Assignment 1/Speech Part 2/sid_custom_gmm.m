clc;
clear all;
close all;
% addpath(genpath('C:\Users\Sawan Singh Mahara\Downloads\vlfeat-0.9.21'));
base_path='myData';
d=dir(base_path);
folders=char(d.name);
folders=folders(3:end,:);

Nomfccs=14;Nbands=22;Framesize=20;FrameShift=10;Del=1;DelDel=1;NDL=2;

%% training
for i=1:size(folders,1)
    dd=dir(strcat(base_path,'/',folders(i,:)));
    files_1=char(dd.name);
    files_1=files_1(3:end,:);
    for j=1:size(files_1,1)-1
        filepath=strcat(base_path,'/',folders(i,:),'/',files_1(j,:));
        [d,fs]=audioread(filepath);
        
        d=d-mean(d);
        d=d./(1.01*max(abs(d)));
        if fs~=8000
            d=resample(d,8000,fs);fs=8000;
        end
        
        [MFCC, DMFCC, DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,Nomfccs,Nbands,Framesize,FrameShift,Del,DelDel,NDL);
        mfcc{j}=[MFCC(:,2:end),DMFCC(:,2:end), DDMFCC(:,2:end)]';
        
        
        
        
    end
    
      [m{i},var{i} ,weights{i} ] = vl_gmm([mfcc{:}], 2, 'verbose', 'MaxNumIterations',100);      
    
    
end

%testing


for i=1:size(folders,1)
    dd=dir(strcat(base_path,'/',folders(i,:)));
    files_1=char(dd.name);
    files_1=files_1(3:end,:);
    for j=size(files_1,1)
        filepath=strcat(base_path,'/',folders(i,:),'/',files_1(j,:));
        [d,fs]=audioread(filepath);
        
        d=d-mean(d);
        d=d./(1.01*max(abs(d)));
        
        if fs~=8000
            d=resample(d,8000,fs);fs=8000;
        end
        
        [MFCC, DMFCC, DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,Nomfccs,Nbands,Framesize,FrameShift,Del,DelDel,NDL);
        xmfcc=[MFCC(:,2:end),DMFCC(:,2:end), DDMFCC(:,2:end)]';
        
        
        
        
    end
    for k=1:size(folders,1)
        llk(k) = mean(compute_llk(xmfcc,m{k},var{k},weights{k}));      
    end
    score{i}=llk;
    
    id=find(llk==max(llk));
    
    
    
    
end






