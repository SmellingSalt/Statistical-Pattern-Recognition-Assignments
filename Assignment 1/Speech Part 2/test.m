myVoice = audiorecorder;
FS = 8000;

% Define callbacks to show when recording starts and completes.
for i = 1:3
    
    my_input= input('Are you ready to record your voice? y/n \n', 's');
    if my_input == 'y'
        name = input('Enter your name: ', 's');
        file_name = input('give your file name (001, 002,003):', 's');
        myVoice.StartFcn = 'disp(''Say: Allow me to log into my system.'')';
        recordblocking(myVoice, 5);
        myVoice.StopFcn = 'disp(''End of recording.'')';
    else
        disp("k")
    end
    play(myVoice);

    B = strcat('myData/',name);
    %B=string(B);
    mkdir(B)
    vv=strcat(B,'/',file_name,'.wav');
    y = getaudiodata(myVoice);
    audiowrite(vv,y, FS);
    plot(y);
    %record(myVoice, 5);
    [k, FS] = audioread(vv);
    %sound(k, FS)
       
end
% 
% load handel.mat
% 
% filename = 'handel.wav';
% audiowrite(filename,y,Fs);
% clear y Fs