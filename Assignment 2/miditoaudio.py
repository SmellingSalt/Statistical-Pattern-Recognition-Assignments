import os
from glob import glob
# convert_command=timidity
# midi_file=0AuraLee.mid

# -Ow -o - | lame - -b 64 
# mp3_file=0AuraLee2.mp3

path_to_midi_data=os.path.relpath(os.path.join('MusicDatasets/Beethoven/MIDIFILES'))
midi_data_files_list=glob(os.path.join(path_to_midi_data, "*.mid"))
mp3_data_path=os.path.relpath(os.path.join('MusicDatasets/Beethoven/MP3FILES'))
i=0
for midi_file in midi_data_files_list:    
    # convert_command="timidity "+midi_file+" -Ow -o - | lame - -b 64 "+mp3_data_path+"/"+str(i)+".mp3"
    # print(convert_command)    
    # os.system(convert_command)
    convert_command="timidity "+midi_file+" -Ow "+mp3_data_path+"/"+str(i)+".wav"
    os.system(convert_command)
    i+=1
    print("\n Finished Converting ",i," of ",len(midi_data_files_list))
# os.system("ls")
