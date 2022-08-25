import os, glob
#import librosa
import datetime
import sox


#todays_date = date.today()

#date_req = str(todays_date.month) + '_' + str(todays_date.day)


file_path = '/newdrive/ASR/Code/stt_of_audioFiles_by_ameyoASR/dataset'
dest_path = '/newdrive/ASR/Code/stt_of_audioFiles_by_ameyoASR/audio/'
count = 0

label = 'train'

for files in glob.glob(file_path + '/*wav'):
    fnames = os.path.basename(files)  #
    #count +=1

    #duration = librosa.get_duration(filename=files)
    #duration = int(duration * 1000)

    #new_wav_file_name = dest_path + '/' + label + '__' + date_req + str(count) + '.wav'

    cmd = "sox {0} -r 16000 {1}".format(files, dest_path+fnames)
    os.system(cmd)
