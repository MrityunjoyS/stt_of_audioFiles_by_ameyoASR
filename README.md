For transcribing audio files:

$ docker build -t wav2vec2 -f Dockerfile .
$ docker run --rm -itd --ipc=host -v $PWD/result:/fairseq/result --name w2v wav2vec2
or
$ docker run --runtime=nvidia --rm -itd --ipc=host -v $PWD/result:/fairseq/result -v /<psth for audio that will be transcripted>/:/fairseq/dataset --name w2v wav2vec2
$ docker exec -it w2v bash

$ vi speech_to_text_for_audio_file_v2.py +163  → update dir of audio
or
$ docker run --runtime=nvidia --rm -itd --ipc=host -v $PWD/result:/fairseq/result -v /root/ASR/Code/infer_v2/dataset_hi/test_hi_used/:/fairseq/dataset --name w2v wav2vec2

$docker run --runtime=nvidia --rm -itd --ipc=host -v $PWD/result:/fairseq/result -v /newdrive/ASR/Data/Speech/Supervised/dataset_hi_upto28Sep/hi-in/valid_hi_used/:/fairseq/dataset --name w2v wav2vec2

$docker exec -it w2v /bin/bash

##by_just_audio_file
$ python3 speech_to_text_for_audio_file_v2.py 'false' /asr_models/hindi_aws/model_and_dict/ /asr_models/hindi_aws/model_and_dict/checkpoint_best.pt 'false' 'false' test//train//valid 'false'

##by_csv_file
$ python3 speech_to_text_for_audio_file_v2.py 'false' /asr_models/hindi_aws/model_and_dict/ /asr_models/hindi_aws/model_and_dict/checkpoint_best.pt 'false' 'false' 'false' 16Aug_11am-12pm_hindi_data_feedback_iter1.txt



for coping from container :-
$ docker cp w2v:/fairseq/result/transcription.txt .
