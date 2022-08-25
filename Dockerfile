FROM flml/flashlight:cuda-base-consolidation-latest
#FROM nvidia/cuda:10.2-base
# just in case for visibility
ENV MKLROOT="/opt/intel/mkl"
ENV KENLM_ROOT=/opt/kenlm


# ==================================================================
# flashlight with CUDA backend
# ------------------------------------------------------------------
# Setup and build flashlight - python binding for GPU


RUN git clone https://github.com/bikramjitroy/flashlight.git && \
    cd /flashlight/bindings/python/ && \
    pip3 install packaging==19.1 && \
    python3 setup.py install


RUN git clone https://github.com/pytorch/fairseq.git && \
    cd fairseq && \
    pip3 install --editable ./

RUN pip3 install editdistance

#WORKDIR /root
#RUN git clone https://github.com/pytorch/fairseq.git
#RUN pip install editdistance
#WORKDIR /root/fairseq
#RUN pip install --editable ./


RUN mkdir -p /root/asrstream && \
    mkdir -p /asr_models 


COPY ./asr_models/hindi_aws/ /asr_models/hindi_aws/

#COPY CODE
COPY ./speech_to_text_for_audio_file_v2.py /fairseq/speech_to_text_for_audio_file_v2.py
COPY ./requirements.txt /root/asrstream/requirements.txt
#COPY ./audio_to_text.py /fairseq/audio_to_text.py

RUN cd /root/asrstream && \
    pip3 install -r requirements.txt

WORKDIR /fairseq

RUN mkdir audio 

COPY ./audio/sample_mono_16k.wav audio/sample_mono_16k.wav

RUN mkdir dataset

COPY ./dataset dataset

#RUN mkdir result

RUN mkdir -p /newdrive/ASR/Code/fairseq/sav_multilingual_dir/
COPY ./asr_models/hindi_aws/embedding_model/checkpoint644.pt /newdrive/ASR/Code/fairseq/sav_multilingual_dir/checkpoint644.pt
