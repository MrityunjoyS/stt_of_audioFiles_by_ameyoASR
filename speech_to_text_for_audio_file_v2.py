#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
import datetime
import glob

import editdistance
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import StopwatchMeter, TimeMeter

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=0.2,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder",
        choices=["viterbi", "kenlm", "fairseqlm"],
        help="use a w2l decoder",
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def init_module(args, dataset_type=None,csv_file=None):
    global task
    model_state=None
    check_args(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    logging.info(args)

    global use_cuda
    use_cuda = torch.cuda.is_available() and not args.cpu

    logging.info("| using cuda {}".format(use_cuda))
    logging.info("| decoding with criterion {}".format(args.criterion))

    task = tasks.setup_task(args)

    global models

    # Load ensemble
    if args.load_emissions:
        models, criterions = [], []
        task.load_dataset(args.gen_subset)
    else:
        logging.info("| loading model(s) from {}".format(args.path))
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.path),
            arg_overrides=ast.literal_eval(args.model_overrides),
            task=task,
            suffix=args.checkpoint_suffix,
            strict=(args.checkpoint_shard_count == 1),
            num_shards=args.checkpoint_shard_count,
            state=model_state,
        )
        optimize_models(args, use_cuda, models)
        #BIKRAM: This is where dataset is loaded for inference
        #task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    # Set dictionary
    # global tgt_dict
    # tgt_dict = task.target_dictionary
    fname = "audio/sample_mono_16k.wav"
    logging.info("Testing of file {}".format(fname))
    wav = read_raw_audio(fname)
    make_prediction(wav, 16000)
    
    if dataset_type:
        fi = '/fairseq/result/' + dataset_type + '_PROD_feedback_v2_transcription.txt'

        wrt_file = open(fi, 'w+')

        for files in glob.glob("dataset/" + "*.wav"):
                logging.info("Transcripting for file {}".format(files))
                wav = read_raw_audio(files)
                transcription = make_prediction(wav, 16000)
                if '<unk>' in transcription:
                                transcription = transcription.replace('<unk>', ' ')
                basename = os.path.basename(files)
                logging.info("Audio_file: {} || Transciprion ---> {}".format(basename, transcription))
                wrt_file.write("{}   {}".format(basename, transcription))
                wrt_file.write("\n")
        wrt_file.close()
    else:
        cols = ['file_name', 'Google_ASR', 'Ameyo_ASR']

        lst=[]

        file_open=open('16Aug_11am-12pm_hindi_data_feedback_iter1.txt','r')
        file_lines=file_open.readlines()

        audio_path='dataset/'
        for wrt in file_lines:

    	    split_file=wrt.split('	')
    	    wav_files = split_file[0]
    	    google_data=split_file[-1].strip()
    	    files=audio_path+wav_files
            logging.info("Transcripting for file {}".format(files))
            wav = read_raw_audio(files)
            transcription = make_prediction(wav, 16000)
            if '<unk>' in transcription:
                            transcription = transcription.replace('<unk>', ' ')
            logging.info("Audio_file: {} || Transciprion ---> {}".format(wav_files, transcription))
            lst.append([wav_files, google_data, transcription])
        df = pd.DataFrame(lst, columns=cols)
        df.to_csv('/fairseq/result/16Aug_11am-12pm_hindi_data_feedback_iter1_with_google_and_ameyo.csv',index=False)
        print(df.head())


def build_generator(args, task):
    w2l_decoder = getattr(args, "w2l_decoder", None)
    if w2l_decoder == "viterbi":
        from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

        return W2lViterbiDecoder(args, task.target_dictionary)
    elif w2l_decoder == "kenlm":
        from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

        return W2lKenLMDecoder(args, task.target_dictionary)
    elif w2l_decoder == "fairseqlm":
        from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder

        return W2lFairseqLMDecoder(args, task.target_dictionary)
    else:
        logging.error("only flashlight decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment")
        return None

def read_raw_audio(fname):
    import soundfile as sf
    wav, curr_sample_rate = sf.read(fname)
    return wav


def make_prediction(wav, sample_rate):
    start_time = datetime.datetime.now()

    feats = torch.from_numpy(wav).float()
    # if feats.dim() == 2:
    #     feats = feats.mean(-1)

    # assert feats.dim() == 1, feats.dim()

    # if args.normalize:
    #     with torch.no_grad():
    #         feats = F.layer_norm(feats, feats.shape)   
    
    #Add a new dimention. This is done because we are using only sample - hence batch size is one
    sources = feats.unsqueeze(0)

    sample = {'net_input': {'source': sources, 'padding_mask': None}}
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    end_time = datetime.datetime.now()
    consumed_time = end_time - start_time
    logging.info("Feature loading Time Taken: {} seconds".format(consumed_time.total_seconds()))

    start_time = datetime.datetime.now()
    #BIKRAM::Decoder is an object - create new decoder object fro every prediction
    generator = build_generator(args, task)
    hypos = task.inference_step(generator, models, sample, None)
    #Make string from int outs - IF the output in GPU then we need to bring it to CPU then do the ops
    hyp_pieces = task.target_dictionary.string(hypos[0][0]["tokens"].int().cpu())
    hyp_words = post_process(hyp_pieces, args.post_process)
    end_time = datetime.datetime.now()
    consumed_time = end_time - start_time
    logging.info("Inference Time Taken: {} seconds".format(consumed_time.total_seconds()))

    logging.info("Predicted sentence: {}".format(hyp_words))
    return hyp_words


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


def cli_main(language_model_use, base_path, model_path, lm_path, lexicon_path, dataset_type, csv_file):
    parser = make_parser()
    modelpath = model_path
    #modelpath = "/dacx/checkpoint_models/w2v2-ctc/en-in/checkpoint_best.pt"
    #Disctionary path , dictionary name :  dict.ltr.txt
    #dictionary_path = "/dacx/checkpoint_models/w2v2-ctc/en-in/"
    dictionary_path = base_path
    #lm_path= "/dacx/collection.bin"
    #lexicon_path = "/dacx/collection.lexicons"
    #Pass the data argument as a list of arguments to read
    global args
    # if language_model_use:
    #     flags = [dictionary_path, "--task","audio_pretraining","--nbest", "1","--path", modelpath,"--criterion","ctc","--labels","ltr","--batch-size","1","--post-process","letter","--w2l-decoder","kenlm","--lm-model",lm_path,"--lm-weight", "2","--word-score", "-1", "--sil-weight", "0","--lexicon",lexicon_path]
    # else:
    flags = [dictionary_path, "--task","audio_finetuning","--nbest", "1","--path", modelpath, "--gen-subset", "test", "--results-path", "/root/fairseq/result", "--w2l-decoder","viterbi",  "--lm-weight", "2", "--word-score", "-1", "--sil-weight", "0" ,"--criterion","ctc","--labels","ltr","--batch-size","1","--post-process","letter"]
    args = options.parse_args_and_arch(parser, flags)
    if csv_file:
        init_module(args, csv_file)
    else:
        init_module(args, dataset_type)


if __name__ == "__main__":
    language_model_use = sys.argv[1]
    base_path = sys.argv[2]
    model_path = sys.argv[3]
    lm_path = sys.argv[4]
    lexicon_path = sys.argv[5]
    dataset_type = sys.argv[6]
    csv_file = sys.argv[7]
    cli_main(language_model_use, base_path, model_path, lm_path, lexicon_path, dataset_type,csv_file)

