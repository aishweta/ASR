from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json

from deepspeech import Model, printVersions
from timeit import default_timer as timer
import time

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# load models
lm = '/home/shridhar/aws/shree/pre_trained/deepspeech-0.6.1-models/lm.binary'
trie = '/home/shridhar/aws/shree/pre_trained/deepspeech-0.6.1-models/trie'
alpha= '/home/shridhar/aws/shree/pre_trained/deepspeech-0.6.1-models/alphabet.txt'
model = '/home/shridhar/aws/shree/pre_trained/deepspeech-0.6.1-models/output_graph.pbmm'
lm_alpha = 0.75
lm_beta = 1.85

ds = Model(model,500)    ### ds = Model(args.model, args.beam_width)
desired_sample_rate = ds.sampleRate()
ds.enableDecoderWithLM(lm, trie, lm_alpha, lm_beta)


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def stt(audio):
    fin = wave.open(audio, 'rb')
    fs = fin.getframerate()

    if fs != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs, desired_sample_rate), file=sys.stderr)
        fs, audio = convert_samplerate(audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs)
    fin.close()

    print('Running inference.', file=sys.stderr)

    t0=time.time()
    text = ds.stt(audio)
    print('total stt time = ',time.time()-t0)
    return text

# specifying the path for audio file
#audio = 'ewm.wav'
#text = stt(audio)
#print('the output is: ', text)

