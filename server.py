#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import asyncio
import collections
import deepspeech as ds
import functools
import janus
import json
import numpy as np
import pyaudio
import shlex
import snowboydecoder
import subprocess
import time
import wave
import webrtcvad
import websockets

from datetime import datetime, timedelta


ALPHA = 1.5
BETA = 2.25
N_FEATURES = 26
N_CONTEXT = 9
BEAM_WIDTH = 512
SAMPLE_RATE = 16000

SOX_CMD = 'rec -q -V0 --compression 0.0 --no-dither -e signed -L -c 1 -b 16 -r 16k -t raw - '
# SOX_CMD = 'sox -t coreaudio "Background Musi" -q -V0 --compression 0.0 --no-dither -e signed -L -c 1 -b 16 -r 16k -t raw -'

listening = False
subproc = None
sctx = None
model = None

parser = argparse.ArgumentParser(description='DeepSpeech speech-to-text from microphone')
parser.add_argument('--model', required=True,
                    help='Path to the model (protocol buffer binary file)')
parser.add_argument('--alphabet', required=True,
                    help='Path to the configuration file specifying the alphabet used by the network')
parser.add_argument('--lm', nargs='?',
                    help='Path to the language model binary file')
parser.add_argument('--snowboy', nargs='?',
                    help='Path to the Snowboy hot-word detection model file')
args = parser.parse_args()

print('Loading acoustic model...')
model = ds.Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)

print('Loading language model...')
model.enableDecoderWithLM(args.alphabet, args.lm, args.alphabet, ALPHA, BETA)

print('Loading voice activity detector...')
vad = webrtcvad.Vad(mode=0)

# Thread messaging queue
loop = asyncio.get_event_loop()
queue = janus.Queue(loop=loop)


def play_audio_file(fname="listening.wav"):
    ding_wav = wave.open(fname, 'rb')
    ding_data = ding_wav.readframes(ding_wav.getnframes())
    audio = pyaudio.PyAudio()
    stream_out = audio.open(
        format=audio.get_format_from_width(ding_wav.getsampwidth()),
        channels=ding_wav.getnchannels(),
        rate=ding_wav.getframerate(), input=False, output=True)
    stream_out.start_stream()
    stream_out.write(ding_data)
    time.sleep(0.2)
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()


def snowboy_thread(q):
    def callback():
        q.put(datetime.utcnow())
        play_audio_file()

    print('Loading hot-word detector...')
    detector = snowboydecoder.HotwordDetector(args.snowboy, sensitivity=0.5)
    detector.start(detected_callback=callback, sleep_time=0.03)


async def producer(ws):
    print('Producer started')
    global listening

    ringbuffer = collections.deque(maxlen=10) # 200ms of buffering
    heard_something = False

    i = 0

    while True:
        if listening:
            frame = subproc.stdout.read(320)
            if frame:
                model.feedAudioContent(sctx, np.frombuffer(frame, np.int16))
                voiced = vad.is_speech(frame, SAMPLE_RATE)

                ringbuffer.append(1 if voiced else 0)
                num_voiced = sum(ringbuffer)

                if not heard_something and num_voiced > 0.9 * ringbuffer.maxlen:
                    print('Heard something')
                    heard_something = True

                if heard_something and num_voiced <= 0.1 * ringbuffer.maxlen:
                    print('VAD detected silence')
                    listening = False
                    heard_something = False
                    ringbuffer.clear()

            if not frame or not listening:
                print('EOF or VAD, stop listening')
                listening = False
                heard_something = False
                transcription = model.finishStream(sctx)
                print('Transcription:', transcription)
                await ws.send(json.dumps({
                    'type': 'result',
                    'result': transcription
                }))
                subproc.terminate()

            await asyncio.sleep(.02)
        else:
            await asyncio.sleep(.1)


async def consumer(ws, q):
    global sctx
    global subproc
    global stdout
    global listening

    print('Waiting for hot-word...')
    while True:
        detected = await q.get()
        if datetime.utcnow() - detected  > timedelta(seconds=5):
            continue # discard old detections (browser not connected)

        print('Hot-word detected, preparing...')
        await ws.send(json.dumps({
            'type': 'listening'
        }))
        sctx = model.setupStream()
        subproc = subprocess.Popen(
            shlex.split(SOX_CMD),
            stdout=subprocess.PIPE,
            bufsize=0)
        print('Listening...')
        listening = True


async def serve(websocket, path):
    consumer_task = asyncio.ensure_future(consumer(websocket, queue.async_q))
    producer_task = asyncio.ensure_future(producer(websocket))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


start_server = websockets.serve(serve, 'localhost', 8777)

print('Starting event loop...')
snowboy = loop.run_in_executor(None, snowboy_thread, queue.sync_q)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_until_complete(snowboy)
asyncio.get_event_loop().run_forever()
