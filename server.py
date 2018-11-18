#!/usr/bin/env python
from __future__ import division, print_function

from datetime import datetime
import os
import logging

os.makedirs('log', exist_ok=True)
log = logging.getLogger('esper')
log.setLevel(logging.DEBUG)

class MyFormatter(logging.Formatter):
    converter=datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

log_fmt = MyFormatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
fh = logging.FileHandler('log/esper.log')
fh.setFormatter(log_fmt)
log.addHandler(fh)

session_counter = 0
try:
    with open('log/session_counter') as fin:
        session_counter = int(fin.read())
except:
    pass

with open('log/session_counter', 'w') as fout:
    fout.write(str(session_counter+1))

os.makedirs(os.path.join('log', 'session', str(session_counter)), exist_ok=True)

import argparse
import asyncio
import collections
import deepspeech as ds
import functools
import janus
import json
import numpy as np
import shlex
import snowboydecoder
import subprocess
import sys
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

SOX_CMD = 'rec -q -V0 --compression 0.0 --no-dither -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2 lowpass -2 4k'
# SOX_CMD = 'sox -t coreaudio "Background Musi" -q -V0 --compression 0.0 --no-dither -e signed -L -c 1 -b 16 -r 16k -t raw - gain -3 lowpass -2 4k'

listening = False
started_listening = None
subproc = None
sctx = None
model = None

parser = argparse.ArgumentParser(description='DeepSpeech speech-to-text from microphone')
parser.add_argument('--model', default='output_graph.pbmm',
                    help='Path to the model (protocol buffer binary file)')
parser.add_argument('--alphabet', default='alphabet.txt',
                    help='Path to the configuration file specifying the alphabet used by the network')
parser.add_argument('--lm', default='esper-lm.binary',
                    help='Path to the language model binary file')
parser.add_argument('--snowboy', default='heyesper.pmdl',
                    help='Path to the Snowboy hot-word detection model file')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

if args.verbose:
    sh = logging.StreamHandler()
    sh.setFormatter(log_fmt)
    log.addHandler(sh)

log.debug('Session {} start'.format(session_counter))

log.debug('Loading acoustic model...')
model = ds.Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)

log.debug('Loading language model...')
model.enableDecoderWithLM(args.alphabet, args.lm, args.alphabet, ALPHA, BETA)

log.debug('Loading voice activity detector...')
vad = webrtcvad.Vad(mode=3)

# Thread messaging queue
loop = asyncio.get_event_loop()
queue = janus.Queue(loop=loop)

def play_audio_file(fname="listening.wav"):
    subprocess.call(shlex.split("play -q -V0 \"{}\" gain -6 speed 1.5".format(fname)))


def snowboy_thread(q):
    def callback():
        if connected_clients:
            q.put(datetime.utcnow())
            play_audio_file()

    log.debug('Loading hot-word detector...')
    detector = snowboydecoder.HotwordDetector(args.snowboy, sensitivity=0.4)
    detector.start(detected_callback=callback, sleep_time=0.03)


async def send_to_all_clients(msg):
    for ws, cond in connected_clients:
        try:
            await ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            log.debug('Client connection closed: {}'.format(ws.remote_address))
            with await cond:
                cond.notify()


async def producer():
    log.debug('Producer started')
    global listening
    global started_listening

    ringbuffer = collections.deque(maxlen=20)
    ringbuffer_frames = collections.deque(maxlen=20)
    voiced_threshold = 0.9 * ringbuffer.maxlen
    unvoiced_threshold = 0

    heard_something = False

    recording = None

    count = 0
    cur_wav = 'log/session/{}/cmd{}.wav'.format(session_counter, count)
    wavout = wave.open(cur_wav, 'wb')
    wavout.setnchannels(1)
    wavout.setsampwidth(2)
    wavout.setframerate(SAMPLE_RATE)

    while True:
        if listening:
            frame = subproc.stdout.read(320)
            if frame:
                voiced = vad.is_speech(frame, SAMPLE_RATE)
                ringbuffer.append(1 if voiced else 0)
                num_voiced = sum(ringbuffer)
                wavout.writeframes(frame)

                model.feedAudioContent(sctx, np.frombuffer(frame, np.int16))

                if not heard_something and num_voiced >= voiced_threshold:
                    log.debug('Heard something')
                    heard_something = True

                if heard_something and num_voiced <= unvoiced_threshold:
                    listening = False
                    ringbuffer.clear()

                if (datetime.now() - started_listening) > timedelta(seconds=3):
                    log.debug('3 seconds timeout')
                    listening = False
                    ringbuffer.clear()

            if not frame or not listening:
                log.debug('EOF or VAD, stop listening')
                listening = False
                if heard_something:
                    transcription = model.finishStream(sctx)
                    res = {
                        'type': 'result',
                        'result': transcription
                    }

                    await send_to_all_clients(json.dumps(res))

                    log.debug(json.dumps({
                        'session_counter': session_counter,
                        'cmd_path': cur_wav,
                        'res': res
                    }))
                else:
                    res = {
                        'type': 'no_voice'
                    }

                    await send_to_all_clients(json.dumps(res))

                    log.debug(json.dumps({
                        'session_counter': session_counter,
                        'res': res
                    }))

                heard_something = False
                wavout.close()

                count += 1
                cur_wav = 'log/session/{}/cmd{}.wav'.format(session_counter, count)
                wavout = wave.open(cur_wav, 'wb')
                wavout.setnchannels(1)
                wavout.setsampwidth(2)
                wavout.setframerate(SAMPLE_RATE)

                subproc.terminate()
        else:
            await asyncio.sleep(.1)

async def consumer(q):
    global sctx
    global subproc
    global stdout
    global listening
    global started_listening

    log.debug('Waiting for hot-word...')
    while True:
        detected = await q.get()
        if datetime.utcnow() - detected  > timedelta(seconds=3):
            continue # discard old detections

        log.debug('Hot-word detected, preparing...')

        await send_to_all_clients(json.dumps({
            'type': 'listening'
        }))

        sctx = model.setupStream()
        subproc = subprocess.Popen(shlex.split(SOX_CMD), stdout=subprocess.PIPE)
        log.debug('Listening...')
        started_listening = datetime.now()
        listening = True


async def ping():
    while True:
        await send_to_all_clients(json.dumps({
            'type': 'ping'
        }))
        await asyncio.sleep(3)


connected_clients = set()


async def serve(websocket, path):
    log.debug('Client connected: {}'.format(websocket.remote_address))
    cond = asyncio.Condition()
    connected_clients.add((websocket, cond))
    async with cond:
        await cond.wait()
        connected_clients.remove((websocket, cond))


async def run_workers():
    consumer_task = asyncio.ensure_future(consumer(queue.async_q))
    producer_task = asyncio.ensure_future(producer())
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


start_server = websockets.serve(serve, 'localhost', 8777)

log.debug('Starting event loop...')
snowboy = loop.run_in_executor(None, snowboy_thread, queue.sync_q)
asyncio.ensure_future(consumer(queue.async_q))
asyncio.ensure_future(producer())
asyncio.ensure_future(ping())
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
