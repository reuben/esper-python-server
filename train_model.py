import base64
import collections
import io
import requests
import shlex
import subprocess
import sys
import time
import wave
import webrtcvad


SAMPLE_RATE = 16000
SOX_CMD = 'rec -q -V0 --compression 0.0 --no-dither -e signed -L -c 1 -b 16 -r 16k -t raw - gain -3 lowpass -2 4k'

endpoint = 'https://snowboy.kitt.ai/api/v1/train/'

############# MODIFY THE FOLLOWING #############
token = '2ccf2bcdf66faa469b017cc466b9abac6c3a5fcf'
hotword_name = 'Esper'
language = 'en'
age_group = '20_29'
microphone = 'macbook microphone'
############### END OF MODIFY ##################


vad = webrtcvad.Vad(mode=3)

def record_hotword(name):
    recording = io.BytesIO()
    wavout = wave.open(recording, 'wb')
    wavout.setnchannels(1)
    wavout.setsampwidth(2)
    wavout.setframerate(SAMPLE_RATE)

    ringbuffer = collections.deque(maxlen=15)
    ringbuffer_frames = collections.deque(maxlen=15)
    voiced_threshold = 0.8 * ringbuffer.maxlen
    unvoiced_threshold = 0
    heard_something = False

    subproc = subprocess.Popen(shlex.split(SOX_CMD), stdout=subprocess.PIPE)
    print('Say "{}"...'.format(name))
    listening = True

    while True:
        frame = subproc.stdout.read(320)
        if frame:
            voiced = vad.is_speech(frame, SAMPLE_RATE)

            if heard_something:
                wavout.writeframes(frame)
            else:
                ringbuffer_frames.append(frame)

            ringbuffer.append(1 if voiced else 0)
            num_voiced = sum(ringbuffer)

            if not heard_something and num_voiced >= voiced_threshold:
                print('Heard something')
                heard_something = True
                for f in ringbuffer_frames:
                    wavout.writeframes(f)

            if heard_something and num_voiced <= unvoiced_threshold:
                listening = False
                ringbuffer.clear()
                ringbuffer_frames.clear()

        if not frame or not listening:
            print('Silence detected, stopped listening')
            listening = False
            subproc.terminate()
            ringbuffer.clear()
            ringbuffer_frames.clear()
            return recording.getvalue()

def get_wave(fname):
    with open(fname, 'rb') as infile:
        return base64.b64encode(infile.read()).decode('ascii')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        out = sys.argv[1]
    else:
        out = 'heyesper.pmdl'

    print('Training a model to identify your voice')

    gender = input('What is your gender? (M = male/F = female/? = prefer not to say) ')
    if gender not in ('M', 'F'):
        gender = 'M'

    input('Okay, press ENTER to continue...')

    wavs = []
    while len(wavs) < 3:
        wav = record_hotword('Hey Esper')
        if wav:
            wavs.append(wav)
            fout = 'wav{}.wav'.format(len(wavs))
        else:
            print('Did not hear anything, please try again...')

    data = {
        'name': hotword_name,
        'language': language,
        'age_group': age_group,
        'gender': gender,
        'microphone': microphone,
        'token': token,
        'voice_samples': [{'wave': base64.b64encode(wav).decode('ascii')} for wav in wavs],
    }

    response = requests.post(endpoint, json=data)
    if response.ok:
        with open(out, 'wb') as outfile:
            outfile.write(response.content)
        print('Saved model to "%s".' % out)
    else:
        print('Request failed.')
        print(response.text)
