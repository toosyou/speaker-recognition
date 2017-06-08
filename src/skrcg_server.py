#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: speaker-recognition.py
# Date: Sun Feb 22 22:36:46 2015 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from __future__ import print_function
import argparse
import sys
import glob
import os
import itertools
import scipy.io.wavfile as wavfile

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'gui'))
from gui.interface import ModelInterface
from gui.utils import read_wav
from filters.silence import remove_silence

import tornado.ioloop
import tornado.web
from tornado import gen
import cStringIO
import time
import numpy as np
import wave

CHANNELS = 1
RATE = 44100
CHUNK = 8192
PREDICT_SECS = 2
PREDICT_TIME_FRAMES = RATE*PREDICT_SECS
frames = np.zeros(int(PREDICT_TIME_FRAMES), dtype=np.int16)

rcg_model = ModelInterface.load('voice.model')

class MainHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        global frames
        header_length = self.request.headers.get('Content-Length')
        header_time = self.request.headers.get('time')
        voice_slice = self.request.body

        if not header_length or not header_time or not voice_slice or len(voice_slice) != int(header_length): # something's wrong
            self.set_status(400)
            self.finish()
            return
        else: # send OK to client
            self.set_status(200)
            self.finish()

        voice_slice = np.fromstring(voice_slice, dtype=np.int16)
        frames = np.append( frames[len(voice_slice):], voice_slice )
        print(rcg_model.predict_scores(RATE, frames), 'delay:', time.time() - float(header_time))

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print('Server starts!')
    tornado.ioloop.IOLoop.current().start()
