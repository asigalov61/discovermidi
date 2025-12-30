#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#
#	Discover Search and Filter Python Module
#	Version 1.0
#
#   NOTE: Module code starts after the partial MIDI.py module @ line 1122
#
#	Based upon MIDI.py module v.6.7. by Peter Billam / pjb.com.au
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#	PARTIAL MIDI.py Module v.6.7. by Peter Billam
#   Please see TMIDI 2.3/tegridy-tools repo for full MIDI.py module code
# 
#   Or you can always download the latest full version from:
#
#   https://pjb.com.au/
#   https://peterbillam.gitlab.io/miditools/
#	
#	Copyright 2020 Peter Billam
#
###################################################################################
###################################################################################
#
#   Critical dependencies
#
#   !pip install huggingface_hub
#   !pip install hf-transfer
#   !pip install ipywidgets
#   !pip install tqdm
#
#   !pip install torch
#   !pip install scikit-learn
#   !pip install scipy
#   !pip install matplotlib
#   !pip install midirenderer
#   !pip install mididoctor
#   !pip install numpy==1.26.4
#
###################################################################################
###################################################################################
#
#   Basic use example
#
#   import search_and_filter
#
#   search_and_filter.download_dataset()
#
#   search_and_filter.parallel_extract()
#
#   features_matrixes, features_matrixes_file_names = search_and_filter.load_features_matrixes()
#
#   search_and_filter.search_and_filter(features_matrixes, features_matrixes_file_names)
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading Discover Search and Filter Python module...')
print('Please wait...')

__version__ = '1.0.0'

###################################################################################
###################################################################################

import os, sys, struct, copy

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time

import random

import json

from collections import Counter, OrderedDict, defaultdict

from itertools import combinations

import math

import torch

import numpy as np

import midirenderer

from pathlib import Path
    
import shutil

import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download

from typing import Tuple, Optional, Iterable, Union, Sequence, List

import tqdm

###################################################################################
###################################################################################
# Partial MIDI.py module code is below
###################################################################################
###################################################################################

Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4
_no_warning = False

#------------------------------- Encoding stuff --------------------------

def score2opus(score=None, text_encoding='ISO-8859-1'):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of score-events, and each event is itself a list.  A score-event
is similar to an opus-event (see above), except that in a score:
 1) the times are expressed as an absolute number of ticks
    from the track's start time
 2) the pairs of 'note_on' and 'note_off' events in an "opus"
    are abstracted into a single 'note' event in a "score":
    ['note', start_time, duration, channel, pitch, velocity]
score2opus() returns a list specifying the equivalent "opus".

my_score = [
    96,
    [   # track 0:
        ['patch_change', 0, 1, 8],
        ['note', 5, 96, 1, 25, 96],
        ['note', 101, 96, 1, 29, 96]
    ],   # end of track 0
]
my_opus = score2opus(my_score)
'''
    if len(score) < 2:
        score=[1000, [],]
    tracks = copy.deepcopy(score)
    ticks = int(tracks.pop(0))
    opus_tracks = []
    for scoretrack in tracks:
        time2events = dict([])
        for scoreevent in scoretrack:
            if scoreevent[0] == 'note':
                note_on_event = ['note_on',scoreevent[1],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                note_off_event = ['note_off',scoreevent[1]+scoreevent[2],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                if time2events.get(note_on_event[1]):
                   time2events[note_on_event[1]].append(note_on_event)
                else:
                   time2events[note_on_event[1]] = [note_on_event,]
                if time2events.get(note_off_event[1]):
                   time2events[note_off_event[1]].append(note_off_event)
                else:
                   time2events[note_off_event[1]] = [note_off_event,]
                continue
            if time2events.get(scoreevent[1]):
               time2events[scoreevent[1]].append(scoreevent)
            else:
               time2events[scoreevent[1]] = [scoreevent,]

        sorted_times = []  # list of keys
        for k in time2events.keys():
            sorted_times.append(k)
        sorted_times.sort()

        sorted_events = []  # once-flattened list of values sorted by key
        for time in sorted_times:
            sorted_events.extend(time2events[time])

        abs_time = 0
        for event in sorted_events:  # convert abs times => delta times
            delta_time = event[1] - abs_time
            abs_time = event[1]
            event[1] = delta_time
        opus_tracks.append(sorted_events)
    opus_tracks.insert(0,ticks)
    _clean_up_warnings()
    return opus_tracks

#--------------------------- Decoding stuff ------------------------

def midi2opus(midi=b'', do_not_check_MIDI_signature=False):
    r'''Translates MIDI into a "opus".  For a description of the
"opus" format, see opus2midi()
'''
    my_midi=bytearray(midi)
    if len(my_midi) < 4:
        _clean_up_warnings()
        return [1000,[],]
    id = bytes(my_midi[0:4])
    if id != b'MThd':
        _warn("midi2opus: midi starts with "+str(id)+" instead of 'MThd'")
        _clean_up_warnings()
        if do_not_check_MIDI_signature == False:
          return [1000,[],]
    [length, format, tracks_expected, ticks] = struct.unpack(
     '>IHHH', bytes(my_midi[4:14]))
    if length != 6:
        _warn("midi2opus: midi header length was "+str(length)+" instead of 6")
        _clean_up_warnings()
        return [1000,[],]
    my_opus = [ticks,]
    my_midi = my_midi[14:]
    track_num = 1   # 5.1
    while len(my_midi) >= 8:
        track_type   = bytes(my_midi[0:4])
        if track_type != b'MTrk':
            #_warn('midi2opus: Warning: track #'+str(track_num)+' type is '+str(track_type)+" instead of b'MTrk'")
            pass
        [track_length] = struct.unpack('>I', my_midi[4:8])
        my_midi = my_midi[8:]
        if track_length > len(my_midi):
            _warn('midi2opus: track #'+str(track_num)+' length '+str(track_length)+' is too large')
            _clean_up_warnings()
            return my_opus   # 5.0
        my_midi_track = my_midi[0:track_length]
        my_track = _decode(my_midi_track)
        my_opus.append(my_track)
        my_midi = my_midi[track_length:]
        track_num += 1   # 5.1
    _clean_up_warnings()
    return my_opus

def opus2score(opus=[]):
    r'''For a description of the "opus" and "score" formats,
see opus2midi() and score2opus().
'''
    if len(opus) < 2:
        _clean_up_warnings()
        return [1000,[],]
    tracks = copy.deepcopy(opus)  # couple of slices probably quicker...
    ticks = int(tracks.pop(0))
    score = [ticks,]
    for opus_track in tracks:
        ticks_so_far = 0
        score_track = []
        chapitch2note_on_events = dict([])   # 4.0
        for opus_event in opus_track:
            ticks_so_far += opus_event[1]
            if opus_event[0] == 'note_off' or (opus_event[0] == 'note_on' and opus_event[4] == 0):  # 4.8
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                if chapitch2note_on_events.get(key):
                    new_event = chapitch2note_on_events[key].pop(0)
                    new_event[2] = ticks_so_far - new_event[1]
                    score_track.append(new_event)
                elif pitch > 127:
                    pass #_warn('opus2score: note_off with no note_on, bad pitch='+str(pitch))
                else:
                    pass #_warn('opus2score: note_off with no note_on cha='+str(cha)+' pitch='+str(pitch))
            elif opus_event[0] == 'note_on':
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                new_event = ['note',ticks_so_far,0,cha,pitch, opus_event[4]]
                if chapitch2note_on_events.get(key):
                    chapitch2note_on_events[key].append(new_event)
                else:
                    chapitch2note_on_events[key] = [new_event,]
            else:
                opus_event[1] = ticks_so_far
                score_track.append(opus_event)
        # check for unterminated notes (Oisín) -- 5.2
        for chapitch in chapitch2note_on_events:
            note_on_events = chapitch2note_on_events[chapitch]
            for new_e in note_on_events:
                new_e[2] = ticks_so_far - new_e[1]
                score_track.append(new_e)
                pass #_warn("opus2score: note_on with no note_off cha="+str(new_e[3])+' pitch='+str(new_e[4])+'; adding note_off at end')
        score.append(score_track)
    _clean_up_warnings()
    return score

def midi2score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score", using midi2opus() then opus2score()
'''
    return opus2score(midi2opus(midi, do_not_check_MIDI_signature))

def midi2ms_score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score" with one beat per second and one
tick per millisecond, using midi2opus() then to_millisecs()
then opus2score()
'''
    return opus2score(to_millisecs(midi2opus(midi, do_not_check_MIDI_signature)))

def midi2single_track_ms_score(midi_path_or_bytes, 
                                recalculate_channels = False, 
                                pass_old_timings_events= False, 
                                verbose = False, 
                                do_not_check_MIDI_signature=False
                                ):
    r'''
Translates MIDI into a single track "score" with 16 instruments and one beat per second and one
tick per millisecond
'''

    if type(midi_path_or_bytes) == bytes:
      midi_data = midi_path_or_bytes

    elif type(midi_path_or_bytes) == str:
      midi_data = open(midi_path_or_bytes, 'rb').read() 

    score = midi2score(midi_data, do_not_check_MIDI_signature)

    if recalculate_channels:

      events_matrixes = []

      itrack = 1
      events_matrixes_channels = []
      while itrack < len(score):
          events_matrix = []
          for event in score[itrack]:
              if event[0] == 'note' and event[3] != 9:
                event[3] = (16 * (itrack-1)) + event[3]
                if event[3] not in events_matrixes_channels:
                  events_matrixes_channels.append(event[3])

              events_matrix.append(event)
          events_matrixes.append(events_matrix)
          itrack += 1

      events_matrix1 = []
      for e in events_matrixes:
        events_matrix1.extend(e)

      if verbose:
        if len(events_matrixes_channels) > 16:
          print('MIDI has', len(events_matrixes_channels), 'instruments!', len(events_matrixes_channels) - 16, 'instrument(s) will be removed!')

      for e in events_matrix1:
        if e[0] == 'note' and e[3] != 9:
          if e[3] in events_matrixes_channels[:15]:
            if events_matrixes_channels[:15].index(e[3]) < 9:
              e[3] = events_matrixes_channels[:15].index(e[3])
            else:
              e[3] = events_matrixes_channels[:15].index(e[3])+1
          else:
            events_matrix1.remove(e)
        
        if e[0] in ['patch_change', 'control_change', 'channel_after_touch', 'key_after_touch', 'pitch_wheel_change'] and e[2] != 9:
          if e[2] in [e % 16 for e in events_matrixes_channels[:15]]:
            if [e % 16 for e in events_matrixes_channels[:15]].index(e[2]) < 9:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])
            else:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])+1
          else:
            events_matrix1.remove(e)
    
    else:
      events_matrix1 = []
      itrack = 1
     
      while itrack < len(score):
          for event in score[itrack]:
            events_matrix1.append(event)
          itrack += 1    

    opus = score2opus([score[0], events_matrix1])
    ms_score = opus2score(to_millisecs(opus, pass_old_timings_events=pass_old_timings_events))

    return ms_score

#------------------------ Other Transformations ---------------------

def to_millisecs(old_opus=None, desired_time_in_ms=1, pass_old_timings_events = False):
    r'''Recallibrates all the times in an "opus" to use one beat
per second and one tick per millisecond.  This makes it
hard to retrieve any information about beats or barlines,
but it does make it easy to mix different scores together.
'''
    if old_opus == None:
        return [1000 * desired_time_in_ms,[],]
    try:
        old_tpq  = int(old_opus[0])
    except IndexError:   # 5.0
        _warn('to_millisecs: the opus '+str(type(old_opus))+' has no elements')
        return [1000 * desired_time_in_ms,[],]
    new_opus = [1000 * desired_time_in_ms,]
    # 6.7 first go through building a table of set_tempos by absolute-tick
    ticks2tempo = {}
    itrack = 1
    while itrack < len(old_opus):
        ticks_so_far = 0
        for old_event in old_opus[itrack]:
            if old_event[0] == 'note':
                raise TypeError('to_millisecs needs an opus, not a score')
            ticks_so_far += old_event[1]
            if old_event[0] == 'set_tempo':
                ticks2tempo[ticks_so_far] = old_event[2]
        itrack += 1
    # then get the sorted-array of their keys
    tempo_ticks = []  # list of keys
    for k in ticks2tempo.keys():
        tempo_ticks.append(k)
    tempo_ticks.sort()
    # then go through converting to millisec, testing if the next
    # set_tempo lies before the next track-event, and using it if so.
    itrack = 1
    while itrack < len(old_opus):
        ms_per_old_tick = 400 / old_tpq  # float: will round later 6.3
        i_tempo_ticks = 0
        ticks_so_far = 0
        ms_so_far = 0.0
        previous_ms_so_far = 0.0

        if pass_old_timings_events:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],['old_tpq', 0, old_tpq]]  # new "crochet" is 1 sec
        else:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],]  # new "crochet" is 1 sec
        for old_event in old_opus[itrack]:
            # detect if ticks2tempo has something before this event
            # 20160702 if ticks2tempo is at the same time, leave it
            event_delta_ticks = old_event[1] * desired_time_in_ms
            if (i_tempo_ticks < len(tempo_ticks) and
              tempo_ticks[i_tempo_ticks] < (ticks_so_far + old_event[1]) * desired_time_in_ms):
                delta_ticks = tempo_ticks[i_tempo_ticks] - ticks_so_far
                ms_so_far += (ms_per_old_tick * delta_ticks * desired_time_in_ms)
                ticks_so_far = tempo_ticks[i_tempo_ticks]
                ms_per_old_tick = ticks2tempo[ticks_so_far] / (1000.0*old_tpq * desired_time_in_ms)
                i_tempo_ticks += 1
                event_delta_ticks -= delta_ticks
            new_event = copy.deepcopy(old_event)  # now handle the new event
            ms_so_far += (ms_per_old_tick * old_event[1] * desired_time_in_ms)
            new_event[1] = round(ms_so_far - previous_ms_so_far)

            if pass_old_timings_events:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
              else:
                  new_event[0] = 'old_set_tempo'
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            else:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            ticks_so_far += event_delta_ticks
        new_opus.append(new_track)
        itrack += 1
    _clean_up_warnings()
    return new_opus

#----------------------------- Event stuff --------------------------

_sysex2midimode = {
    "\x7E\x7F\x09\x01\xF7": 1,
    "\x7E\x7F\x09\x02\xF7": 0,
    "\x7E\x7F\x09\x03\xF7": 2,
}

# Some public-access tuples:
MIDI_events = tuple('''note_off note_on key_after_touch
control_change patch_change channel_after_touch
pitch_wheel_change'''.split())

Text_events = tuple('''text_event copyright_text_event
track_name instrument_name lyric marker cue_point text_event_08
text_event_09 text_event_0a text_event_0b text_event_0c
text_event_0d text_event_0e text_event_0f'''.split())

Nontext_meta_events = tuple('''end_track set_tempo
smpte_offset time_signature key_signature sequencer_specific
raw_meta_event sysex_f0 sysex_f7 song_position song_select
tune_request'''.split())
# unsupported: raw_data

# Actually, 'tune_request' is is F-series event, not strictly a meta-event...
Meta_events = Text_events + Nontext_meta_events
All_events  = MIDI_events + Meta_events

# And three dictionaries:
Number2patch = {   # General MIDI patch numbers:
0:'Acoustic Grand',
1:'Bright Acoustic',
2:'Electric Grand',
3:'Honky-Tonk',
4:'Electric Piano 1',
5:'Electric Piano 2',
6:'Harpsichord',
7:'Clav',
8:'Celesta',
9:'Glockenspiel',
10:'Music Box',
11:'Vibraphone',
12:'Marimba',
13:'Xylophone',
14:'Tubular Bells',
15:'Dulcimer',
16:'Drawbar Organ',
17:'Percussive Organ',
18:'Rock Organ',
19:'Church Organ',
20:'Reed Organ',
21:'Accordion',
22:'Harmonica',
23:'Tango Accordion',
24:'Acoustic Guitar(nylon)',
25:'Acoustic Guitar(steel)',
26:'Electric Guitar(jazz)',
27:'Electric Guitar(clean)',
28:'Electric Guitar(muted)',
29:'Overdriven Guitar',
30:'Distortion Guitar',
31:'Guitar Harmonics',
32:'Acoustic Bass',
33:'Electric Bass(finger)',
34:'Electric Bass(pick)',
35:'Fretless Bass',
36:'Slap Bass 1',
37:'Slap Bass 2',
38:'Synth Bass 1',
39:'Synth Bass 2',
40:'Violin',
41:'Viola',
42:'Cello',
43:'Contrabass',
44:'Tremolo Strings',
45:'Pizzicato Strings',
46:'Orchestral Harp',
47:'Timpani',
48:'String Ensemble 1',
49:'String Ensemble 2',
50:'SynthStrings 1',
51:'SynthStrings 2',
52:'Choir Aahs',
53:'Voice Oohs',
54:'Synth Voice',
55:'Orchestra Hit',
56:'Trumpet',
57:'Trombone',
58:'Tuba',
59:'Muted Trumpet',
60:'French Horn',
61:'Brass Section',
62:'SynthBrass 1',
63:'SynthBrass 2',
64:'Soprano Sax',
65:'Alto Sax',
66:'Tenor Sax',
67:'Baritone Sax',
68:'Oboe',
69:'English Horn',
70:'Bassoon',
71:'Clarinet',
72:'Piccolo',
73:'Flute',
74:'Recorder',
75:'Pan Flute',
76:'Blown Bottle',
77:'Skakuhachi',
78:'Whistle',
79:'Ocarina',
80:'Lead 1 (square)',
81:'Lead 2 (sawtooth)',
82:'Lead 3 (calliope)',
83:'Lead 4 (chiff)',
84:'Lead 5 (charang)',
85:'Lead 6 (voice)',
86:'Lead 7 (fifths)',
87:'Lead 8 (bass+lead)',
88:'Pad 1 (new age)',
89:'Pad 2 (warm)',
90:'Pad 3 (polysynth)',
91:'Pad 4 (choir)',
92:'Pad 5 (bowed)',
93:'Pad 6 (metallic)',
94:'Pad 7 (halo)',
95:'Pad 8 (sweep)',
96:'FX 1 (rain)',
97:'FX 2 (soundtrack)',
98:'FX 3 (crystal)',
99:'FX 4 (atmosphere)',
100:'FX 5 (brightness)',
101:'FX 6 (goblins)',
102:'FX 7 (echoes)',
103:'FX 8 (sci-fi)',
104:'Sitar',
105:'Banjo',
106:'Shamisen',
107:'Koto',
108:'Kalimba',
109:'Bagpipe',
110:'Fiddle',
111:'Shanai',
112:'Tinkle Bell',
113:'Agogo',
114:'Steel Drums',
115:'Woodblock',
116:'Taiko Drum',
117:'Melodic Tom',
118:'Synth Drum',
119:'Reverse Cymbal',
120:'Guitar Fret Noise',
121:'Breath Noise',
122:'Seashore',
123:'Bird Tweet',
124:'Telephone Ring',
125:'Helicopter',
126:'Applause',
127:'Gunshot',
}
Notenum2percussion = {   # General MIDI Percussion (on Channel 9):
35:'Acoustic Bass Drum',
36:'Bass Drum 1',
37:'Side Stick',
38:'Acoustic Snare',
39:'Hand Clap',
40:'Electric Snare',
41:'Low Floor Tom',
42:'Closed Hi-Hat',
43:'High Floor Tom',
44:'Pedal Hi-Hat',
45:'Low Tom',
46:'Open Hi-Hat',
47:'Low-Mid Tom',
48:'Hi-Mid Tom',
49:'Crash Cymbal 1',
50:'High Tom',
51:'Ride Cymbal 1',
52:'Chinese Cymbal',
53:'Ride Bell',
54:'Tambourine',
55:'Splash Cymbal',
56:'Cowbell',
57:'Crash Cymbal 2',
58:'Vibraslap',
59:'Ride Cymbal 2',
60:'Hi Bongo',
61:'Low Bongo',
62:'Mute Hi Conga',
63:'Open Hi Conga',
64:'Low Conga',
65:'High Timbale',
66:'Low Timbale',
67:'High Agogo',
68:'Low Agogo',
69:'Cabasa',
70:'Maracas',
71:'Short Whistle',
72:'Long Whistle',
73:'Short Guiro',
74:'Long Guiro',
75:'Claves',
76:'Hi Wood Block',
77:'Low Wood Block',
78:'Mute Cuica',
79:'Open Cuica',
80:'Mute Triangle',
81:'Open Triangle',
}

Event2channelindex = { 'note':3, 'note_off':2, 'note_on':2,
 'key_after_touch':2, 'control_change':2, 'patch_change':2,
 'channel_after_touch':2, 'pitch_wheel_change':2
}

################################################################
# The code below this line is full of frightening things, all to
# do with the actual encoding and decoding of binary MIDI data.

def _twobytes2int(byte_a):
    r'''decode a 16 bit quantity from two bytes,'''
    return (byte_a[1] | (byte_a[0] << 8))

def _int2twobytes(int_16bit):
    r'''encode a 16 bit quantity into two bytes,'''
    return bytes([(int_16bit>>8) & 0xFF, int_16bit & 0xFF])

def _read_14_bit(byte_a):
    r'''decode a 14 bit quantity from two bytes,'''
    return (byte_a[0] | (byte_a[1] << 7))

def _write_14_bit(int_14bit):
    r'''encode a 14 bit quantity into two bytes,'''
    return bytes([int_14bit & 0x7F, (int_14bit>>7) & 0x7F])

def _ber_compressed_int(integer):
    r'''BER compressed integer (not an ASN.1 BER, see perlpacktut for
details).  Its bytes represent an unsigned integer in base 128,
most significant digit first, with as few digits as possible.
Bit eight (the high bit) is set on each byte except the last.
'''
    ber = bytearray(b'')
    seven_bits = 0x7F & integer
    ber.insert(0, seven_bits)  # XXX surely should convert to a char ?
    integer >>= 7
    while integer > 0:
        seven_bits = 0x7F & integer
        ber.insert(0, 0x80|seven_bits)  # XXX surely should convert to a char ?
        integer >>= 7
    return ber

def _unshift_ber_int(ba):
    r'''Given a bytearray, returns a tuple of (the ber-integer at the
start, and the remainder of the bytearray).
'''
    if not len(ba):  # 6.7
        _warn('_unshift_ber_int: no integer found')
        return ((0, b""))
    byte = ba[0]
    ba = ba[1:]
    integer = 0
    while True:
        integer += (byte & 0x7F)
        if not (byte & 0x80):
            return ((integer, ba))
        if not len(ba):
            _warn('_unshift_ber_int: no end-of-integer found')
            return ((0, ba))
        byte = ba[0]
        ba = ba[1:]
        integer <<= 7


def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if _previous_times > 1:
        # E:1176, 0: invalid syntax (<string>, line 1176) (syntax-error) ???
        # print('  previous message repeated '+str(_previous_times)+' times', file=sys.stderr)
        # 6.7
        sys.stderr.write('  previous message repeated {0} times\n'.format(_previous_times))
    elif _previous_times > 0:
        sys.stderr.write('  previous message repeated\n')
    _previous_times = 0
    _previous_warning = ''


def _warn(s=''):
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s) + "\n")
        _previous_warning = s


def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    if str(type(text)).find("'str'") >= 0:  # 6.4 test for back-compatibility
        data = bytes(text, encoding=text_encoding)
    else:
        data = bytes(text)
    return b'\xFF' + bytes((which_kind,)) + _ber_compressed_int(len(data)) + data


def _consistentise_ticks(scores):  # 3.6
    # used by mix_scores, merge_scores, concatenate_scores
    if len(scores) == 1:
        return copy.deepcopy(scores)
    are_consistent = True
    ticks = scores[0][0]
    iscore = 1
    while iscore < len(scores):
        if scores[iscore][0] != ticks:
            are_consistent = False
            break
        iscore += 1
    if are_consistent:
        return copy.deepcopy(scores)
    new_scores = []
    iscore = 0
    while iscore < len(scores):
        score = scores[iscore]
        new_scores.append(opus2score(to_millisecs(score2opus(score))))
        iscore += 1
    return new_scores


###########################################################################
def _decode(trackdata=b'', exclude=None, include=None,
            event_callback=None, exclusive_event_callback=None, no_eot_magic=False):
    r'''Decodes MIDI track data into an opus-style list of events.
The options:
  'exclude' is a list of event types which will be ignored SHOULD BE A SET
  'include' (and no exclude), makes exclude a list
       of all possible events, /minus/ what include specifies
  'event_callback' is a coderef
  'exclusive_event_callback' is a coderef
'''
    trackdata = bytearray(trackdata)
    if exclude == None:
        exclude = []
    if include == None:
        include = []
    if include and not exclude:
        exclude = All_events
    include = set(include)
    exclude = set(exclude)

    # Pointer = 0;  not used here; we eat through the bytearray instead.
    event_code = -1;  # used for running status
    event_count = 0;
    events = []

    while (len(trackdata)):
        # loop while there's anything to analyze ...
        eot = False  # When True, the event registrar aborts this loop
        event_count += 1

        E = []
        # E for events - we'll feed it to the event registrar at the end.

        # Slice off the delta time code, and analyze it
        [time, trackdata] = _unshift_ber_int(trackdata)

        # Now let's see what we can make of the command
        first_byte = trackdata[0] & 0xFF
        trackdata = trackdata[1:]
        if (first_byte < 0xF0):  # It's a MIDI event
            if (first_byte & 0x80):
                event_code = first_byte
            else:
                # It wants running status; use last event_code value
                trackdata.insert(0, first_byte)
                if (event_code == -1):
                    _warn("Running status not set; Aborting track.")
                    return []

            command = event_code & 0xF0
            channel = event_code & 0x0F

            if (command == 0xF6):  # 0-byte argument
                pass
            elif (command == 0xC0 or command == 0xD0):  # 1-byte argument
                parameter = trackdata[0]  # could be B
                trackdata = trackdata[1:]
            else:  # 2-byte argument could be BB or 14-bit
                parameter = (trackdata[0], trackdata[1])
                trackdata = trackdata[2:]

            #################################################################
            # MIDI events

            if (command == 0x80):
                if 'note_off' in exclude:
                    continue
                E = ['note_off', time, channel, parameter[0], parameter[1]]
            elif (command == 0x90):
                if 'note_on' in exclude:
                    continue
                E = ['note_on', time, channel, parameter[0], parameter[1]]
            elif (command == 0xA0):
                if 'key_after_touch' in exclude:
                    continue
                E = ['key_after_touch', time, channel, parameter[0], parameter[1]]
            elif (command == 0xB0):
                if 'control_change' in exclude:
                    continue
                E = ['control_change', time, channel, parameter[0], parameter[1]]
            elif (command == 0xC0):
                if 'patch_change' in exclude:
                    continue
                E = ['patch_change', time, channel, parameter]
            elif (command == 0xD0):
                if 'channel_after_touch' in exclude:
                    continue
                E = ['channel_after_touch', time, channel, parameter]
            elif (command == 0xE0):
                if 'pitch_wheel_change' in exclude:
                    continue
                E = ['pitch_wheel_change', time, channel,
                     _read_14_bit(parameter) - 0x2000]
            else:
                _warn("Shouldn't get here; command=" + hex(command))

        elif (first_byte == 0xFF):  # It's a Meta-Event! ##################
            # [command, length, remainder] =
            #    unpack("xCwa*", substr(trackdata, $Pointer, 6));
            # Pointer += 6 - len(remainder);
            #    # Move past JUST the length-encoded.
            command = trackdata[0] & 0xFF
            trackdata = trackdata[1:]
            [length, trackdata] = _unshift_ber_int(trackdata)
            if (command == 0x00):
                if (length == 2):
                    E = ['set_sequence_number', time, _twobytes2int(trackdata)]
                else:
                    _warn('set_sequence_number: length must be 2, not ' + str(length))
                    E = ['set_sequence_number', time, 0]

            elif command >= 0x01 and command <= 0x0f:  # Text events
                # 6.2 take it in bytes; let the user get the right encoding.
                # text_str = trackdata[0:length].decode('ascii','ignore')
                # text_str = trackdata[0:length].decode('ISO-8859-1')
                # 6.4 take it in bytes; let the user get the right encoding.
                text_data = bytes(trackdata[0:length])  # 6.4
                # Defined text events
                if (command == 0x01):
                    E = ['text_event', time, text_data]
                elif (command == 0x02):
                    E = ['copyright_text_event', time, text_data]
                elif (command == 0x03):
                    E = ['track_name', time, text_data]
                elif (command == 0x04):
                    E = ['instrument_name', time, text_data]
                elif (command == 0x05):
                    E = ['lyric', time, text_data]
                elif (command == 0x06):
                    E = ['marker', time, text_data]
                elif (command == 0x07):
                    E = ['cue_point', time, text_data]
                # Reserved but apparently unassigned text events
                elif (command == 0x08):
                    E = ['text_event_08', time, text_data]
                elif (command == 0x09):
                    E = ['text_event_09', time, text_data]
                elif (command == 0x0a):
                    E = ['text_event_0a', time, text_data]
                elif (command == 0x0b):
                    E = ['text_event_0b', time, text_data]
                elif (command == 0x0c):
                    E = ['text_event_0c', time, text_data]
                elif (command == 0x0d):
                    E = ['text_event_0d', time, text_data]
                elif (command == 0x0e):
                    E = ['text_event_0e', time, text_data]
                elif (command == 0x0f):
                    E = ['text_event_0f', time, text_data]

            # Now the sticky events -------------------------------------
            elif (command == 0x2F):
                E = ['end_track', time]
                # The code for handling this, oddly, comes LATER,
                # in the event registrar.
            elif (command == 0x51):  # DTime, Microseconds/Crochet
                if length != 3:
                    _warn('set_tempo event, but length=' + str(length))
                E = ['set_tempo', time,
                     struct.unpack(">I", b'\x00' + trackdata[0:3])[0]]
            elif (command == 0x54):
                if length != 5:  # DTime, HR, MN, SE, FR, FF
                    _warn('smpte_offset event, but length=' + str(length))
                E = ['smpte_offset', time] + list(struct.unpack(">BBBBB", trackdata[0:5]))
            elif (command == 0x58):
                if length != 4:  # DTime, NN, DD, CC, BB
                    _warn('time_signature event, but length=' + str(length))
                E = ['time_signature', time] + list(trackdata[0:4])
            elif (command == 0x59):
                if length != 2:  # DTime, SF(signed), MI
                    _warn('key_signature event, but length=' + str(length))
                E = ['key_signature', time] + list(struct.unpack(">bB", trackdata[0:2]))
            elif (command == 0x7F):  # 6.4
                E = ['sequencer_specific', time, bytes(trackdata[0:length])]
            else:
                E = ['raw_meta_event', time, command,
                     bytes(trackdata[0:length])]  # 6.0
                # "[uninterpretable meta-event command of length length]"
                # DTime, Command, Binary Data
                # It's uninterpretable; record it as raw_data.

            # Pointer += length; #  Now move Pointer
            trackdata = trackdata[length:]

        ######################################################################
        elif (first_byte == 0xF0 or first_byte == 0xF7):
            # Note that sysexes in MIDI /files/ are different than sysexes
            # in MIDI transmissions!! The vast majority of system exclusive
            # messages will just use the F0 format. For instance, the
            # transmitted message F0 43 12 00 07 F7 would be stored in a
            # MIDI file as F0 05 43 12 00 07 F7. As mentioned above, it is
            # required to include the F7 at the end so that the reader of the
            # MIDI file knows that it has read the entire message. (But the F7
            # is omitted if this is a non-final block in a multiblock sysex;
            # but the F7 (if there) is counted in the message's declared
            # length, so we don't have to think about it anyway.)
            # command = trackdata.pop(0)
            [length, trackdata] = _unshift_ber_int(trackdata)
            if first_byte == 0xF0:
                # 20091008 added ISO-8859-1 to get an 8-bit str
                # 6.4 return bytes instead
                E = ['sysex_f0', time, bytes(trackdata[0:length])]
            else:
                E = ['sysex_f7', time, bytes(trackdata[0:length])]
            trackdata = trackdata[length:]

        ######################################################################
        # Now, the MIDI file spec says:
        #  <track data> = <MTrk event>+
        #  <MTrk event> = <delta-time> <event>
        #  <event> = <MIDI event> | <sysex event> | <meta-event>
        # I know that, on the wire, <MIDI event> can include note_on,
        # note_off, and all the other 8x to Ex events, AND Fx events
        # other than F0, F7, and FF -- namely, <song position msg>,
        # <song select msg>, and <tune request>.
        #
        # Whether these can occur in MIDI files is not clear specified
        # from the MIDI file spec.  So, I'm going to assume that
        # they CAN, in practice, occur.  I don't know whether it's
        # proper for you to actually emit these into a MIDI file.

        elif (first_byte == 0xF2):  # DTime, Beats
            #  <song position msg> ::=     F2 <data pair>
            E = ['song_position', time, _read_14_bit(trackdata[:2])]
            trackdata = trackdata[2:]

        elif (first_byte == 0xF3):  # <song select msg> ::= F3 <data singlet>
            # E = ['song_select', time, struct.unpack('>B',trackdata.pop(0))[0]]
            E = ['song_select', time, trackdata[0]]
            trackdata = trackdata[1:]
            # DTime, Thing (what?! song number?  whatever ...)

        elif (first_byte == 0xF6):  # DTime
            E = ['tune_request', time]
            # What would a tune request be doing in a MIDI /file/?

            #########################################################
            # ADD MORE META-EVENTS HERE.  TODO:
            # f1 -- MTC Quarter Frame Message. One data byte follows
            #     the Status; it's the time code value, from 0 to 127.
            # f8 -- MIDI clock.    no data.
            # fa -- MIDI start.    no data.
            # fb -- MIDI continue. no data.
            # fc -- MIDI stop.     no data.
            # fe -- Active sense.  no data.
            # f4 f5 f9 fd -- unallocated

            r'''
        elif (first_byte > 0xF0) { # Some unknown kinda F-series event ####
            # Here we only produce a one-byte piece of raw data.
            # But the encoder for 'raw_data' accepts any length of it.
            E = [ 'raw_data',
                         time, substr(trackdata,Pointer,1) ]
            # DTime and the Data (in this case, the one Event-byte)
            ++Pointer;  # itself

'''
        elif first_byte > 0xF0:  # Some unknown F-series event
            # Here we only produce a one-byte piece of raw data.
            # E = ['raw_data', time, bytest(trackdata[0])]   # 6.4
            E = ['raw_data', time, trackdata[0]]  # 6.4 6.7
            trackdata = trackdata[1:]
        else:  # Fallthru.
            _warn("Aborting track.  Command-byte first_byte=" + hex(first_byte))
            break
        # End of the big if-group

        ######################################################################
        #  THE EVENT REGISTRAR...
        if E and (E[0] == 'end_track'):
            # This is the code for exceptional handling of the EOT event.
            eot = True
            if not no_eot_magic:
                if E[1] > 0:  # a null text-event to carry the delta-time
                    E = ['text_event', E[1], '']
                else:
                    E = []  # EOT with a delta-time of 0; ignore it.

        if E and not (E[0] in exclude):
            # if ( $exclusive_event_callback ):
            #    &{ $exclusive_event_callback }( @E );
            # else:
            #    &{ $event_callback }( @E ) if $event_callback;
            events.append(E)
        if eot:
            break

    # End of the big "Event" while-block

    return events

###################################################################################
###################################################################################
# This is the beginning of the Discover Search and Filter Python Module
###################################################################################
###################################################################################

ALL_CHORDS_SORTED = [[0], [0, 2], [0, 3], [0, 4], [0, 2, 4], [0, 5], [0, 2, 5], [0, 3, 5], [0, 6],
                    [0, 2, 6], [0, 3, 6], [0, 4, 6], [0, 2, 4, 6], [0, 7], [0, 2, 7], [0, 3, 7],
                    [0, 4, 7], [0, 5, 7], [0, 2, 4, 7], [0, 2, 5, 7], [0, 3, 5, 7], [0, 8],
                    [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8], [0, 6, 8], [0, 2, 4, 8],
                    [0, 2, 5, 8], [0, 2, 6, 8], [0, 3, 5, 8], [0, 3, 6, 8], [0, 4, 6, 8],
                    [0, 2, 4, 6, 8], [0, 9], [0, 2, 9], [0, 3, 9], [0, 4, 9], [0, 5, 9], [0, 6, 9],
                    [0, 7, 9], [0, 2, 4, 9], [0, 2, 5, 9], [0, 2, 6, 9], [0, 2, 7, 9],
                    [0, 3, 5, 9], [0, 3, 6, 9], [0, 3, 7, 9], [0, 4, 6, 9], [0, 4, 7, 9],
                    [0, 5, 7, 9], [0, 2, 4, 6, 9], [0, 2, 4, 7, 9], [0, 2, 5, 7, 9],
                    [0, 3, 5, 7, 9], [0, 10], [0, 2, 10], [0, 3, 10], [0, 4, 10], [0, 5, 10],
                    [0, 6, 10], [0, 7, 10], [0, 8, 10], [0, 2, 4, 10], [0, 2, 5, 10],
                    [0, 2, 6, 10], [0, 2, 7, 10], [0, 2, 8, 10], [0, 3, 5, 10], [0, 3, 6, 10],
                    [0, 3, 7, 10], [0, 3, 8, 10], [0, 4, 6, 10], [0, 4, 7, 10], [0, 4, 8, 10],
                    [0, 5, 7, 10], [0, 5, 8, 10], [0, 6, 8, 10], [0, 2, 4, 6, 10],
                    [0, 2, 4, 7, 10], [0, 2, 4, 8, 10], [0, 2, 5, 7, 10], [0, 2, 5, 8, 10],
                    [0, 2, 6, 8, 10], [0, 3, 5, 7, 10], [0, 3, 5, 8, 10], [0, 3, 6, 8, 10],
                    [0, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10], [1], [1, 3], [1, 4], [1, 5], [1, 3, 5],
                    [1, 6], [1, 3, 6], [1, 4, 6], [1, 7], [1, 3, 7], [1, 4, 7], [1, 5, 7],
                    [1, 3, 5, 7], [1, 8], [1, 3, 8], [1, 4, 8], [1, 5, 8], [1, 6, 8], [1, 3, 5, 8],
                    [1, 3, 6, 8], [1, 4, 6, 8], [1, 9], [1, 3, 9], [1, 4, 9], [1, 5, 9], [1, 6, 9],
                    [1, 7, 9], [1, 3, 5, 9], [1, 3, 6, 9], [1, 3, 7, 9], [1, 4, 6, 9],
                    [1, 4, 7, 9], [1, 5, 7, 9], [1, 3, 5, 7, 9], [1, 10], [1, 3, 10], [1, 4, 10],
                    [1, 5, 10], [1, 6, 10], [1, 7, 10], [1, 8, 10], [1, 3, 5, 10], [1, 3, 6, 10],
                    [1, 3, 7, 10], [1, 3, 8, 10], [1, 4, 6, 10], [1, 4, 7, 10], [1, 4, 8, 10],
                    [1, 5, 7, 10], [1, 5, 8, 10], [1, 6, 8, 10], [1, 3, 5, 7, 10],
                    [1, 3, 5, 8, 10], [1, 3, 6, 8, 10], [1, 4, 6, 8, 10], [1, 11], [1, 3, 11],
                    [1, 4, 11], [1, 5, 11], [1, 6, 11], [1, 7, 11], [1, 8, 11], [1, 9, 11],
                    [1, 3, 5, 11], [1, 3, 6, 11], [1, 3, 7, 11], [1, 3, 8, 11], [1, 3, 9, 11],
                    [1, 4, 6, 11], [1, 4, 7, 11], [1, 4, 8, 11], [1, 4, 9, 11], [1, 5, 7, 11],
                    [1, 5, 8, 11], [1, 5, 9, 11], [1, 6, 8, 11], [1, 6, 9, 11], [1, 7, 9, 11],
                    [1, 3, 5, 7, 11], [1, 3, 5, 8, 11], [1, 3, 5, 9, 11], [1, 3, 6, 8, 11],
                    [1, 3, 6, 9, 11], [1, 3, 7, 9, 11], [1, 4, 6, 8, 11], [1, 4, 6, 9, 11],
                    [1, 4, 7, 9, 11], [1, 5, 7, 9, 11], [1, 3, 5, 7, 9, 11], [2], [2, 4], [2, 5],
                    [2, 6], [2, 4, 6], [2, 7], [2, 4, 7], [2, 5, 7], [2, 8], [2, 4, 8], [2, 5, 8],
                    [2, 6, 8], [2, 4, 6, 8], [2, 9], [2, 4, 9], [2, 5, 9], [2, 6, 9], [2, 7, 9],
                    [2, 4, 6, 9], [2, 4, 7, 9], [2, 5, 7, 9], [2, 10], [2, 4, 10], [2, 5, 10],
                    [2, 6, 10], [2, 7, 10], [2, 8, 10], [2, 4, 6, 10], [2, 4, 7, 10],
                    [2, 4, 8, 10], [2, 5, 7, 10], [2, 5, 8, 10], [2, 6, 8, 10], [2, 4, 6, 8, 10],
                    [2, 11], [2, 4, 11], [2, 5, 11], [2, 6, 11], [2, 7, 11], [2, 8, 11],
                    [2, 9, 11], [2, 4, 6, 11], [2, 4, 7, 11], [2, 4, 8, 11], [2, 4, 9, 11],
                    [2, 5, 7, 11], [2, 5, 8, 11], [2, 5, 9, 11], [2, 6, 8, 11], [2, 6, 9, 11],
                    [2, 7, 9, 11], [2, 4, 6, 8, 11], [2, 4, 6, 9, 11], [2, 4, 7, 9, 11],
                    [2, 5, 7, 9, 11], [3], [3, 5], [3, 6], [3, 7], [3, 5, 7], [3, 8], [3, 5, 8],
                    [3, 6, 8], [3, 9], [3, 5, 9], [3, 6, 9], [3, 7, 9], [3, 5, 7, 9], [3, 10],
                    [3, 5, 10], [3, 6, 10], [3, 7, 10], [3, 8, 10], [3, 5, 7, 10], [3, 5, 8, 10],
                    [3, 6, 8, 10], [3, 11], [3, 5, 11], [3, 6, 11], [3, 7, 11], [3, 8, 11],
                    [3, 9, 11], [3, 5, 7, 11], [3, 5, 8, 11], [3, 5, 9, 11], [3, 6, 8, 11],
                    [3, 6, 9, 11], [3, 7, 9, 11], [3, 5, 7, 9, 11], [4], [4, 6], [4, 7], [4, 8],
                    [4, 6, 8], [4, 9], [4, 6, 9], [4, 7, 9], [4, 10], [4, 6, 10], [4, 7, 10],
                    [4, 8, 10], [4, 6, 8, 10], [4, 11], [4, 6, 11], [4, 7, 11], [4, 8, 11],
                    [4, 9, 11], [4, 6, 8, 11], [4, 6, 9, 11], [4, 7, 9, 11], [5], [5, 7], [5, 8],
                    [5, 9], [5, 7, 9], [5, 10], [5, 7, 10], [5, 8, 10], [5, 11], [5, 7, 11],
                    [5, 8, 11], [5, 9, 11], [5, 7, 9, 11], [6], [6, 8], [6, 9], [6, 10],
                    [6, 8, 10], [6, 11], [6, 8, 11], [6, 9, 11], [7], [7, 9], [7, 10], [7, 11],
                    [7, 9, 11], [8], [8, 10], [8, 11], [9], [9, 11], [10], [11]]

###################################################################################
###################################################################################

def create_files_list(datasets_paths=['./'],
                      files_exts=['.mid', '.midi', '.kar', '.MID', '.MIDI', '.KAR'],
                      randomize_files_list=True,
                      verbose=True
                     ):
    if verbose:
        print('=' * 70)
        print('Searching for files...')
        print('This may take a while on a large dataset in particular...')
        print('=' * 70)

    filez_set = defaultdict(None)

    files_exts = tuple(files_exts)
    
    for dataset_addr in tqdm.tqdm(datasets_paths):
        for dirpath, dirnames, filenames in os.walk(dataset_addr):
            for file in filenames:
                if file not in filez_set and file.endswith(files_exts):
                    filez_set[os.path.join(dirpath, file)] = None
    
    filez = list(filez_set.keys())

    if verbose:
        print('Done!')
        print('=' * 70)
    
    if filez:
        if randomize_files_list:
            
            if verbose:
                print('Randomizing file list...')
                
            random.shuffle(filez)
            
            if verbose:
                print('Done!')
                print('=' * 70)
                
        if verbose:
            print('Found', len(filez), 'files.')
            print('=' * 70)
 
    else:
        if verbose:
            print('Could not find any files...')
            print('Please check dataset dirs and files extensions...')
            print('=' * 70)
        
    return filez

###################################################################################

def check_and_fix_tones_chord(tones_chord):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  for c in tones_chord_combs:
    if c in ALL_CHORDS_SORTED:
      checked_tones_chord = c
      break

  return sorted(checked_tones_chord)

###################################################################################

def chordify_score(score,
                  return_choridfied_score=True,
                  return_detected_score_information=False
                  ):

    if score:
    
      num_tracks = 1
      single_track_score = []
      score_num_ticks = 0

      if type(score[0]) == int and len(score) > 1:

        score_type = 'MIDI_PY'
        score_num_ticks = score[0]

        while num_tracks < len(score):
            for event in score[num_tracks]:
              single_track_score.append(event)
            num_tracks += 1
      
      else:
        score_type = 'CUSTOM'
        single_track_score = score

      if single_track_score and single_track_score[0]:
        
        try:

          if type(single_track_score[0][0]) == str or single_track_score[0][0] == 'note':
            single_track_score.sort(key = lambda x: x[1])
            score_timings = [s[1] for s in single_track_score]
          else:
            score_timings = [s[0] for s in single_track_score]

          is_score_time_absolute = lambda sct: all(x <= y for x, y in zip(sct, sct[1:]))

          score_timings_type = ''

          if is_score_time_absolute(score_timings):
            score_timings_type = 'ABS'

            chords = []
            cho = []

            if score_type == 'MIDI_PY':
              pe = single_track_score[0]
            else:
              pe = single_track_score[0]

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
                ptime = pe[1]
              else:
                time = e[0]
                ptime = pe[0]

              if time == ptime:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

              pe = e

            if len(cho) > 0:
              chords.append(cho)

          else:
            score_timings_type = 'REL'
            
            chords = []
            cho = []

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
              else:
                time = e[0]

              if time == 0:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

            if len(cho) > 0:
              chords.append(cho)

          requested_data = []

          if return_detected_score_information:
            
            detected_score_information = []

            detected_score_information.append(['Score type', score_type])
            detected_score_information.append(['Score timings type', score_timings_type])
            detected_score_information.append(['Score tpq', score_num_ticks])
            detected_score_information.append(['Score number of tracks', num_tracks])
            
            requested_data.append(detected_score_information)

          if return_choridfied_score and return_detected_score_information:
            requested_data.append(chords)

          if return_choridfied_score and not return_detected_score_information:
            requested_data.extend(chords)

          return requested_data

        except Exception as e:
          print('Error!')
          print('Check score for consistency and compatibility!')
          print('Exception detected:', e)

      else:
        return None

    else:
      return None

###################################################################################

def augment_enhanced_score_notes(enhanced_score_notes,
                                  timings_divider=16,
                                  full_sorting=True,
                                  timings_shift=0,
                                  pitch_shift=0,
                                  ceil_timings=False,
                                  round_timings=False,
                                  legacy_timings=True,
                                  sort_drums_last=False
                                ):

    esn = copy.deepcopy(enhanced_score_notes)

    pe = enhanced_score_notes[0]

    abs_time = max(0, int(enhanced_score_notes[0][1] / timings_divider))

    for i, e in enumerate(esn):
      
      dtime = (e[1] / timings_divider) - (pe[1] / timings_divider)

      if round_timings:
        dtime = round(dtime)
      
      else:
        if ceil_timings:
          dtime = math.ceil(dtime)
        
        else:
          dtime = int(dtime)

      if legacy_timings:
        abs_time = int(e[1] / timings_divider) + timings_shift

      else:
        abs_time += dtime

      e[1] = max(0, abs_time + timings_shift)

      if round_timings:
        e[2] = max(1, round(e[2] / timings_divider)) + timings_shift
      
      else:
        if ceil_timings:
          e[2] = max(1, math.ceil(e[2] / timings_divider)) + timings_shift
        else:
          e[2] = max(1, int(e[2] / timings_divider)) + timings_shift
      
      e[4] = max(1, min(127, e[4] + pitch_shift))

      pe = enhanced_score_notes[i]

    if full_sorting:

      # Sorting by patch, reverse pitch and start-time
      esn.sort(key=lambda x: x[6])
      esn.sort(key=lambda x: x[4], reverse=True)
      esn.sort(key=lambda x: x[1])
      
    if sort_drums_last:
        esn.sort(key=lambda x: (x[1], -x[4], x[6]) if x[6] != 128 else (x[1], x[6], -x[4]))

    return esn

###################################################################################

def compute_sustain_intervals(events):

    intervals = []
    pedal_on = False
    current_start = None
    
    for t, cc in events:
        if not pedal_on and cc >= 64:

            pedal_on = True
            current_start = t
        elif pedal_on and cc < 64:

            pedal_on = False
            intervals.append((current_start, t))
            current_start = None

    if pedal_on:
        intervals.append((current_start, float('inf')))

    merged = []
    
    for interval in intervals:
        if merged and interval[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
            
        else:
            merged.append(interval)
            
    return merged

###################################################################################

def apply_sustain_to_ms_score(score):

    sustain_by_channel = {}
    
    for track in score[1:]:
        for event in track:
            if event[0] == 'control_change' and event[3] == 64:
                channel = event[2]
                sustain_by_channel.setdefault(channel, []).append((event[1], event[4]))
    
    sustain_intervals_by_channel = {}
    
    for channel, events in sustain_by_channel.items():
        events.sort(key=lambda x: x[0])
        sustain_intervals_by_channel[channel] = compute_sustain_intervals(events)
    
    global_max_off = 0
    
    for track in score[1:]:
        for event in track:
            if event[0] == 'note':
                global_max_off = max(global_max_off, event[1] + event[2])
                
    for channel, intervals in sustain_intervals_by_channel.items():
        updated_intervals = []
        for start, end in intervals:
            if end == float('inf'):
                end = global_max_off
            updated_intervals.append((start, end))
        sustain_intervals_by_channel[channel] = updated_intervals
        
    if sustain_intervals_by_channel:
        
        for track in score[1:]:
            for event in track:
                if event[0] == 'note':
                    start = event[1]
                    nominal_dur = event[2]
                    nominal_off = start + nominal_dur
                    channel = event[3]
                    
                    intervals = sustain_intervals_by_channel.get(channel, [])
                    effective_off = nominal_off
        
                    for intv_start, intv_end in intervals:
                        if intv_start < nominal_off < intv_end:
                            effective_off = intv_end
                            break
                    
                    effective_dur = effective_off - start
                    
                    event[2] = effective_dur

    return score

###################################################################################

def advanced_score_processor(raw_score, 
                             patches_to_analyze=list(range(129)), 
                             return_score_analysis=False,
                             return_enhanced_score=False,
                             return_enhanced_score_notes=False,
                             return_enhanced_monophonic_melody=False,
                             return_chordified_enhanced_score=False,
                             return_chordified_enhanced_score_with_lyrics=False,
                             return_score_tones_chords=False,
                             return_text_and_lyric_events=False,
                             apply_sustain=False  
                            ):

  '''TMIDIX Advanced Score Processor'''

  # Score data types detection

  if raw_score and type(raw_score) == list:

      num_ticks = 0
      num_tracks = 1

      basic_single_track_score = []

      if type(raw_score[0]) != int:
        if len(raw_score[0]) < 5 and type(raw_score[0][0]) != str:
          return ['Check score for errors and compatibility!']

        else:
          basic_single_track_score = copy.deepcopy(raw_score)
      
      else:
        num_ticks = raw_score[0]
        while num_tracks < len(raw_score):
            for event in raw_score[num_tracks]:
              ev = copy.deepcopy(event)
              basic_single_track_score.append(ev)
            num_tracks += 1

      for e in basic_single_track_score:

          if e[0] == 'note':
              e[3] = e[3] % 16
              e[4] = e[4] % 128
              e[5] = e[5] % 128

          if e[0] == 'patch_change':
              e[2] = e[2] % 16
              e[3] = e[3] % 128

      if apply_sustain:
          apply_sustain_to_ms_score([1000, basic_single_track_score])
          
      basic_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      basic_single_track_score.sort(key=lambda x: x[1])

      enhanced_single_track_score = []
      patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      all_score_patches = []
      num_patch_changes = 0

      for event in basic_single_track_score:
        if event[0] == 'patch_change':
              patches[event[2]] = event[3]
              enhanced_single_track_score.append(event)
              num_patch_changes += 1

        if event[0] == 'note':            
            if event[3] != 9:
              event.extend([patches[event[3]]])
              all_score_patches.extend([patches[event[3]]])
            else:
              event.extend([128])
              all_score_patches.extend([128])

            if enhanced_single_track_score:
                if (event[1] == enhanced_single_track_score[-1][1]):
                    if ([event[3], event[4]] != enhanced_single_track_score[-1][3:5]):
                        enhanced_single_track_score.append(event)
                else:
                    enhanced_single_track_score.append(event)

            else:
                enhanced_single_track_score.append(event)

        if event[0] not in ['note', 'patch_change']:
          enhanced_single_track_score.append(event)

      enhanced_single_track_score.sort(key=lambda x: x[6] if x[0] == 'note' else -1)
      enhanced_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      enhanced_single_track_score.sort(key=lambda x: x[1])

      # Analysis and chordification

      cscore = []
      cescore = []
      chords_tones = []
      tones_chords = []
      all_tones = []
      all_chords_good = True
      bad_chords = []
      bad_chords_count = 0
      score_notes = []
      score_pitches = []
      score_patches = []
      num_text_events = 0
      num_lyric_events = 0
      num_other_events = 0
      text_and_lyric_events = []
      text_and_lyric_events_latin = None

      analysis = {}

      score_notes = [s for s in enhanced_single_track_score if s[0] == 'note' and s[6] in patches_to_analyze]
      score_patches = [sn[6] for sn in score_notes]

      if return_text_and_lyric_events:
        text_and_lyric_events = [e for e in enhanced_single_track_score if e[0] in ['text_event', 'lyric']]
        
        if text_and_lyric_events:
          text_and_lyric_events_latin = True
          for e in text_and_lyric_events:
            try:
              tle = str(e[2].decode())
            except:
              tle = str(e[2])

            for c in tle:
              if not 0 <= ord(c) < 128:
                text_and_lyric_events_latin = False

      if (return_chordified_enhanced_score or return_score_analysis) and any(elem in patches_to_analyze for elem in score_patches):

        cescore = chordify_score([num_ticks, enhanced_single_track_score])

        if return_score_analysis:

          cscore = chordify_score(score_notes)
          
          score_pitches = [sn[4] for sn in score_notes]
          
          text_events = [e for e in enhanced_single_track_score if e[0] == 'text_event']
          num_text_events = len(text_events)

          lyric_events = [e for e in enhanced_single_track_score if e[0] == 'lyric']
          num_lyric_events = len(lyric_events)

          other_events = [e for e in enhanced_single_track_score if e[0] not in ['note', 'patch_change', 'text_event', 'lyric']]
          num_other_events = len(other_events)
          
          for c in cscore:
            tones = sorted(set([t[4] % 12 for t in c if t[3] != 9]))

            if tones:
              chords_tones.append(tones)
              all_tones.extend(tones)

              if tones not in ALL_CHORDS:
                all_chords_good = False
                bad_chords.append(tones)
                bad_chords_count += 1
          
          analysis['Number of ticks per quarter note'] = num_ticks
          analysis['Number of tracks'] = num_tracks
          analysis['Number of all events'] = len(enhanced_single_track_score)
          analysis['Number of patch change events'] = num_patch_changes
          analysis['Number of text events'] = num_text_events
          analysis['Number of lyric events'] = num_lyric_events
          analysis['All text and lyric events Latin'] = text_and_lyric_events_latin
          analysis['Number of other events'] = num_other_events
          analysis['Number of score notes'] = len(score_notes)
          analysis['Number of score chords'] = len(cscore)
          analysis['Score patches'] = sorted(set(score_patches))
          analysis['Score pitches'] = sorted(set(score_pitches))
          analysis['Score tones'] = sorted(set(all_tones))
          if chords_tones:
            analysis['Shortest chord'] = sorted(min(chords_tones, key=len))
            analysis['Longest chord'] = sorted(max(chords_tones, key=len))
          analysis['All chords good'] = all_chords_good
          analysis['Number of bad chords'] = bad_chords_count
          analysis['Bad chords'] = sorted([list(c) for c in set(tuple(bc) for bc in bad_chords)])

      else:
        analysis['Error'] = 'Provided score does not have specified patches to analyse'
        analysis['Provided patches to analyse'] = sorted(patches_to_analyze)
        analysis['Patches present in the score'] = sorted(set(all_score_patches))

      if return_enhanced_monophonic_melody:

        score_notes_copy = copy.deepcopy(score_notes)
        chordified_score_notes = chordify_score(score_notes_copy)

        melody = [c[0] for c in chordified_score_notes]

        fixed_melody = []

        for i in range(len(melody)-1):
          note = melody[i]
          nmt = melody[i+1][1]

          if note[1]+note[2] >= nmt:
            note_dur = nmt-note[1]-1
          else:
            note_dur = note[2]

          melody[i][2] = note_dur

          fixed_melody.append(melody[i])
        fixed_melody.append(melody[-1])

      if return_score_tones_chords:
        cscore = chordify_score(score_notes)
        for c in cscore:
          tones_chord = sorted(set([t[4] % 12 for t in c if t[3] != 9]))
          if tones_chord:
            tones_chords.append(tones_chord)

      if return_chordified_enhanced_score_with_lyrics:
        score_with_lyrics = [e for e in enhanced_single_track_score if e[0] in ['note', 'text_event', 'lyric']]
        chordified_enhanced_score_with_lyrics = chordify_score(score_with_lyrics)
      
      # Returned data

      requested_data = []

      if return_score_analysis and analysis:
        requested_data.append([[k, v] for k, v in analysis.items()])

      if return_enhanced_score and enhanced_single_track_score:
        requested_data.append([num_ticks, enhanced_single_track_score])

      if return_enhanced_score_notes and score_notes:
        requested_data.append(score_notes)

      if return_enhanced_monophonic_melody and fixed_melody:
        requested_data.append(fixed_melody)
        
      if return_chordified_enhanced_score and cescore:
        requested_data.append(cescore)

      if return_chordified_enhanced_score_with_lyrics and chordified_enhanced_score_with_lyrics:
        requested_data.append(chordified_enhanced_score_with_lyrics)

      if return_score_tones_chords and tones_chords:
        requested_data.append(tones_chords)

      if return_text_and_lyric_events and text_and_lyric_events:
        requested_data.append(text_and_lyric_events)

      return requested_data
  
  else:
    return ['Check score for errors and compatibility!']

###################################################################################

def remove_duplicate_pitches_from_escore_notes(escore_notes, 
                                               pitches_idx=4, 
                                               patches_idx=6, 
                                               return_dupes_count=False
                                              ):
    
    cscore = chordify_score([1000, escore_notes])

    new_escore = []

    bp_count = 0

    for c in cscore:
        
        cho = []
        seen = []

        for cc in c:
            if [cc[pitches_idx], cc[patches_idx]] not in seen:
                cho.append(cc)
                seen.append([cc[pitches_idx], cc[patches_idx]])

            else:
                bp_count += 1

        new_escore.extend(cho)
        
    if return_dupes_count:
        return bp_count
        
    else:
        return new_escore

###################################################################################

def fix_monophonic_score_durations(monophonic_score,
                                   min_notes_gap=1,
                                   min_notes_dur=1,
                                   extend_durs=False
                                   ):
  
    fixed_score = []

    if monophonic_score[0][0] == 'note':

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][1]

        if note[1]+note[2] >= nmt:
          note_dur = max(1, nmt-note[1]-min_notes_gap)
        else:
            if extend_durs:
                note_dur = max(1, nmt-note[1]-min_notes_gap)

            else:
                note_dur = note[2]

        new_note = [note[0], note[1], note_dur] + note[3:]
        
        if new_note[2] >= min_notes_dur:
            fixed_score.append(new_note)
      
      if monophonic_score[-1][2] >= min_notes_dur:
          fixed_score.append(monophonic_score[-1])

    elif type(monophonic_score[0][0]) == int:

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][0]

        if note[0]+note[1] >= nmt:
            note_dur = max(1, nmt-note[0]-min_notes_gap)
        else:
            if extend_durs:
                note_dur = max(1, nmt-note[0]-min_notes_gap)

            else:
                note_dur = note[1]
          
        new_note = [note[0], note_dur] + note[2:]
        
        if new_note[1] >= min_notes_dur:
            fixed_score.append(new_note)
      
      if monophonic_score[-1][1] >= min_notes_dur:
          fixed_score.append(monophonic_score[-1]) 

    return fixed_score

###################################################################################

def ordered_groups(data, ptc_idx, pat_idx):
    
    groups = OrderedDict()
    
    for sublist in data:
        key = tuple([sublist[ptc_idx], sublist[pat_idx]])
        
        if key not in groups:
            groups[key] = []
            
        groups[key].append(sublist)
    
    return list(groups.items())

###################################################################################

def fix_escore_notes_durations(escore_notes,
                               min_notes_gap=1,
                               min_notes_dur=1,
                               times_idx=1,
                               durs_idx=2,
                               channels_idx = 3, 
                               pitches_idx=4,
                               patches_idx=6
                              ):

    notes = [e for e in escore_notes if e[channels_idx] != 9]
    drums = [e for e in escore_notes if e[channels_idx] == 9]
    
    escore_groups = ordered_groups(notes, pitches_idx, patches_idx)

    merged_score = []

    for k, g in escore_groups:
        if len(g) > 2:
            fg = fix_monophonic_score_durations(g, 
                                                min_notes_gap=min_notes_gap, 
                                                min_notes_dur=min_notes_dur
                                               )
            merged_score.extend(fg)

        elif len(g) == 2:

            if g[0][times_idx]+g[0][durs_idx] >= g[1][times_idx]:
                g[0][durs_idx] = max(1, g[1][times_idx] - g[0][times_idx] - min_notes_gap)
                
            merged_score.extend(g)

        else:
            merged_score.extend(g)

    return sorted(merged_score + drums, key=lambda x: x[times_idx])

###################################################################################

def get_midi_features_matrixes(path_to_MIDI_file,
                               transpose_factor=6,
                               remove_patches=False,
                               remove_drums=False
                              ):

    """
    Convert a single-track MIDI file into one or more fixed-length feature histograms
    ("source matrices") suitable for downstream ML or analysis tasks.

    Summary
    -------
    This function:
    - Loads a MIDI file and converts it into an internal "enhanced score" representation.
    - Normalizes and augments note events (timings, durations, duplicate-pitch removal).
    - Optionally transposes the score across a small range of semitone shifts.
    - Chordifies the score (groups simultaneous/near-simultaneous events into chord frames).
    - For each chord frame, accumulates counts into a fixed-length vector (length 961)
      that encodes: inter-chord delta-times, note durations, instrument/patch values,
      pitch classes (with separate handling for drums), and recognized chord tokens.
    - Returns a list of these vectors (one per transpose shift attempted).

    Parameters
    ----------
    path_to_MIDI_file : str
        Filesystem path to the MIDI file to process. The function expects a single-track
        MIDI or will convert multi-track input into a single-track enhanced score.
    transpose_factor : int, optional
        Maximum semitone shift to explore in both directions. The value is clipped to
        the range [0, 6]. If zero (default), no transposition is performed and a single
        matrix is returned. If > 0, matrices are produced for each integer transpose
        value in range(-transpose_factor, transpose_factor) (i.e., symmetric shifts).
    remove_patches : bool, optional
        If True, patch/instrument information is omitted from the feature vector.
        If False (default), patch counts are included.
    remove_drums : bool, optional
        If True, drum events (MIDI channel 9) are ignored. If False (default), drum
        events are included and encoded in the pitch region reserved for drums.

    Returns
    -------
    List[List[int]]
        A list of integer vectors (each length 961). Each vector is a histogram-like
        feature representation of the chordified score for one transpose shift.
        The outer list length equals 1 when `transpose_factor == 0`, otherwise it
        equals `2 * transpose_factor` (range(-t, t) excludes the endpoint t).

    Feature vector layout (index ranges)
    -----------------------------------
    The returned vector has fixed length 961. Indices are used as follows:

    - 0 .. 127
      **Delta-time counts**: counts of the inter-chord onset distance (clipped to 0..127).
      Index = clipped_delta_time.

    - 128 .. 255
      **Note duration counts**: counts of note durations (clipped to 1..127).
      Index = 128 + clipped_duration.

    - 256 .. 383
      **Patch / instrument counts** (only if remove_patches is False):
      Index = 256 + clipped_patch (clipped to 0..128).

    - 384 .. 510
      **Melodic pitch counts**: counts of melodic note pitches after optional transpose.
      Pitch values are clipped to 1..127 before indexing.
      Index = 384 + clipped_pitch (where clipped_pitch = max(1, min(127, pitch + transpose))).

    - 513 .. 639
      **Drum pitch counts**: drum note pitches (channel 9) are offset into this
      subrange by adding 128 to the clipped pitch before adding the 384 base:
      Index = 384 + (clipped_drum_pitch + 128) => 513..639.

    - 640 .. (640 + N_chords - 1)
      **Chord token counts**: if the set of pitch classes in a chord (sorted, modulo 12)
      matches an entry in the global `ALL_CHORDS_SORTED` list, the corresponding
      chord token index is incremented at `640 + chord_index`.
      (The code expects `ALL_CHORDS_SORTED` to be defined and indexable.)

    Notes and assumptions
    ---------------------
    - The function relies on several helper functions and global variables that must
      be available in the module scope:
        * `midi2single_track_ms_score(path_to_MIDI_file, do_not_check_MIDI_signature=True)`
        * `advanced_score_processor(raw_score, return_enhanced_score_notes=True)`
        * `augment_enhanced_score_notes(escore, timings_divider=...)`
        * `remove_duplicate_pitches_from_escore_notes(escore)`
        * `fix_escore_notes_durations(escore, min_notes_gap=0)`
        * `chordify_score([1000, escore])`
        * `ALL_CHORDS_SORTED` (list of chord pitch-class sets)
      These helpers must return/accept the expected formats described below.

    - Expected event / chord data format:
      * `chordify_score` returns a list of chord frames; each chord frame `c` is a
        list of event tuples/lists. The code accesses event fields by numeric indices:
          - `e[1]` : event onset time (used to compute delta-time between chord frames)
          - `e[2]` : event duration (used for duration histogram)
          - `e[3]` : event channel (channel == 9 indicates drums)
          - `e[4]` : event pitch (MIDI note number)
          - `e[6]` : event patch/program number (instrument)
        If your event tuples differ, adapt the helper functions or this function.

    - Clipping behavior:
      * Delta-times are clipped to 0..127.
      * Durations are clipped to 1..127 (duration 0 is treated as 1).
      * Pitches used for melodic indexing are clipped to 1..127.
      * Patch values are clipped to 0..128.
      * These clamps ensure indices remain inside the fixed vector bounds.

    - Transposition:
      * For melodic pitches, the transpose offset `tv` is added to the pitch before
        clipping and indexing. Drum pitches are not transposed (drum pitch indexing
        uses the raw pitch and is offset into the drum subrange).

    - Drums:
      * Drum events are detected by `e[3] == 9`. When `remove_drums` is False,
        drum durations, patches (if enabled), and drum pitch counts are included.
      * If a chord frame contains only drum events (no melodic pitches), a delta-time
        count is still recorded for that frame.

    Complexity
    ----------
    - Time complexity is linear in the number of chord frames and events.
    - Memory usage is dominated by the returned list of fixed-size vectors.

    Example
    -------
    >>> matrices = extract_src_matrixes_from_midi("song.mid", transpose_factor=2,
    ...                                           remove_patches=False, remove_drums=False)
    >>> len(matrices)
    4  # transpose_factor=2 -> range(-2,2) => 4 matrices
    >>> len(matrices[0])
    961

    Implementation details
    ----------------------
    - The function enforces `transpose_factor = max(0, min(6, transpose_factor))` so
      the maximum allowed transposition range is +/-6 semitones.
    - The function uses a fixed `timings_divider=32` when augmenting enhanced notes
      (this is applied inside `augment_enhanced_score_notes` in the original flow).
    - If you change the vector layout or the clipping ranges, update downstream
      consumers accordingly.

    Returns
    -------
    list of int lists
        One histogram vector per transpose shift attempted.
    """

    raw_score = midi2single_track_ms_score(path_to_MIDI_file, do_not_check_MIDI_signature=True)

    escore = advanced_score_processor(raw_score, return_enhanced_score_notes=True)

    escore = augment_enhanced_score_notes(escore[0], timings_divider=32)
    
    escore = remove_duplicate_pitches_from_escore_notes(escore)

    escore = fix_escore_notes_durations(escore, min_notes_gap=0)
    
    transpose_factor = max(0, min(6, transpose_factor))

    if transpose_factor > 0:
        
        sidx = -transpose_factor
        eidx = transpose_factor

    else:
        sidx = 0
        eidx = 1

    src_matrixes = []
    
    for tv in range(sidx, eidx):
        
        cscore = chordify_score([1000, escore])
        
        matrix = [0] * 961

        pc = cscore[0]

        for c in cscore:

            pitches = sorted(set([e[4]+tv for e in c if e[3] != 9]))
            drums_present = any(True if e[3] == 9 else False for e in c)

            if len(pitches) == 1:
                
                dtime = max(0, min(127, c[0][1]-pc[0][1]))
                matrix[dtime] += 1

                for e in c:
                    if e[3] != 9:
                        
                        dur = max(1, min(127, e[2]))
                        matrix[dur+128] += 1

                        if not remove_patches:
                            pat = max(0, min(128, e[6]))
                            matrix[pat+256] += 1
                        
                        ptc = max(1, min(127, e[4]+tv))

                        matrix[ptc+384] += 1

            if len(pitches) > 1:
                
                dtime = max(0, min(127, c[0][1]-pc[0][1]))
                matrix[dtime] += 1
                
                tones_chord = sorted(set([p % 12 for p in pitches]))

                if tones_chord in ALL_CHORDS_SORTED:
                    chord_tok = ALL_CHORDS_SORTED.index(tones_chord)
                    matrix[chord_tok+640] += 1
                    
                    for e in c:
                        if e[3] != 9:
                        
                            dur = max(1, min(127, e[2]))
                            matrix[dur+128] += 1

                            if not remove_patches:
                                pat = max(0, min(128, e[6]))
                                matrix[pat+256] += 1
                            
                            ptc = max(1, min(127, e[4]+tv))
    
                            matrix[ptc+384] += 1

            if drums_present and not remove_drums:

                if not pitches:
                    dtime = max(0, min(127, c[0][1]-pc[0][1]))
                    matrix[dtime] += 1

                for e in c:

                    if e[3] == 9:

                        dur = max(1, min(127, e[2]))
                        matrix[dur+128] += 1

                        if not remove_patches:
                            pat = max(0, min(128, e[6]))
                            matrix[pat+256] += 1
                        
                        ptc = max(1, min(127, e[4]))
                        ptc += 128

                        matrix[ptc+384] += 1

            pc = c

        src_matrixes.append(matrix)
        
    return src_matrixes

###################################################################################

def fast_mean_std_gpu(
    arrays: Union[np.ndarray, Iterable[np.ndarray]],
    chunk_rows: int = 200_000,
    device: str = "cuda:0",
    use_gpu: bool = True,
    verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Compute per-dimension mean and std using GPU reductions over row-chunks.
    - arrays: single np.ndarray or iterable of np.ndarray with same D
    - chunk_rows: rows transferred per chunk (tune for PCIe and GPU memory)
    - device: torch device string
    - use_gpu: if False, falls back to CPU chunked method
    Returns (mean, std) as np.float32 numpy arrays.
    """
    
    # Normalize input
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    else:
        arrays = tuple(arrays)

    # Basic checks
    if len(arrays) == 0:
        raise ValueError("Provide at least one array")
    D = arrays[0].shape[1]
    for a in arrays:
        if a.shape[1] != D:
            raise ValueError("All arrays must have same number of columns D")

    # CPU fallback (vectorized chunked) if GPU not desired or unavailable
    if not use_gpu or not torch.cuda.is_available():
        if verbose:
            print("GPU unavailable or disabled — using CPU chunked reduction.")
        total_count = 0
        sum_ = np.zeros(D, dtype=np.float64)
        sumsq = np.zeros(D, dtype=np.float64)
        for arr in arrays:
            N = arr.shape[0]
            for i0 in range(0, N, chunk_rows):
                i1 = min(N, i0 + chunk_rows)
                block = arr[i0:i1].astype(np.float64, copy=False)
                sum_ += block.sum(axis=0)
                sumsq += (block * block).sum(axis=0)
                total_count += (i1 - i0)
        if total_count <= 1:
            mean = sum_.astype(np.float32)
            std = np.ones_like(mean, dtype=np.float32)
            return mean, std
        mean = (sum_ / total_count).astype(np.float32)
        var = (sumsq - (sum_ * sum_) / total_count) / (total_count - 1)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var).astype(np.float32)
        return mean, std

    # GPU path
    dev = torch.device(device)
    # Use double precision accumulators on GPU for stability
    sum_t = torch.zeros(D, dtype=torch.float64, device=dev)
    sumsq_t = torch.zeros(D, dtype=torch.float64, device=dev)
    total_count = 0

    try:
        for arr in arrays:
            N = arr.shape[0]
            # iterate in CPU chunks, transfer each chunk to GPU
            for i0 in range(0, N, chunk_rows):
                i1 = min(N, i0 + chunk_rows)
                block = arr[i0:i1].astype(np.float32, copy=False)
                # transfer to GPU
                b_t = torch.from_numpy(block).to(device=dev, dtype=torch.float32)
                # accumulate in float64
                sum_t += b_t.double().sum(dim=0)
                sumsq_t += (b_t.double() * b_t.double()).sum(dim=0)
                total_count += (i1 - i0)
                # free chunk
                del b_t
                torch.cuda.empty_cache()
        if total_count <= 1:
            mean = sum_t.cpu().numpy().astype(np.float32)
            std = np.ones_like(mean, dtype=np.float32)
            return mean, std

        mean_t = (sum_t / total_count).to(dtype=torch.float64)
        var_t = (sumsq_t - (sum_t * sum_t) / total_count) / (total_count - 1)
        var_t = torch.clamp(var_t, min=1e-12)
        std_t = torch.sqrt(var_t)

        mean = mean_t.cpu().numpy().astype(np.float32)
        std = std_t.cpu().numpy().astype(np.float32)
        return mean, std

    except RuntimeError as e:
        # Provide a helpful OOM hint
        if 'out of memory' in str(e).lower():
            raise RuntimeError(
                "CUDA out of memory during mean/std computation. "
                "Reduce chunk_rows, set use_gpu=False, or free GPU memory."
            ) from e
        raise

###################################################################################

def topk_minkowski_between_gpu(
    X: np.ndarray,
    Y: np.ndarray,
    top_k: int = 10,
    p: float = 2.0,
    q_batch_size: int = 8192,
    y_chunk_size: int = 16384,
    dim_chunk: int = 64,
    mismatch_threshold: Optional[float] = None,
    mismatch_penalty: float = 1.0,
    per_dim_threshold: Optional[np.ndarray] = None,
    per_dim_penalty: Optional[np.ndarray] = None,
    precomputed_mean: Optional[np.ndarray] = None,
    precomputed_std: Optional[np.ndarray] = None,
    scale_vector: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    use_fp16: bool = True,
    device: str = "cuda:0",
    exclude_self: bool = True,
    verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Compute top_k nearest neighbors from X to Y using Minkowski distance with mismatch penalty.
    - X: (Nq, D) np.int32
    - Y: (Ny, D) np.int32
    - precomputed_mean/std: optional np.float32 arrays (D,) to skip streaming standardization
    Returns:
      indices: (Nq, top_k) int64 indices into Y (-1 if fewer than top_k)
      scores:  (Nq, top_k) float32 combined scores (smaller is better)
    """
    
    assert X.dtype == np.int32 and Y.dtype == np.int32, "X and Y must be int32"
    Nq, D = X.shape
    Ny, D2 = Y.shape
    assert D == D2, "X and Y must have same number of features (D)"

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    # Prepare standardization stats
    if precomputed_mean is not None and precomputed_std is not None:
        mean = precomputed_mean.astype(np.float32)
        std = precomputed_std.astype(np.float32)
    else:
        mean = None
        std = None

    # thresholds and penalties
    if per_dim_threshold is not None:
        thr = per_dim_threshold.astype(np.float32)
    elif mismatch_threshold is not None:
        thr = np.full(D, float(mismatch_threshold), dtype=np.float32)
    else:
        thr = None

    if per_dim_penalty is not None:
        pen_vec = per_dim_penalty.astype(np.float32)
    else:
        pen_vec = None
        pen_scalar = float(mismatch_penalty)

    # outputs
    final_inds = -np.ones((Nq, top_k), dtype=np.int64)
    final_scores = np.full((Nq, top_k), np.inf, dtype=np.float32)

    def merge_topk(q_start, ids_np, scores_np):
        Q = ids_np.shape[0]
        for i in range(Q):
            qi = q_start + i
            cur_ids = final_inds[qi]
            cur_scores = final_scores[qi]
            mask = np.isfinite(cur_scores)
            all_ids = np.concatenate([cur_ids[mask], ids_np[i]])
            all_scores = np.concatenate([cur_scores[mask], scores_np[i]])
            if all_scores.size == 0:
                continue
            k = min(top_k, all_scores.size)
            idx = np.argpartition(all_scores, k-1)[:k]
            sel = idx[np.argsort(all_scores[idx])]
            final_inds[qi, :k] = all_ids[sel]
            final_scores[qi, :k] = all_scores[sel]
            if k < top_k:
                final_inds[qi, k:] = -1
                final_scores[qi, k:] = np.inf

    # Estimate normalization constants S_d and S_p using a small CPU sample (safe)
    sample_size = min(2000, max(1, Nq, Ny))
    y_sample_idx = np.random.choice(Ny, min(sample_size, Ny), replace=False)
    sample = Y[y_sample_idx].astype(np.float32)
    if sample.shape[0] < sample_size and Nq > 0:
        add = min(sample_size - sample.shape[0], Nq)
        x_sample_idx = np.random.choice(Nq, add, replace=False)
        sample = np.vstack([sample, X[x_sample_idx].astype(np.float32)])
    if mean is not None and std is not None:
        sample = (sample - mean) / std
    if scale_vector is not None:
        sample = sample / scale_vector.astype(np.float32)

    s = min(256, sample.shape[0])
    if s > 1:
        s_sample = sample[:s]
        pair_vals = []
        for a0 in range(0, s, 64):
            a1 = min(s, a0 + 64)
            A = s_sample[a0:a1]
            dif = np.abs(A[:, None, :] - s_sample[None, :, :]) ** p
            dmat_block = np.sum(dif, axis=2) ** (1.0 / p)
            pair_vals.append(dmat_block.ravel())
        all_vals = np.concatenate(pair_vals)
        all_vals = all_vals[all_vals > 0]
        S_d = float(np.median(all_vals)) if all_vals.size > 0 else 1.0
    else:
        S_d = 1.0
    S_d = max(S_d, 1e-6)
    S_p = 1.0 if pen_vec is None else float(np.median(np.abs(pen_vec)))
    S_p = max(S_p, 1e-6)
    if verbose:
        print(f"Estimated S_d={S_d:.6g}, S_p={S_p:.6g}")

    # Determine if we should mask self matches: only when exclude_self True and X and Y share memory and shapes
    same_memory = False
    try:
        same_memory = (X.__array_interface__['data'][0] == Y.__array_interface__['data'][0]) and (Nq == Ny)
    except Exception:
        same_memory = False
    mask_self_enabled = exclude_self and same_memory

    # Main double loop with feature-dimension chunking
    try:
        for q0 in range(0, Nq, q_batch_size):
            q1 = min(Nq, q0 + q_batch_size)
            Q = q1 - q0
            q_block = X[q0:q1].astype(np.float32)
            if mean is not None and std is not None:
                q_block = (q_block - mean) / std
            if scale_vector is not None:
                q_block = q_block / scale_vector.astype(np.float32)
            q_t_full = torch.from_numpy(q_block).to(device=dev, dtype=torch_dtype)

            for y0 in tqdm.tqdm(range(0, Ny, y_chunk_size), disable=not verbose):
                y1 = min(Ny, y0 + y_chunk_size)
                y_block = Y[y0:y1].astype(np.float32)
                if mean is not None and std is not None:
                    y_block = (y_block - mean) / std
                if scale_vector is not None:
                    y_block = y_block / scale_vector.astype(np.float32)
                y_t_full = torch.from_numpy(y_block).to(device=dev, dtype=torch_dtype)

                # accumulators on GPU in float32
                Cc = y1 - y0
                mink_sum = torch.zeros((Q, Cc), device=dev, dtype=torch.float32)
                penalty_sum = torch.zeros((Q, Cc), device=dev, dtype=torch.float32)

                # iterate over feature chunks
                for f0 in range(0, D, dim_chunk):
                    f1 = min(D, f0 + dim_chunk)
                    q_sub = q_t_full[:, f0:f1]
                    y_sub = y_t_full[:, f0:f1]
                    if use_fp16:
                        q_acc = q_sub.float()
                        y_acc = y_sub.float()
                    else:
                        q_acc = q_sub
                        y_acc = y_sub

                    # compute |q_sub - y_sub|^p and sum over features in this slice
                    diff_chunk = torch.abs(q_acc.unsqueeze(1) - y_acc.unsqueeze(0)).pow(p)
                    mink_sum += diff_chunk.sum(dim=2)

                    # penalty accumulation if needed
                    if thr is not None or pen_vec is not None:
                        diff_raw = torch.abs(q_acc.unsqueeze(1) - y_acc.unsqueeze(0))
                        if thr is not None:
                            thr_slice = torch.from_numpy(thr[f0:f1]).to(device=dev, dtype=torch.float32)
                            exceed = torch.clamp(diff_raw - thr_slice.unsqueeze(0).unsqueeze(0), min=0.0)
                            if pen_vec is not None:
                                pen_slice = torch.from_numpy(per_dim_penalty[f0:f1]).to(device=dev, dtype=torch.float32)
                                penalty_sum += (exceed * pen_slice.unsqueeze(0).unsqueeze(0)).sum(dim=2)
                            else:
                                penalty_sum += exceed.sum(dim=2) * pen_scalar
                        else:
                            pen_slice = torch.from_numpy(per_dim_penalty[f0:f1]).to(device=dev, dtype=torch.float32)
                            mask = (diff_raw > 0.0).float()
                            penalty_sum += (mask * pen_slice.unsqueeze(0).unsqueeze(0)).sum(dim=2)

                    # free small temporaries
                    del diff_chunk
                    if 'diff_raw' in locals():
                        del diff_raw
                    torch.cuda.empty_cache()

                # finalize minkowski distance
                mink_dist = mink_sum.pow(1.0 / p)

                # combined score
                score = alpha * (mink_dist / float(S_d)) + beta * (penalty_sum / float(S_p))

                # mask self matches if enabled
                if mask_self_enabled:
                    q_idx_global = torch.arange(q0, q1, device=dev, dtype=torch.long).unsqueeze(1)
                    y_idx_global = torch.arange(y0, y1, device=dev, dtype=torch.long).unsqueeze(0)
                    self_mask = (q_idx_global == y_idx_global)
                    score[self_mask] = float('inf')

                # topk per chunk
                M = min(top_k, y1 - y0)
                neg = -score
                vals, idxs = torch.topk(neg, k=M, dim=1, largest=True, sorted=True)
                chunk_scores = (-vals).cpu().numpy()
                chunk_ids = (idxs + y0).cpu().numpy().astype(np.int64)

                merge_topk(q0, chunk_ids, chunk_scores)

                # free chunk-level GPU memory
                del y_t_full, mink_sum, penalty_sum, mink_dist, score, neg, vals, idxs
                torch.cuda.empty_cache()

            if verbose:
                print(f"Processed queries up to {q1}/{Nq}")

        return final_inds, final_scores

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            raise RuntimeError(
                "CUDA out of memory. Reduce q_batch_size, y_chunk_size, or dim_chunk; set use_fp16=True; "
                "ensure no other GPU processes are running. I can also provide a multi‑GPU sharded version."
            ) from e
        else:
            raise

###################################################################################

def format_hms(seconds):
    
    """Convert seconds → H:MM:SS."""
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours}h {minutes:02d}m {secs:02d}s"

###################################################################################

def search_and_filter(features_matrixes,
                      features_matrixes_file_names,
                      discover_dir='./Discover-MIDI-Dataset/MIDIs/',
                      master_dir='./Master-MIDI-Dataset/',
                      output_dir='./Output-MIDI-Dataset/',
                      max_number_of_top_k_matches=16,
                      include_master_midis=True,
                      master_midis_transpose_factor=6,
                      remove_patches_from_master_midis=False,
                      remove_drums_from_master_midis=False,
                      p=3.0,
                      chunk_rows=200000,
                      q_batch_size=12,
                      y_chunk_size=8192,
                      dim_chunk=961,
                      mismatch_threshold=None,
                      mismatch_penalty=10.0,
                      alpha=1.0,
                      beta=0.5,
                      use_fp16=False,
                      device="cuda",
                      use_gpu=True,
                      verbose=True
                     ):

    """Search and copy nearest-neighbor MIDI matches from the Discover MIDI Dataset.
    
    Summary
    ----------------
    Perform an efficient, large-scale nearest-neighbor search between a precomputed
    database of MIDI feature vectors and every MIDI file in a master directory, then
    export the matched MIDI files into organized per-master output folders. For each
    master MIDI the routine:
    
    - extracts one or more feature-matrix representations (optionally across a range
      of transpositions),
    - computes normalization statistics (mean and standard deviation) jointly with
      the database to enable normalized distance computations,
    - runs a GPU-accelerated top-k Minkowski search to obtain nearest neighbors and
      similarity scores for each source feature vector,
    - copies the matched MIDI files from the Discover dataset into a dedicated output
      folder named after the master MIDI, prefixing each copied filename with the
      rounded similarity score and the transpose offset.
    
    This function is optimized for throughput and memory efficiency on large datasets:
    it uses chunking for mean/std computation and for the top-k search, supports
    FP16 arithmetic, and exposes multiple parameters to tune batching and memory
    usage for different GPU/CPU environments.
    
    Parameters
    ----------
    features_matrixes : numpy.ndarray
        2D array of precomputed feature vectors for the Discover dataset (shape: N x D).
        Values will be cast to `np.float32` for normalization and distance computation.
    features_matrixes_file_names : Sequence[str]
        Sequence of identifiers (md5-like strings) corresponding to rows in
        `features_matrixes`. Each identifier is used to reconstruct the original
        filename as `<md5>.mid` and to locate the file under
        `discover_dir/<md5[0]>/<md5[1]>/<md5>.mid`.
    discover_dir : str, optional
        Root path to the Discover MIDI dataset (default './Discover-MIDI-Dataset/MIDIs/').
    master_dir : str, optional
        Directory containing master MIDI files to search from (default './Master-MIDI-Dataset/').
    output_dir : str, optional
        Directory where per-master-MIDI result folders will be created
        (default './Output-MIDI-Dataset/').
    max_number_of_top_k_matches : int, optional
        Number of top matches to retrieve per source feature vector (default 16).
    include_master_midis : bool, optional
        If True, copy the original master MIDI into its output folder (default True).
    master_midis_transpose_factor : int, optional
        Maximum semitone transpose range to apply to master MIDIs. Value is
        clamped to [0, 6]. If > 0, the function searches matches for each transpose
        offset in `range(-transpose_factor, transpose_factor)` (default 6).
    remove_patches_from_master_midis : bool, optional
        Remove patches from master MIDIs before search. This is useful when patches
        are not important for desired search results.
    remove_drums_from_master_midis : bool, optional
        Remove drums from master MIDIs before search. This is useful when drums
        are not important for desired search results.
    p : float, optional
        Minkowski distance exponent used by the GPU top-k routine (default 3.0).
    chunk_rows : int, optional
        Row chunk size used by the mean/std GPU routine to control memory usage
        (default 200000).
    q_batch_size : int, optional
        Query batch size for the GPU top-k routine (default 12).
    y_chunk_size : int, optional
        Chunk size for the target (database) vectors in the GPU top-k routine
        (default 8192).
    dim_chunk : int, optional
        Dimensional chunking parameter for the GPU top-k routine to limit per-iteration
        memory (default 961).
    mismatch_threshold : float or None, optional
        Optional threshold used by the distance routine to treat large mismatches
        specially. If None, no thresholding is applied (default None).
    mismatch_penalty : float, optional
        Penalty value applied when mismatches exceed `mismatch_threshold` (default 10.0).
    alpha : float, optional
        Scaling factor applied to normalized distances before ranking (default 1.0).
    beta : float, optional
        Secondary scaling factor used by the distance routine (default 0.5).
    use_fp16 : bool, optional
        If True, use FP16 arithmetic where supported to reduce memory and increase
        throughput (default False).
    device : str, optional
        Device string for computation (e.g., 'cuda' or 'cpu') (default 'cuda').
    use_gpu : bool, optional
        If True, GPU-accelerated routines are used; otherwise CPU fallbacks are used
        (default True).
    verbose : bool, optional
        If True, print progress and diagnostic messages to stdout (default True).
    
    Returns
    -------
    None
        Results are written to disk under `output_dir`. For each master MIDI file
        `X.mid`, a folder `output_dir/X/` is created. Matched files are copied from
        the Discover dataset into that folder with filenames of the form:
        `<similarity>_<transpose_offset>_<md5>.mid`. If `include_original_midis` is True,
        the original master MIDI is also copied into the same folder.
    
    Side effects
    ------------
    - Creates `master_dir` and `output_dir` if they do not exist.
    - Reads master MIDI files from `master_dir`.
    - Copies matched MIDI files from `discover_dir` into `output_dir`.
    - Uses GPU memory and compute when `use_gpu` is True and `device` points to a GPU.
    
    Discover dataset layout expectation
    -----------------------------------
    Files in the Discover dataset must be sharded by the first two characters of the
    md5 identifier. For an md5 identifier `md5`, the expected path is:
    `discover_dir + md5[0] + '/' + md5[1] + '/' + md5 + '.mid'`.
    
    Detailed behavior
    -----------------
    - `transpose_factor` is clamped to the integer range [0, 6]. If `transpose_factor > 0`,
      the function iterates transpose offsets `tv` in `range(-transpose_factor, transpose_factor)`;
      otherwise it uses a single offset 0.
    - For each master MIDI file:
      1. Build a list of source feature matrices using `get_midi_features_matrixes(midi, transpose_factor=...)`.
      2. Convert source matrices to `np.int32` and compute mean/std across the union of
         `features_matrixes` and the source matrices using `fast_mean_std_gpu`.
      3. Call `topk_minkowski_between_gpu` to compute top-k nearest neighbors and similarity
         scores between source matrices and `features_matrixes`. The routine accepts
         `precomputed_mean` and `precomputed_std` to normalize distances.
      4. Create `output_dir/<master_basename>/` and copy matched MIDI files from the
         Discover dataset into that folder. Filenames are prefixed with the rounded
         similarity score and the transpose offset. Duplicate md5 matches for the same
         master MIDI are suppressed using an internal `seen` set.
    
    Error handling
    --------------
    - Per-file copy and lookup operations are wrapped in try/except. On exception the
      function logs the error (when `verbose` is True) and continues processing other matches.
    - The function does not raise for individual copy failures; unhandled exceptions from
      helper utilities or the environment will propagate.
    
    Performance and tuning notes
    ----------------------------
    - Designed to scale to large MIDI datasets by chunking both the mean/std computation
      and the top-k search. Tune `chunk_rows`, `y_chunk_size`, `dim_chunk`, and `q_batch_size`
      to match available GPU memory and desired throughput.
    - Enabling `use_fp16` reduces memory usage and may increase throughput on supported GPUs,
      but can reduce numerical precision.
    - `max_number_of_top_k_matches` increases I/O and disk usage proportionally to the
      number of matches copied per source vector.
    
    Dependencies
    ------------
    The function relies on helper functions and standard libraries being available:
    - `create_files_list(paths)` -> list of file paths under `master_dir`.
    - `get_midi_features_matrixes(midi_path, transpose_factor)` -> list/array of feature vectors.
    - `fast_mean_std_gpu(tuple_of_arrays, chunk_rows, device, use_gpu, verbose)` -> (mean, std).
    - `topk_minkowski_between_gpu(src, target, top_k, p, q_batch_size, y_chunk_size,
      dim_chunk, mismatch_threshold, mismatch_penalty, precomputed_mean, precomputed_std,
      alpha, beta, use_fp16, device, verbose)` -> (indices, scores).
    Also uses standard libraries: `os`, `shutil`, `numpy as np`.
    
    Example
    -------
    >>> # features_matrixes: numpy array shape (N, D)
    >>> # features_matrixes_file_names: list of md5 strings length N
    >>> search_and_filter(features_matrixes,
    ...                   features_matrixes_file_names,
    ...                   discover_dir='./Discover-MIDI-Dataset/MIDIs/',
    ...                   master_dir='./Master-MIDI-Dataset/',
    ...                   output_dir='./Output-MIDI-Dataset/',
    ...                   max_number_of_top_k_matches=8,
    ...                   master_midis_transpose_factor=3,
    ...                   device='cuda',
    ...                   use_gpu=True,
    ...                   verbose=True)
    
    Notes
    -----
    - Ensure `discover_dir` is accessible and that there is sufficient disk space in
      `output_dir` for copied matches.
    - If you change the Discover dataset sharding or naming convention, update the
      file lookup logic accordingly.
    """
    
    if verbose:
        print("=" * 70)
        print('Discover MIDI Dataset Search and Filter')
        print("=" * 70)

    transpose_factor = max(0, min(6, master_midis_transpose_factor))
    
    if transpose_factor > 0:
        tsidx = -transpose_factor
        teidx = transpose_factor
        
    else:
        tsidx = 0
        teidx = 1

    master_midis = create_files_list([master_dir])
    
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    warmup = 5
    total = len(master_midis)
    start_time = time.time()
    durations = []
   
    for fnum, midi in enumerate(master_midis):

        iter_start = time.time()
        
        inp_fn = os.path.basename(midi)

        if verbose:
            print("=" * 70)
            print("Processing MIDI file #", fnum + 1, "/", len(master_midis))
            print("MIDI file name:", inp_fn)
            print("=" * 70)
    
        src_fmatrixes = get_midi_features_matrixes(midi,
                                                   transpose_factor=transpose_factor,
                                                   remove_patches=remove_patches_from_master_midis,
                                                   remove_drums=remove_drums_from_master_midis
                                                  )

        src_fmatrixes = np.array(src_fmatrixes, dtype=np.int32)
        
        if verbose:
            print('Computing mean/std...')

        mean, std = fast_mean_std_gpu((features_matrixes.astype(np.float32),
                                       src_fmatrixes.astype(np.float32)
                                      ),
                                      chunk_rows=chunk_rows,
                                      device=device,
                                      use_gpu=use_gpu,
                                      verbose=verbose
                                     )

        if verbose:    
            print('Calculating distances...')
        
        inds, scores = topk_minkowski_between_gpu(
            src_fmatrixes,
            features_matrixes,
            top_k=max_number_of_top_k_matches,
            p=p,
            q_batch_size=q_batch_size,
            y_chunk_size=y_chunk_size,
            dim_chunk=dim_chunk,
            mismatch_threshold=mismatch_threshold,
            mismatch_penalty=mismatch_penalty,
            precomputed_mean=mean,
            precomputed_std=std,
            alpha=alpha,
            beta=beta,
            use_fp16=use_fp16,
            device=device,
            verbose=verbose
        )
        
        if verbose:
            print('Copying matched MIDIs...')
        
        out_dir = os.path.splitext(inp_fn)[0]

        midi_output_dir = os.path.join(output_dir, out_dir)
        os.makedirs(midi_output_dir, exist_ok=True)

        if include_master_midis:
            shutil.copy2(midi, os.path.join(midi_output_dir, inp_fn))

            

        flat_inds = [x for row in inds.tolist() for x in row]
        flat_scores = [x for row in scores.tolist() for x in row]
        flat_tvs = [v for v in range(tsidx, teidx) for _ in range(max_number_of_top_k_matches)]

        zipped_ist = sorted(zip(flat_inds, flat_scores, flat_tvs), key=lambda x: x[1])
        
        seen = set()
            
        for idx, sim, tv in zipped_ist:

            try:
                md5 = features_matrixes_file_names[idx]
                fn = md5 + '.mid'
                sim = round(sim * 100, 8)

                if md5 not in seen:

                    shutil.copy2(discover_dir + fn[0] + '/' + fn[1] + '/' + fn, midi_output_dir + '/' + str(sim) + '_' + str(tv) + '_' + fn)
                    seen.add(md5)
    
            except Exception as ex:
                if verbose:
                    print('=' * 70)
                    print('Exception error!!!')
                    print(ex)
                    print('File name:', fn)
                    
                    continue
                    
        if verbose:
            print('=' * 70)

        iter_end = time.time()
        
        if fnum < warmup:
            durations.append(iter_end - iter_start)
            
        if fnum == warmup - 1:
            avg_time = sum(durations) / len(durations)
            remaining = total - warmup
            eta_seconds = remaining * avg_time
    
        elapsed = time.time() - start_time
        
        if fnum < warmup:

            if verbose:
                print(f"Elapsed time: {format_hms(elapsed)} (warming up ETA...)")
        
        else:
            remaining = total - (fnum + 1)
            eta = remaining * avg_time
            
            if verbose:
                print(f"Elapsed time: {format_hms(elapsed)}, ETA: {format_hms(eta)}")

    if verbose:    
        print("=" * 70)
        print("Done!")
        print("=" * 70)
        
###################################################################################

def align_feature_vectors(
    src_fnames: Sequence[str],
    trg_fnames: Sequence[str],
    feat_vectors: np.ndarray,
    verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
    
    """
    Align feat_vectors (rows correspond to src_fnames) to the order of trg_fnames.

    Returns:
      - aligned_feat_vectors: np.ndarray with rows ordered to match aligned_fnames.
      - aligned_fnames: list of filenames from trg_fnames that were found in src_fnames,
                        in the same order as aligned_feat_vectors.
      - missing_fnames: list of filenames from trg_fnames that were not found in src_fnames.

    Behavior:
      - If a filename appears multiple times in src_fnames, the last occurrence is used.
      - If no trg_fnames match, aligned_feat_vectors has shape (0, D) where D is feature dim.
    """
    
    # Validate feat_vectors
    if not isinstance(feat_vectors, np.ndarray) or feat_vectors.ndim != 2:
        raise ValueError("feat_vectors must be a 2D numpy array")

    if len(src_fnames) != feat_vectors.shape[0]:
        raise ValueError("Length of src_fnames must equal number of rows in feat_vectors")

    # Build mapping from filename -> last index in src_fnames
    fname_to_idx = {}
    for i, fn in enumerate(src_fnames):
        fname_to_idx[fn] = i

    # Optional duplicate warning
    if verbose and len(set(src_fnames)) != len(src_fnames):
        dup_count = len(src_fnames) - len(set(src_fnames))
        print(f"Warning: {dup_count} duplicate(s) found in src_fnames; using last occurrence for each.")

    aligned_indices: List[int] = []
    aligned_fnames: List[str] = []
    missing_fnames: List[str] = []

    iterator = trg_fnames
    iterator = tqdm.tqdm(trg_fnames, desc="Aligning", unit="file", disable=not verbose)

    for fn in iterator:
        idx = fname_to_idx.get(fn)
        if idx is None:
            missing_fnames.append(fn)
            if verbose and not show_progress:
                print(f"Skipped (not found): {fn}")
        else:
            aligned_indices.append(idx)
            aligned_fnames.append(fn)

    if verbose:
        print(f"Targets processed: {len(trg_fnames)}; Matched: {len(aligned_indices)}; Missing: {len(missing_fnames)}")

    # If no matches, return empty array with correct feature dimension
    if len(aligned_indices) == 0:
        D = feat_vectors.shape[1]
        empty = np.zeros((0, D), dtype=feat_vectors.dtype)
        return empty, [], missing_fnames

    # Gather rows in requested order
    aligned_feat_vectors = feat_vectors[np.array(aligned_indices, dtype=int), :]

    return aligned_feat_vectors, aligned_fnames, missing_fnames

###################################################################################

def load_features_matrixes(all_midis_files_list_path='./Discover-MIDI-Dataset/DATA/Files Lists/all_midis_files_list.jsonl',
                           custom_midis_files_list_path='',
                           features_matrixes_path='./Discover-MIDI-Dataset/DATA/Features Matrixes/features_matrixes.npz',
                           verbose=True
                          ):

    """
    Load feature matrices and their corresponding file identifiers, optionally
    reordering the matrices to match a custom file list.

    This function loads a NumPy `.npz` archive expected to contain an array
    under the key `'features_matrixes'` (shape `(N, D)` where `N` is the number
    of items and `D` is the feature dimensionality). It also reads a JSONL file
    that lists all MIDI files (default `all_midis_files_list_path`) and extracts
    the file identifiers (the `'md5'` field) to produce a parallel list of
    filenames for the loaded feature rows. Optionally, a second JSONL file
    (`custom_midis_files_list_path`) can be provided; when present the function
    aligns and reorders the loaded feature rows so they match the order of the
    custom list. Alignment uses `align_feature_vectors` which matches by filename
    and returns only the rows that were found in the original features array.

    Parameters
    ----------
    all_midis_files_list_path : str, optional
        Path to the JSON Lines file that enumerates the full set of MIDI files
        corresponding to the stored feature matrices. Each line must be a JSON
        object containing an `'md5'` key (used as the filename/identifier).
        Default: `'./Discover-MIDI-Dataset/DATA/Files Lists/all_midis_files_list.jsonl'`.
    custom_midis_files_list_path : str, optional
        Optional path to a JSONL file (same format as above) that defines a
        custom ordering and/or subset of files. When provided, the returned
        feature matrix and filename list are reordered to match this custom
        list; files present in the custom list but missing from the loaded
        features are reported and omitted from the returned matrix. Default: ''.
    features_matrixes_path : str, optional
        Path to the `.npz` archive containing the feature matrices. The archive
        must contain an array under the key `'features_matrixes'`. Default:
        `'./Discover-MIDI-Dataset/DATA/Features Matrixes/features_matrixes.npz'`.
    verbose : bool, optional
        If True, print progress and status messages. If False, suppress printing.
        Default: True.

    Returns
    -------
    tuple
        A tuple `(fmats, fmats_fnames)` where:
        - **fmats** (`np.ndarray`) — 2D array of feature vectors. If a custom
          list is provided, `fmats` contains only the rows that matched the
          custom list and is ordered to match `fmats_fnames`. If no matches
          exist for the custom list, `fmats` will have shape `(0, D)` where `D`
          is the original feature dimensionality.
        - **fmats_fnames** (`List[str]`) — list of file identifiers (the `'md5'`
          values) corresponding to the rows of `fmats`, in the same order.

    Behavior and notes
    ------------------
    - The function expects the `.npz` file to contain the key `'features_matrixes'`.
      If that key is missing, NumPy will raise a `KeyError` when indexing the
      loaded archive.
    - The JSONL readers expect each line to be a JSON object with an `'md5'`
      field. The helper `read_jsonl` is used to parse these files.
    - When `custom_midis_files_list_path` is provided:
        * Only filenames present in both the loaded feature list and the custom
          list are returned (in the order of the custom list).
        * Filenames present in the custom list but not in the loaded features
          are omitted from `fmats` and collected as "missing" by the alignment
          helper (these missing names are not returned by this function but are
          printed by the alignment helper when `verbose` is True).
        * If `src_fnames` (from the `.jsonl` that accompanies the `.npz`) contain
          duplicates, the alignment uses the last occurrence for each filename.
    - The function prints progress/status messages when `verbose` is True.
    - Memory: the entire `'features_matrixes'` array is loaded into memory by
      `np.load(...)` before any optional reordering. If the array is large,
      ensure sufficient memory is available.

    Exceptions
    ----------
    FileNotFoundError
        Raised if `features_matrixes_path` or either JSONL path (when provided)
        does not exist or cannot be opened.
    KeyError
        If the `.npz` archive does not contain the expected `'features_matrixes'`
        key.
    ValueError, OSError, IOError
        Propagated from underlying I/O or parsing functions (e.g., malformed
        JSONL lines, invalid array shapes, or other file I/O errors).
    """

    if verbose:
        print('=' * 70)
        print('Loading features matrixes...')

    fmats = np.load(features_matrixes_path)['features_matrixes']

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Loading features matrixes files names...')

    all_midis_flist = read_jsonl(all_midis_files_list_path, verbose=verbose)
    fmats_fnames = [d['md5'] for d in all_midis_flist]

    if custom_midis_files_list_path:
        if verbose:
            print('Aligning features matrixes to a custom files list...')
            
        custom_midis_flist = read_jsonl(custom_midis_files_list_path, verbose=verbose)
        custom_fnames = [d['md5'] for d in custom_midis_flist]
        res = align_feature_vectors(fmats_fnames, custom_fnames, fmats, verbose=verbose)
        fmats = res[0]
        fmats_fnames = res[1]
        
        if verbose:
            print('=' * 70)
            print('Done!')
            print('=' * 70)

    return fmats, fmats_fnames

###################################################################################

def read_jsonl(file_name='./Discover-MIDI-Dataset/DATA/Files Lists/all_midis_files_list', 
               file_ext='.jsonl',
               max_lines=-1,
               verbose=True
              ):
    
    """Read a JSON Lines (jsonl) file and return its records as a list of Python objects.
    
    This helper opens a `.jsonl` file (optionally appending a file extension if none
    is provided), iterates through each line, parses JSON objects, and accumulates
    them into a list. It supports an optional `max_lines` limit, prints progress
    when `verbose` is True, and tolerates corrupted lines by skipping them while
    continuing processing.
    
    Parameters
    ----------
    file_name : str, optional
        Path to the jsonl file to read. If the provided path has no extension,
        `file_ext` is appended. Default:
        `'./Discover-MIDI-Dataset/DATA/Files Lists/all_midis_files_list'`.
    file_ext : str, optional
        Extension to append when `file_name` has no extension. Default: `'.jsonl'`.
    max_lines : int, optional
        Maximum number of JSON records to read. If `max_lines` is negative (default),
        the function reads the entire file. If positive, reading stops after that
        many successfully parsed records. Default: `-1`.
    verbose : bool, optional
        If True, print progress and status messages and enable a `tqdm` progress
        iterator. If False, suppress progress output. Default: True.
    
    Returns
    -------
    list
        A list of parsed JSON objects (typically dictionaries) in the same order
        they appear in the file. Corrupted lines that fail JSON parsing are skipped
        and not included in the returned list.
    
    Raises
    ------
    FileNotFoundError
        If the resolved `file_name` does not exist when attempting to open it.
    OSError, IOError
        For other I/O related errors when opening or reading the file.
    
    Behavioral details
    ------------------
    - If `file_name` has no extension, `file_ext` is appended before opening.
    - Each line is parsed with `json.loads`. Lines that raise `json.JSONDecodeError`
      are skipped; a brief error message is printed when `verbose` is True.
    - A `KeyboardInterrupt` during iteration is caught: the function prints a
      stopping message (when `verbose` is True), closes the file, and returns the
      records parsed up to that point.
    - The function uses `tqdm` for a progress bar when `verbose` is True; the
      progress bar is disabled when `verbose` is False.
    - The function reads the file line-by-line and keeps only the parsed objects in
      memory; however, the returned list contains all parsed records and may be
      large depending on the file size.
    
    Notes
    -----
    - The function assumes UTF-8 (or the system default) encoding when opening the
      file. If your jsonl uses a different encoding, open the file separately and
      pass a path adjusted for that encoding or modify the function to accept an
      `encoding` parameter.
    - If strict validation or schema enforcement is required, validate each parsed
      record after reading.
    
    """

    if verbose:
        print('=' * 70)
        print('Reading jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    with open(file_name, 'r') as f:

        records = []
        gl_count = 0
        
        for i, line in tqdm.tqdm(enumerate(f), disable=not verbose):
            
            try:
                record = json.loads(line)
                records.append(record)
                gl_count += 1

                if max_lines > 0:
                    if gl_count == max_lines:
                        break

            except KeyboardInterrupt:
                if verbose:
                    print('=' * 70)
                    print('Stoping...')
                    print('=' * 70)
                    
                f.close()
    
                return records
               
            except json.JSONDecodeError:
                if verbose:
                    print('=' * 70)
                    print('[ERROR] Line', i, 'is corrupted! Skipping it...')
                    print('=' * 70)
                    
                continue
                
    f.close()
    
    if verbose:
        print('=' * 70)
        print('Loaded total of', gl_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return records

###################################################################################

def write_jsonl(records_dicts_list, 
                file_name='data', 
                file_ext='.jsonl', 
                file_mode='w', 
                line_sep='\n', 
                verbose=True
               ):
    
    """
    Write a sequence of Python dictionaries (or other JSON-serializable objects)
    to a newline-delimited JSON file (JSONL).

    Each item in `records_dicts_list` is serialized with `json.dumps` and written
    as a single line followed by `line_sep`. The function prints progress and
    summary messages when `verbose` is True and shows a tqdm progress bar while
    writing.

    Parameters
    ----------
    records_dicts_list : Iterable[dict] or Iterable[Any]
        An iterable of records to write. Each record must be JSON-serializable
        by the standard `json` module (for example: dict, list, str, int, float,
        bool, None). The function iterates over this iterable once.
    file_name : str, optional
        Target file path or base name. If `file_name` has no extension, `file_ext`
        is appended. Defaults to `'data'`.
    file_ext : str, optional
        Extension to append to `file_name` when it has no extension. Defaults to
        `'.jsonl'`.
    file_mode : str, optional
        Mode used to open the file (passed to `open`). Typical values:
        `'w'` (overwrite), `'a'` (append), `'x'` (create exclusive). Defaults to
        `'w'`.
    line_sep : str, optional
        Line separator appended after each JSON record. Defaults to newline
        `'\n'`. Use `''` to avoid adding an extra separator if records already
        include one.
    verbose : bool, optional
        If True, print header/footer messages and enable the tqdm progress bar.
        If False, suppress console output and disable the progress bar. Defaults
        to True.

    Returns
    -------
    None
        This function writes to disk and does not return a value.

    Raises
    ------
    TypeError
        If a record is not JSON-serializable, `json.dumps` will raise a
        `TypeError` (or `TypeError`/`ValueError` depending on the object).
    OSError
        If the file cannot be opened or written to (permission issues, invalid
        path, disk full, etc.), the underlying I/O call will raise an `OSError`
        (or a subclass such as `FileNotFoundError`).
    Exception
        Any exception raised by `json.dumps` or the file I/O will propagate to
        the caller.

    Notes
    -----
    - The function uses `os.path.splitext` to detect whether `file_name`
      already contains an extension; if not, `file_ext` is appended.
    - A `tqdm` progress bar is used to show progress when `verbose` is True.
      Ensure `tqdm` is available in the environment or import it as `tqdm`.
    - The function opens the file using a context manager (`with open(...)`)
      which ensures the file is closed on normal exit or when an exception is
      raised. (The explicit `f.close()` after the `with` block is redundant but
      harmless.)
    - Each record is written as a single JSON object per line. This format is
      compatible with many tools that consume JSONL/NDJSON.
    """

    if verbose:
        print('=' * 70)
        print('Writing', len(records_dicts_list), 'records to jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    l_count = 0

    with open(file_name, mode=file_mode) as f:
        for record in tqdm.tqdm(records_dicts_list, disable=not verbose):
            f.write(json.dumps(record) + line_sep)
            l_count += 1

    f.close()

    if verbose:
        print('=' * 70)
        print('Written total of', l_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

###################################################################################

def write_file(data: bytes, target_path: str):

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, "wb") as f:
        f.write(data)

###################################################################################

def parallel_extract(tar_path: str = './Discover-MIDI-Dataset/Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                     extract_path: str = './Discover-MIDI-Dataset/', 
                     max_workers: int = 256, 
                     batch_size: int = 16384
                    ):

    """Extract a large tar.gz archive to disk using parallel writes.
    
    This helper streams a tar.gz archive and extracts its members to `extract_path`
    while performing file writes concurrently with a thread pool. It is optimized
    for very large archives by avoiding loading the entire archive into memory and
    by batching completed write futures to limit memory growth.
    
    Behavior summary
    - Opens the archive in streaming mode (`tarfile.open(..., mode="r|gz")`) so
      members are read sequentially from the archive stream.
    - Creates directories on demand and tracks created directories in `created_dirs`
      to avoid redundant `os.makedirs` calls.
    - Skips extraction for files that already exist at the target path.
    - Reads each file member into memory briefly, then schedules `write_file(data, path)`
      on a `ThreadPoolExecutor` for concurrent disk writes.
    - Flushes and waits for a batch of futures once `batch_size` futures are queued,
      and again at the end of processing to ensure all writes complete.
    
    Parameters
    ----------
    tar_path : str, optional
        Path to the tar.gz archive to extract. Default:
        `'./Discover-MIDI-Dataset/Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz'`.
    extract_path : str, optional
        Destination directory where archive members will be extracted. The directory
        is created if it does not exist. Default: `'./Discover-MIDI-Dataset/'`.
    max_workers : int, optional
        Maximum number of worker threads used for concurrent file writes. Higher
        values increase parallelism but also increase contention and resource use.
        Choose a value appropriate for your system and storage device. Default: 256.
    batch_size : int, optional
        Number of scheduled write futures to accumulate before waiting for them to
        complete. This limits the number of in-flight futures and the memory used
        to buffer file contents. Default: 16384.
    
    Returns
    -------
    None
        The function performs extraction as a side effect and does not return a
        value. All scheduled write tasks are awaited before the function returns.
    
    Dependencies and requirements
    - Requires `tarfile`, `os`, `tqdm`, `concurrent.futures.ThreadPoolExecutor`,
      and a callable `write_file(data: bytes, path: str)` available in scope.
    - `write_file` must be thread-safe and handle creating parent directories if
      necessary (the function already creates directories for `member.isdir()` but
      `write_file` should be robust).
    
    Exceptions
    ----------
    FileNotFoundError
        If `tar_path` does not exist.
    tarfile.ReadError
        If the archive cannot be read as a tar file or is corrupted.
    OSError, IOError, PermissionError
        For I/O errors while reading the archive or writing files to disk.
    RuntimeError
        If a worker thread raises an exception during `write_file`, that exception
        will be propagated when awaiting the future.
    
    Notes and recommendations
    - The archive is opened in streaming mode (`"r|gz"`). This is memory-efficient
      but means random access to archive members is not possible.
    - `batch_size` controls memory usage: each scheduled future holds the file's
      bytes in memory until the future is awaited. Reduce `batch_size` for low-RAM
      environments.
    - `max_workers` should be tuned for the target storage medium. Very large
      values can cause contention and degrade throughput on HDDs or network filesystems.
    - The function uses a hard-coded `tqdm` progress bar with `total=5439450` and
      `miniters=100`. If your archive has a different number of members, consider
      adjusting or removing the `total` argument to avoid misleading progress.
    - The function skips extraction when a target file already exists. If you need
      to overwrite existing files, remove the existence check or add an overwrite flag.
    
    """
    
    os.makedirs(extract_path, exist_ok=True)
    
    created_dirs = set()
    futures = []

    with tarfile.open(tar_path, mode="r|gz") as tar, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:

        for member in tqdm.tqdm(tar, total=5439450, miniters=100):
            
            target_path = os.path.join(extract_path, member.name)
            
            if member.isdir():
                if target_path not in created_dirs:
                    os.makedirs(target_path, exist_ok=True)
                    created_dirs.add(target_path)
                    
            elif member.isfile():

                if os.path.exists(target_path):
                    continue

                fobj = tar.extractfile(member)
                
                if fobj is None:
                    continue
                    
                data = fobj.read()
                futures.append(executor.submit(write_file, data, target_path))

                if len(futures) >= batch_size:
                    for future in as_completed(futures):
                        future.result()
                    futures = []
                    
        for future in as_completed(futures):
            future.result()

###################################################################################

def download_dataset(repo_id='projectlosangeles/Discover-MIDI-Dataset',
                     filename='Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                     local_dir='./Discover-MIDI-Dataset/',
                     verbose=True,
                     **kwargs
                    ):

    """Download the Discover MIDI Dataset archive from the Hugging Face Hub.
    
    This helper wraps `huggingface_hub.hf_hub_download` to fetch a dataset file
    from a specified repository on the Hugging Face Hub and save it to a local
    directory. It returns the absolute path to the downloaded file (or the
    cached file path if the file was already present in the local cache).
    
    Parameters
    ----------
    repo_id : str, optional
        Identifier of the Hugging Face dataset repository in the form
        `'namespace/repo_name'`. Default: `'projectlosangeles/Discover-MIDI-Dataset'`.
    filename : str, optional
        Name of the file to download from the repository (for example, a tarball
        or zip archive). Default:
        `'Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz'`.
    local_dir : str, optional
        Local directory where the downloaded file will be stored or cached.
        The directory will be created by the underlying `hf_hub_download` if
        necessary. Default: `'./Discover-MIDI-Dataset/'`.
    kwargs : dict, optional
        Additional args for hf_hub_download function
    
    Returns
    -------
    str
        Absolute path to the downloaded file on the local filesystem. If the
        file already exists in the Hugging Face cache, the cached path is
        returned.
    
    Raises
    ------
    Exception
        Propagates exceptions raised by `huggingface_hub.hf_hub_download` (for
        example network/HTTP errors, repository or filename not found) and
        standard I/O errors (e.g., `OSError`) that may occur while writing to
        disk.
    
    Notes
    -----
    - This function relies on the `huggingface_hub` package and its
      authentication/caching behavior. If the repository is private, ensure
      that the environment is configured with appropriate HF credentials.
    - `hf_hub_download` may return a cached path instead of re-downloading the
      file if the same file has been previously fetched.
    - Use this helper when you want a simple, single-call way to obtain the
      dataset archive and receive its local path for subsequent processing.

    """

    print('=' * 70)
    print('Downloading dataset...')
    print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='dataset',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    return result

###################################################################################

def render_midi(input_midi_file,
                output_wav_file='',
                sf2_path='./Discover-MIDI-Dataset/SOUNDFONTS/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                verbose=True
               ):
    
    """Render a MIDI file to a WAV file using a SoundFont and a MIDI renderer.
    
    This helper reads a SoundFont (`.sf2`) and a MIDI file, uses the `midirenderer`
    interface to synthesize audio, and writes the resulting waveform to disk.
    It is a thin wrapper around `midirenderer.render_wave_from` that handles
    file I/O, default output naming, and optional progress printing.
    
    Parameters
    ----------
    input_midi_file : str or pathlib.Path
        Path to the input MIDI file to render. The file will be read as bytes
        and passed to the renderer.
    output_wav_file : str, optional
        Path where the rendered WAV will be saved. If empty or not provided,
        the function will create a file next to `input_midi_file` with the same
        base name and a `.wav` extension (for example, `song.mid` -> `song.wav`).
        Default: '' (auto-derived from `input_midi_file`).
    sf2_path : str or pathlib.Path, optional
        Path to the SoundFont (`.sf2`) file used for synthesis. The file is
        read as bytes and supplied to the renderer. Default:
        `'./Discover-MIDI-Dataset/SOUNDFONTS/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'`.
    verbose : bool, optional
        If True, print simple progress messages to stdout. Default: True.
    
    Returns
    -------
    str
        The path to the saved WAV file (the same value as `output_wav_file` after
        any default naming logic is applied).
    
    Raises
    ------
    FileNotFoundError
        If `input_midi_file` or `sf2_path` does not exist when attempting to read.
    OSError, IOError, PermissionError
        If there is an error reading the input files or writing the output WAV.
    ValueError, RuntimeError
        If `midirenderer.render_wave_from` fails or returns invalid data.
    TypeError
        If the renderer returns a non-bytes object or if provided arguments are
        of incompatible types.
    
    Notes
    -----
    - This function expects a `midirenderer` object in scope that exposes the
      method `render_wave_from(sf2_bytes: bytes, midi_bytes: bytes) -> bytes`.
      The renderer must accept raw bytes for the SoundFont and MIDI and return
      raw WAV bytes suitable for writing directly to a `.wav` file.
    - The function reads the entire SoundFont and MIDI into memory before
      rendering. For very large SoundFonts or constrained environments, ensure
      sufficient memory is available.
    - The function writes the returned bytes directly to disk without additional
      WAV header manipulation; the renderer is expected to return a complete,
      valid WAV byte stream.
    - If you need to overwrite existing files, ensure `output_wav_file` points to
      the desired path; this function will overwrite without prompting.
    
    """
    if verbose:
        print('=' * 70)
        print('Rendering MIDI...')
        
    wav_data = midirenderer.render_wave_from(
        Path(sf2_path).read_bytes(),
        Path(input_midi_file).read_bytes()
    )

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Saving rendered MIDI...')

    if not output_wav_file:
        output_wav_file = os.path.splitext(input_midi_file)[0] + '.wav'
    
    with open(output_wav_file, 'wb') as fi:
        fi.write(wav_data)

    if verbose:
        print('Done!')
        print('=' * 70)

    return output_wav_file

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Discover_search_and_filter Python module
###################################################################################