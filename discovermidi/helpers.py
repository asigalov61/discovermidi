#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Helpers Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
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
'''

###################################################################################

import os
import hashlib

from .TMIDIX import midi2score, score2midi
from .TMIDIX import midi2single_track_ms_score, advanced_score_processor
from .TMIDIX import escore_notes_monoponic_melodies, Number2patch

###################################################################################

def get_normalized_midi_md5_hash(midi_file):

    bfn = os.path.basename(midi_file)
    fn = os.path.splitext(bfn)[0]
    
    midi_data = open(midi_file, 'rb').read()
    
    old_md5 = hashlib.md5(midi_data).hexdigest()
    
    score = midi2score(midi_data, do_not_check_MIDI_signature=True)
        
    norm_midi = score2midi(score)
    
    new_md5 = hashlib.md5(norm_midi).hexdigest()
    
    output_dic = {'midi_name': fn,
                  'original_md5': old_md5,
                  'normalized_md5': new_md5
                 }

    return output_dic

###################################################################################

def normalize_midi_file(midi_file, output_dir='', output_file_name=''):

    if not output_file_name:
        output_file_name = os.path.basename(midi_file)

    if not output_dir:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    midi_path = os.path.join(output_dir, output_file_name)

    if os.path.exists(midi_path):
        fn = os.path.splitext(output_file_name)[0]
        output_file_name = f'{fn}_normalized.mid'
        midi_path = os.path.join(output_dir, output_file_name)
    
    midi_data = open(midi_file, 'rb').read()
    
    score = midi2score(midi_data, do_not_check_MIDI_signature=True)
        
    norm_midi = score2midi(score)

    with open(midi_path, 'wb') as fi:
        fi.write(norm_midi)

    return midi_path

###################################################################################

def check_for_monophonic_melodies(midi_file,
                                  mono_mels_bad_notes_ratio=0.0,
                                  return_human_readable=True
                                 ):

    """
    Detect and report monophonic (single-voice) melody occurrences in a MIDI file.

    This function converts a MIDI file into a single-track score, runs an enhanced
    score processor, and extracts detected monophonic melodies from the first
    enhanced-score track. Results can be returned either as raw (numeric) tuples
    or as a human-readable list of dictionaries mapping instrument patch names to
    occurrence counts.

    Parameters
    ----------
    midi_file : str or file-like object
        Path to a MIDI file or a file-like object containing MIDI data. The
        implementation calls `TMIDIX.midi2single_track_ms_score(...)` to produce
        a single-track score; ensure the provided object is acceptable to that
        function. Note: the current implementation references `input_midi` when
        calling TMIDIX; confirm that `midi_file` is passed correctly to avoid a
        NameError.

    mono_mels_bad_notes_ratio : float, optional
        Threshold ratio (between 0.0 and 1.0) used by
        `TMIDIX.escore_notes_monoponic_melodies(...)` to determine how tolerant
        the detection is to "bad" or non-monophonic notes inside an otherwise
        monophonic segment. Lower values make the detector stricter (fewer
        tolerated bad notes). Default is 0.0 (no bad notes allowed).

    return_human_readable : bool, optional
        If True (default), the function returns a list of dictionaries with
        human-friendly patch (instrument) names and counts:
            [{'patch': <patch_name>, 'count': <occurrence_count>}, ...]
        If False, the function returns the raw output from
        `TMIDIX.escore_notes_monoponic_melodies(...)`, which is a sequence of
        `[patch_number, count]` lists.

    Returns
    -------
    list of dict or list of tuple or None
        - If `return_human_readable` is True and monophonic melodies are found:
          a list of dictionaries: `{'patch': <str>, 'count': <int>}`.
        - If `return_human_readable` is False and monophonic melodies are found:
          a list of lists: `[patch_number: int, count: int]`.
        - If no enhanced-score notes are produced or no monophonic melodies are
          detected, the function returns empty list [].

    """

    raw_score = midi2single_track_ms_score(midi_file, do_not_check_MIDI_signature=True)
    
    escore_notes = advanced_score_processor(raw_score, return_enhanced_score_notes=True)
    
    if escore_notes and escore_notes[0]:
    
        mono_mels = escore_notes_monoponic_melodies(escore_notes[0],
                                                    bad_notes_ratio=mono_mels_bad_notes_ratio
                                                   )

        if return_human_readable:

            return_list = []

            for patch, count in mono_mels:
                return_list.append({'patch': Number2patch[patch],
                                    'count': count                 
                                   })  

            return return_list

        else:
            return mono_mels

    else:
        return []

###################################################################################
# This is the end of the Helpers Python Module
###################################################################################