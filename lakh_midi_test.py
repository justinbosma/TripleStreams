import pretty_midi
import os
import pytest
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lakh_midi

lakh_test_path = '/Users/justinbosma/Desktop/TripleStreams/lakh_midi/lmd_test_data/lmd_full'
separated_path = '/Users/justinbosma/Desktop/TripleStreams/lakh_midi/lmd_test_data/lmd_separate'
file_1 = '3646443731743766be902dbfae10cc09.mid'
file_1_strip = file_1.strip('.')[0]

piano_1 = pretty_midi.Instrument(
        program=2
    )
note_a_1 = pretty_midi.Note(pitch = 36, velocity=20, start=0.1, end=0.2)
piano_1.notes.append(note_a_1)
note_a_2 = pretty_midi.Note(pitch = 36, velocity=20, start=0.21, end=0.3)
piano_1.notes.append(note_a_2)
note_a_3 = pretty_midi.Note(pitch = 36, velocity=20, start=0.31, end=0.4)
piano_1.notes.append(note_a_3)
note_a_4 = pretty_midi.Note(pitch = 36, velocity=20, start=0.41, end=0.5)
piano_1.notes.append(note_a_4)
note_a_5 = pretty_midi.Note(pitch = 36, velocity=20, start=0.51, end=0.6)
piano_1.notes.append(note_a_5)
note_a_6 = pretty_midi.Note(pitch = 36, velocity=20, start=0.61, end=0.7)
piano_1.notes.append(note_a_6)
note_a_7 = pretty_midi.Note(pitch = 36, velocity=20, start=0.71, end=0.8)
piano_1.notes.append(note_a_7)
note_a_8 = pretty_midi.Note(pitch = 36, velocity=20, start=0.81, end=0.9)
piano_1.notes.append(note_a_8)
pm_test_global = pretty_midi.PrettyMIDI()
pm_test_global.instruments.append(piano_1)
ts = pretty_midi.containers.TimeSignature(numerator=3, denominator=4, time=0)
pm_test_global.time_signature_changes.append(ts)

# Tests
# 1. split midi into individual files -> check exists and name is correct
#   a) check has correct tempo infor
#   b) check has correct number of notes for instruments
# 2. Merge instruments into family
# a) all pianos go to piano main
# check note list length, etc

#Some files to use
# 1.This one has multiple drum instruments 
# # pretty_midi.PrettyMIDI('lakh_midi/lmd_full/8/80ea068d7bed7c2af7b36bd77cce5b3c.mid')
# 2. This one has 
# pm_bass_testy_sep = pretty_midi.PrettyMIDI('lakh_midi/lmd_separate/1d7ff7f2b6b4001449960fc8529ba468/1d7ff7f2b6b4001449960fc8529ba468_bass.mid')
# pm_bass_testy = pretty_midi.PrettyMIDI('lakh_midi/lmd_full/1/1d7ff7f2b6b4001449960fc8529ba468.mid') 


# /Users/justinbosma/Desktop/TripleStreams/lakh_midi/lmd_test_data/lmd_full/3646443731743766be902dbfae10cc09.mid
def test_separate_drum_instruments_merged(file_full, file_separate, dir):
    drum_separate = pretty_midi.PrettyMIDI(dir + '/lmd_separate/80ea068d7bed7c2af7b36bd77cce5b3c/80ea068d7bed7c2af7b36bd77cce5b3c_drum.mid')
    drum_full = pretty_midi.PrettyMIDI('/Users/justinbosma/Desktop/TripleStreams/lakh_midi/lmd_test_data/lmd_full/80ea068d7bed7c2af7b36bd77cce5b3c.mid')


def test_merge_instruments():
    pm_test_1 = pretty_midi.PrettyMIDI(lakh_test_path + '/' + file_1)
    num_notes = 0
    for instrument in pm_test_1.instruments:
        num_notes += len(instrument.notes)
    instrument_list = lakh_midi.midi_instrument_mapping['Guitar']
    instruments_to_combine = lakh_midi.get_instruments(pm_test_1, instrument_list)
    merged_instruments = lakh_midi.merge_instruments(instruments_to_combine, 'Guitar')
    assert(len(merged_instruments.instruments[0]) == num_notes)
    assert(len(merged_instruments.instruments) == 1)


def test_split_into_instrument_midi():
    pass

def test_correct_meter():
    assert not (lakh_midi.correct_meter(pm_test_global))


def test_correct_velocity():
    assert not (lakh_midi.correct_velocity(pm_test_global))

def test_get_instruments():
    pm_test_1 = pretty_midi.PrettyMIDI(lakh_test_path + '/' + file_1)
    instrument_list = lakh_midi.midi_instrument_mapping['Guitar']
    instruments_to_combine = lakh_midi.get_instruments(pm_test_1, instrument_list)
    assert(len(instruments_to_combine)==2)
    assert(instruments_to_combine[0].program in [24, 25, 26, 27, 28, 29, 30, 31])
    assert(instruments_to_combine[1].program in [24, 25, 26, 27, 28, 29, 30, 31])
    assert(instruments_to_combine[0].program not in [0, 1, 2, 3, 4, 5, 6, 7])


def test_create_separate_midi_tracks_lakh():
    
    lakh_midi.create_separate_midi_tracks_lakh(lakh_test_path)
    assert(os.path.exists(separated_path + '/' + file_1_strip + '_drum.mid'))
    assert(os.path.exists(separated_path + '/' + file_1_strip + '_guitar.mid'))
    assert(os.path.exists(separated_path + '/' + file_1_strip + '_piano.mid'))
    assert(os.path.exists(separated_path + '/' + file_1_strip + '_bass.mid'))