import pretty_midi
import os
#import hvo_sequence
import midi_helper
import sys
from multiprocessing import Pool

# Dictionaries for merging drums via the note pitch

# Hi_MID_LO_MAPPING separates the drums into freqency bins
HI_MID_LO_MAPPING = {
    #Low is kick, low toms
    "LOW": [35, 36, 43, 58, 41],
    #MID is snares, mid/hi tom 
    "MID": [38, 37, 40, 39, 47, 45, 50, 48],
    #High is HiHats and Rides and crash
    "HIGH": [42, 22, 44, 46, 26, 49, 52, 55, 57, 51, 53, 59]

}


# VELOCITY_MAPPING separates the drums into velocity bins
VELOCITY_MAPPING = {
    #Low is velocities below
    "LOW": 43, #43
    #Mid is velocities between LOW and HIGH
    #High is velocty above
    "HIGH": 70
}


#This could be using kick as groove - diff from others where we just flatten 3 streams for groove
FUNCTIONAL_MAPPING = {
    "KICK": [36, 35],
    "TOMS": [43, 58, 41, 47, 45, 50, 48],
    "SNARE": [38, 37, 40, 39],
    "HH": [42, 22, 44, 46, 26]
}

#This maps out the hats/cymbals
FUNCTIONAL_MAPPING_HATS = {
    'CRASH': [49, 52, 55, 57],
    'RIDE': [51, 53, 59],
    'HH_CLOSED': [42, 22, 44],
    'HH_OPEN': [46, 26],
}

TOMS_AND_RIDES = {
    "TOM_LO": [43, 58, 41],
    "TOM_MID": [47, 45],
    "TOM_HI": [50, 48],
    "RIDE": [51, 53, 59]
}
   



#Function for getting the average velocity of all notes in a PrettyMIDI object
def get_note_velocity(pm):
    """
    Returns the average velocity of all notes in a PrettyMIDI object.
    """
    low_velocities = []
    mid_velocities = []
    high_velocities = []

    # Iterate through all instruments and their notes to collect velocities
    for note in pm.instruments[0].notes:
        if note.velocity < VELOCITY_MAPPING["LOW"]:
            low_velocities.append(note)
        elif note.velocity > VELOCITY_MAPPING["HIGH"]:
            high_velocities.append(note)
        else:
            mid_velocities.append(note)

    return low_velocities, mid_velocities, high_velocities


def write_velocity_midi(pm, low_velocity, mid_velocity, high_velocity, file, output_directory):

    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each density category - Add list of tempos and change times
    midi_low = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_low.time_signature_changes.append(ts)
    midi_low._tempo_change_times = tempo_times
    midi_low._tempos = tempi

    midi_mid = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_mid.time_signature_changes.append(ts)
    midi_mid._tempo_change_times = tempo_times
    midi_mid._tempos = tempi

    midi_high = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_high.time_signature_changes.append(ts)
    midi_high._tempo_change_times = tempo_times
    midi_high._tempos = tempi

    midi_groove = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_groove.time_signature_changes.append(ts)
    midi_groove._tempo_change_times = tempo_times
    midi_groove._tempos = tempi

    # Create instruments for each density category
    low_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Low Velocity Instrument')
    mid_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Mid Velocity Instrument')
    high_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='High Velocity Instrument')
    groove_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Groove Instrument')

    

    # Add notes to the respective instruments
    for note in low_velocity:
        low_instrument.notes.append(note)
        groove_instrument.notes.append(note)
    for note in mid_velocity:
        mid_instrument.notes.append(note)
        groove_instrument.notes.append(note)
    for note in high_velocity:
        high_instrument.notes.append(note)
        groove_instrument.notes.append(note)

    # Add instruments to the MIDI objects
    midi_low.instruments.append(low_instrument)
    midi_mid.instruments.append(mid_instrument)
    midi_high.instruments.append(high_instrument)
    midi_groove.instruments.append(groove_instrument)

    #Only write if all list have values
    if (low_instrument.notes and mid_instrument.notes and high_instrument.notes):
        # Save the MIDI files
        midi_low.write(os.path.join(output_directory, file_name + '_velocity_low.mid'))
        midi_mid.write(os.path.join(output_directory, file_name + '_velocity_mid.mid'))
        midi_high.write(os.path.join(output_directory, file_name + '_velocity_high.mid'))
        midi_groove.write(os.path.join(output_directory, file_name + '_velocity_groove.mid'))


def write_lo_mid_hi_midi(pm, file, output_directory):
    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each density category - Add list of tempos and change times
    midi_low = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_low._tempo_change_times = tempo_times
    midi_low._tempos = tempi
    midi_low.time_signature_changes.append(ts)

    midi_mid = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_mid._tempo_change_times = tempo_times
    midi_mid._tempos = tempi
    midi_mid.time_signature_changes.append(ts)

    midi_high = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_high._tempo_change_times = tempo_times
    midi_high._tempos = tempi
    midi_high.time_signature_changes.append(ts)

    midi_groove = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_groove._tempo_change_times = tempo_times
    midi_groove._tempos = tempi
    midi_groove.time_signature_changes.append(ts)

    # Create instruments for each density category
    low_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Low Frequency Instrument')
    mid_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Mid Frequency Instrument')
    high_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='High Frequency Instrument')
    groove_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Groove Frequency Instrument')

    for note in pm.instruments[0].notes:
        groove_instrument.notes.append(note)
        if note.pitch in HI_MID_LO_MAPPING["LOW"]:
            low_instrument.notes.append(note)
        elif note.pitch in HI_MID_LO_MAPPING["MID"]:
            mid_instrument.notes.append(note)
        elif note.pitch in HI_MID_LO_MAPPING["HIGH"]:
            high_instrument.notes.append(note)

    # Add instruments to the MIDI objects
    midi_low.instruments.append(low_instrument)
    midi_mid.instruments.append(mid_instrument)
    midi_high.instruments.append(high_instrument)
    midi_groove.instruments.append(groove_instrument)
    
    #check if one of the instruments is empty, if so, do not write file
    if (low_instrument.notes and mid_instrument.notes and high_instrument.notes):        
        # Save the MIDI files
        midi_low.write(os.path.join(output_directory, file_name + '_pitch_low.mid'))
        midi_mid.write(os.path.join(output_directory, file_name + '_pitch_mid.mid'))
        midi_high.write(os.path.join(output_directory, file_name + '_pitch_high.mid'))
        midi_groove.write(os.path.join(output_directory, file_name + '_pitch_groove.mid'))

def write_functional_midi(pm, file, output_directory):
    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each freqency category - Add list of tempos and change times
    midi_kick = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_kick._tempo_change_times = tempo_times
    midi_kick._tempos = tempi
    midi_kick.time_signature_changes.append(ts)

    midi_toms = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_toms._tempo_change_times = tempo_times
    midi_toms._tempos = tempi
    midi_toms.time_signature_changes.append(ts)

    midi_snare = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_snare._tempo_change_times = tempo_times
    midi_snare._tempos = tempi
    midi_snare.time_signature_changes.append(ts)

    midi_hh = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_hh._tempo_change_times = tempo_times
    midi_hh._tempos = tempi
    midi_hh.time_signature_changes.append(ts)

    # Create instruments for each frequency category
    kick_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Kick')
    toms_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Toms')
    snare_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Snare')
    hh_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional HiHat')

    for note in pm.instruments[0].notes:
        if note.pitch in FUNCTIONAL_MAPPING["KICK"]:
            kick_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING["TOMS"]:
            toms_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING["SNARE"]:
            snare_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING["HH"]:
            hh_instrument.notes.append(note)

    # Add instruments to the MIDI objects
    midi_kick.instruments.append(kick_instrument)
    midi_toms.instruments.append(toms_instrument)
    midi_snare.instruments.append(snare_instrument)
    midi_hh.instruments.append(hh_instrument)

    #check if one of the instruments is empty, if so, do not write file
    if (kick_instrument.notes and toms_instrument.notes and snare_instrument.notes and hh_instrument.notes):
        # Save the MIDI files
        midi_kick.write(os.path.join(output_directory, file_name + '_functional_kick.mid'))
        midi_toms.write(os.path.join(output_directory, file_name + '_functional_toms.mid'))
        midi_snare.write(os.path.join(output_directory, file_name + '_functional_snare.mid'))
        midi_hh.write(os.path.join(output_directory, file_name + '_functional_hh.mid'))

def write_functional_hats_midi(pm, file, output_directory):
    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each freqency category - Add list of tempos and change times
    midi_crash = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_crash._tempo_change_times = tempo_times
    midi_crash._tempos = tempi
    midi_crash.time_signature_changes.append(ts)

    midi_ride = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_ride._tempo_change_times = tempo_times
    midi_ride._tempos = tempi
    midi_ride.time_signature_changes.append(ts)

    midi_hh_closed = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_hh_closed._tempo_change_times = tempo_times
    midi_hh_closed._tempos = tempi
    midi_hh_closed.time_signature_changes.append(ts)

    midi_hh_open = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_hh_open._tempo_change_times = tempo_times
    midi_hh_open._tempos = tempi
    midi_hh_open.time_signature_changes.append(ts)

    # Create instruments for each frequency category
    crash_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Crash')
    ride_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Ride')
    hh_closed_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Closed Hat')
    hh_open_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Functional Open Hat')


    for note in pm.instruments[0].notes:
        if note.pitch in FUNCTIONAL_MAPPING_HATS["CRASH"]:
            crash_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING_HATS["RIDE"]:
            ride_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING_HATS["HH_CLOSED"]:
            hh_closed_instrument.notes.append(note)
        elif note.pitch in FUNCTIONAL_MAPPING_HATS["HH_OPEN"]:
            hh_open_instrument.notes.append(note)

    # Add instruments to the MIDI objects
    midi_crash.instruments.append(crash_instrument)
    midi_ride.instruments.append(ride_instrument)
    midi_hh_closed.instruments.append(hh_closed_instrument)
    midi_hh_open.instruments.append(hh_open_instrument)

    #check if one of the instruments is empty, if so, do not write file
    if (crash_instrument.notes and ride_instrument.notes and hh_open_instrument.notes and hh_closed_instrument.notes):
        # Save the MIDI files
        midi_crash.write(os.path.join(output_directory, file_name + '_functionalhats_crash.mid'))
        midi_ride.write(os.path.join(output_directory, file_name + '_functionalhats_ride.mid'))
        midi_hh_closed.write(os.path.join(output_directory, file_name + '_functionalhats_hhclosed.mid'))
        midi_hh_open.write(os.path.join(output_directory, file_name + '_functionalhats_hhopen.mid'))

def write_toms_and_rides_midi(pm, file, output_directory):
    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each freqency category - Add list of tempos and change times
    midi_tom_lo = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_tom_lo._tempo_change_times = tempo_times
    midi_tom_lo._tempos = tempi
    midi_tom_lo.time_signature_changes.append(ts)

    midi_tom_mid = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_tom_mid._tempo_change_times = tempo_times
    midi_tom_mid._tempos = tempi
    midi_tom_mid.time_signature_changes.append(ts)

    midi_tom_hi = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_tom_hi._tempo_change_times = tempo_times
    midi_tom_hi._tempos = tempi
    midi_tom_hi.time_signature_changes.append(ts)

    midi_ride = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    midi_ride._tempo_change_times = tempo_times
    midi_ride._tempos = tempi
    midi_ride.time_signature_changes.append(ts)

    # Create instruments for each frequency category
    tom_lo_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Tom Low')
    tom_mid_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Tom Mid')
    tom_hi_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Tom Hi')
    ride_instrument = pretty_midi.Instrument(program=10, is_drum=True, name='Tom Ride')
    
    for note in pm.instruments[0].notes:
        if note.pitch in TOMS_AND_RIDES["TOM_LO"]:
            tom_lo_instrument.notes.append(note)
        elif note.pitch in TOMS_AND_RIDES["TOM_MID"]:
            tom_mid_instrument.notes.append(note)
        elif note.pitch in TOMS_AND_RIDES["TOM_HI"]:
            tom_hi_instrument.notes.append(note)
        elif note.pitch in TOMS_AND_RIDES["RIDE"]:
            ride_instrument.notes.append(note)

    # Add instruments to the MIDI objects
    midi_tom_lo.instruments.append(tom_lo_instrument)
    midi_tom_mid.instruments.append(tom_mid_instrument)
    midi_tom_hi.instruments.append(tom_hi_instrument)
    midi_ride.instruments.append(ride_instrument)

    #check if one of the instruments is empty, if so, do not write file
    if (tom_lo_instrument.notes and tom_mid_instrument.notes and tom_hi_instrument.notes and ride_instrument.notes):
        # Save the MIDI files
        midi_tom_lo.write(os.path.join(output_directory, file_name + '_toms_lo.mid'))
        midi_tom_mid.write(os.path.join(output_directory, file_name + '_toms_mid.mid'))
        midi_tom_hi.write(os.path.join(output_directory, file_name + '_toms_hi.mid'))
        midi_ride.write(os.path.join(output_directory, file_name + '_toms_ride.mid'))

def process_file(args):
    root, file, input_dir, output_dir = args
    if not file.endswith('.mid'):
        return

    in_path = os.path.join(root, file)
    try:
        pm = pretty_midi.PrettyMIDI(in_path)
        print(f"[{os.getpid()}] Processing {file}")
        file_name = file.split('.mid')[0]
        file_dir = os.path.join(output_dir, file_name)

        midi_helper.remove_small_intervals(pm)
        lv, mv, hv = get_note_velocity(pm)

        os.makedirs(file_dir, exist_ok=True)

        write_velocity_midi(pm, lv, mv, hv, file, file_dir)
        write_lo_mid_hi_midi(pm, file, file_dir)
        write_functional_midi(pm, file, file_dir)
        write_functional_hats_midi(pm, file, file_dir)
        write_toms_and_rides_midi(pm, file, file_dir)
    except Exception as e:
        print(f"Error {file}: {e}")

def write_midi_files_mp(input_directory, output_directory, n_workers=None):
    os.makedirs(output_directory, exist_ok=True)

    tasks = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            tasks.append((root, file, input_directory, output_directory))

    with Pool(processes=n_workers) as pool:
        pool.map(process_file, tasks)
# def write_midi_files(input_directory, output_directory):
#     """
#     Reads MIDI files from the input directory, processes them, and writes the results to the output directory.
    
#     """
#     os.makedirs(output_directory, exist_ok=True)
#     for root, dirs, files in os.walk(input_directory):
#         for file in files[:5]:
#             try:
#                 if file.endswith('.mid'):
#                     # Load the MIDI file
#                     pm = pretty_midi.PrettyMIDI(os.path.join(root, file))
#                     print(f"Processing file: {file}")
#                     # Remove small intervals - These cause issues when rewriting to MIDI
#                     midi_helper.remove_small_intervals(pm)

#                     lv, mv, hv = get_note_velocity(pm)
              
#                     write_velocity_midi(pm, lv, mv, hv, file, output_directory)
#                     write_lo_mid_hi_midi(pm, file, output_directory)
#                     write_functional_midi(pm, file, output_directory)
#                     write_functional_hats_midi(pm, file, output_directory)
#                     write_toms_and_rides_midi(pm, file, output_directory)
                    
#             except:
#                 print(f"Error processing file: {file}. Skipping this file.")
#                 continue



# def create_hvo_hi_mid_low(directory, file):
#     hvo_seq_gm = hvo_sequence.midi_to_hvo_sequence(
#     filename=os.path.join(directory, file),
#     drum_mapping=HI_MID_LO_MAPPING,
#     beat_division_factors=[4])

#     return hvo_seq_gm

# def create_hvo_functional(directory, file):
#     hvo_seq_gm = hvo_sequence.midi_to_hvo_sequence(
#     filename=os.path.join(directory, file),
#     drum_mapping=FUNCTIONAL_MAPPING,
#     beat_division_factors=[4])

#     return hvo_seq_gm

# def create_hvo_functional_hats(directory, file):
#     hvo_seq_gm = hvo_sequence.midi_to_hvo_sequence(
#     filename=os.path.join(directory, file),
#     drum_mapping=FUNCTIONAL_MAPPING_HATS,
#     beat_division_factors=[4])

#     return hvo_seq_gm

# def create_hvo_ride_toms(directory, file):
#     hvo_seq_gm = hvo_sequence.midi_to_hvo_sequence(
#     filename=os.path.join(directory, file),
#     drum_mapping=TOMS_AND_RIDES,
#     beat_division_factors=[4])

#     return hvo_seq_gm


# def main(params):
#     NUM_PARAMS = 2
#     if len(params) != NUM_PARAMS:
#         print("Main function takes only 2 parameters: input directory and output directory")
#         sys.exit(1)

#     write_midi_files(params[0], params[1])
    

# if __name__ == "__main__":
#     main(sys.argv[1:])

def main(params):
    if len(params) not in (2, 3):
        print("Usage: script.py input_dir output_dir [n_workers]")
        sys.exit(1)
    input_dir, output_dir = params[0], params[1]
    n_workers = int(params[2]) if len(params) == 3 else None
    write_midi_files_mp(input_dir, output_dir, n_workers)

if __name__ == "__main__":
    main(sys.argv[1:])




































    
"""
# DENSITY_MAPPING separates the drums into density bins
DENSITY_MAPPING = {
    #Low is Hits below
    "LOW": 1/4,
    #MID is between LOW and HIGH
    "HIGH": 3/4
}

def get_note_density(pm):
    low_density = []
    mid_density = []
    high_density = []
    total_notes = len(pm.instruments[0].notes)
    #We need this to get a note count for each pitch
    PITCH_DICT = {
        22: [],
        26: [],
        35: [],
        36: [],
        37: [],
        38: [],
        39: [],
        40: [],
        41: [],
        42: [],
        43: [],
        44: [],
        45: [],
        46: [],
        47: [],
        48: [],
        49: [],
        50: [],
        51: [],
        52: [],
        53: [],
        55: [],
        57: [],
        58: [],
        59: []
    }

    
    for note in pm.instruments[0].notes:
        PITCH_DICT[note.pitch].append(note)
    for pitch in PITCH_DICT.keys():
        notes = PITCH_DICT[pitch]
        if (len(notes) / total_notes) < DENSITY_MAPPING["LOW"]:
            low_density.extend(notes)
        elif (len(notes) / total_notes) > DENSITY_MAPPING["HIGH"]:
            high_density.extend(notes)
        else:
            mid_density.extend(notes)

    return low_density, mid_density, high_density

def write_density_midi(pm, low_density, mid_density, high_density, file, output_directory):

    file_name = file.split('.mid')[0]
    #GET the tempo changes and time signature from the original MIDI file
    tempo_times, tempi = pm.get_tempo_changes()
    # Set time signature for the midi files below
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)

    #Create PrettyMIDI objects for each density category - Add list of tempos and change times
    midi_low = pretty_midi.PrettyMIDI()
    midi_low._tempo_change_times = tempo_times
    midi_low._tempos = tempi

    midi_mid = pretty_midi.PrettyMIDI()
    midi_mid._tempo_change_times = tempo_times
    midi_mid._tempos = tempi

    midi_high = pretty_midi.PrettyMIDI()
    midi_high._tempo_change_times = tempo_times
    midi_high._tempos = tempi

    # Create instruments for each density category
    low_instrument = pretty_midi.Instrument(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    low_instrument.time_signature_changes.append(ts)

    mid_instrument = pretty_midi.Instrument(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    mid_instrument.time_signature_changes.append(ts)
    
    high_instrument = pretty_midi.Instrument(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    high_instrument.time_signature_changes.append(ts)
    

    # Add notes to the respective instruments
    for note in low_density:
        low_instrument.notes.append(note)
    for note in mid_density:
        mid_instrument.notes.append(note)
    for note in high_density:
        high_instrument.notes.append(note)
    # Add instruments to the MIDI objects
    midi_low.instruments.append(low_instrument)
    midi_mid.instruments.append(mid_instrument)
    midi_high.instruments.append(high_instrument)
    # Save the MIDI files
    midi_low.write(os.path.join(output_directory, file_name + '_low_density'))
    midi_mid.write(os.path.join(output_directory, file_name + '_mid_density'))
    midi_high.write(os.path.join(output_directory, file_name + '_high_density'))
   
"""