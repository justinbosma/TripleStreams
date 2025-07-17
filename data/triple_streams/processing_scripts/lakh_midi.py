import os
import pretty_midi
from tqdm import tqdm
import sys
#from hvo_sequence import HVO_Sequence
import matplotlib.pyplot as plt
import gc
import multiprocessing


midi_instrument_mapping = {
    "Piano": [0, 1, 2, 3, 4, 5, 6, 7],
    "Percussion": [8, 9, 10, 11, 12, 13, 14, 15],
    "Organ": [16, 17, 18, 19, 20, 21, 22, 23],
    "Guitar": [24, 25, 26, 27, 28, 29, 30, 31],
    "Bass": [32, 33, 34, 35, 36, 37, 38, 39],
    "Strings": [40, 41, 42, 43, 44, 45, 46, 47],
    "Ensemble": [48, 49, 50, 51, 52, 53, 54, 55],
    "Brass": [56, 57, 58, 59, 60, 61, 62, 63],
    "Reed": [64, 65, 66, 67, 68, 69, 70, 71],
    "Pipe": [72, 73, 74, 75, 76, 77, 78, 79],
    "Synth Lead": [80, 81, 82, 83, 84, 85, 86, 87],
    "Synth Pad": [88, 89, 90, 91, 92, 93, 94, 95],
    "Synth Effects": [96, 97, 98, 99, 100, 101, 102, 103],
    "Ethnic": [104, 105, 106, 107, 108, 109, 110, 111],
    "Percussive": [112, 113, 114, 115, 116, 117, 118, 119],
    "Sound Effects": [120, 121, 122, 123, 124, 125, 126, 127]
}
#Used in filtering out midi files that do not have the correct instruments
midi_instrument_mapping_for_writing = {
    "Piano": [0, 1, 2, 3, 4, 5, 6, 7],
    "Percussion": [8, 9, 10, 11, 12, 13, 14, 15],
    #"Organ": [16, 17, 18, 19, 20, 21, 22, 23],
    "Guitar": [24, 25, 26, 27, 28, 29, 30, 31],
    "Bass": [32, 33, 34, 35, 36, 37, 38, 39],
    #"Strings": [40, 41, 42, 43, 44, 45, 46, 47],
    #"Ensemble": [48, 49, 50, 51, 52, 53, 54, 55],
    "Brass": [56, 57, 58, 59, 60, 61, 62, 63],
    #"Reed": [64, 65, 66, 67, 68, 69, 70, 71],
    #"Pipe": [72, 73, 74, 75, 76, 77, 78, 79],
    #"Synth Lead": [80, 81, 82, 83, 84, 85, 86, 87],
    #"Synth Pad": [88, 89, 90, 91, 92, 93, 94, 95],
    #"Synth Effects": [96, 97, 98, 99, 100, 101, 102, 103],
    #"Ethnic": [104, 105, 106, 107, 108, 109, 110, 111],
    "Percussive": [112, 113, 114, 115, 116, 117, 118, 119]
    #"Sound Effects": [120, 121, 122, 123, 124, 125, 126, 127]
}


#Creates a new midi file with the midi instruments combined into their parent. e.g. "grand piano", "electric piano", etc are comined into "Piano"
#Inputs:
#Instruments: List of four instrument names taken from the midi_instrument_mapping
def write_new_midi_from_instruments(instruments, output_file, dir, original_midi):
    print("writing new midi file")
    tempo_times, tempi = original_midi.get_tempo_changes()

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)
    midi.time_signature_changes.append(ts)
    midi._tempo_change_times = tempo_times
    midi._tempos = tempi
    
    for instrument in instruments:
        midi.instruments.append(instrument)
    midi.write(dir + '/' + output_file)

#Takes a list of instruments of the same family and merges into one instrument
#Inputs:
#instruments: list of pretty midi instrument objects
# key: name of instrument family (key in the dictionary)
def merge_instruments(instruments, key):
    merged_instrument = pretty_midi.Instrument(
        program=instruments[0].program,
        is_drum=instruments[0].is_drum,
        name=key
    )
    for instr in instruments:
        merged_instrument.notes.extend(instr.notes)
    merged_instrument.notes.sort(key=lambda note: note.start)
    return merged_instrument

#Goes through midi file and returns a list of instruments that match the instrument_list
#Input: MIDI file and list of program numbers (e.g., [0, 1, 2, 3, 4, 5, 6, 7] for piano)
def get_instruments(midi, instrument_list):
    instruments_to_combine = []
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Exclude drum instruments
            if instrument.program in instrument_list:
                instruments_to_combine.append(instrument)
    return instruments_to_combine

# NEW!!! Checks if there are more than 7 different velocities in the file
# OLD !!! Checks if velocity is all the same - True == All the same, False == Not all the same
#use set to compare velocities
def correct_velocity(midi):
    note_set = set()
    for notes in midi.instruments[0].notes:
        note_set.add(notes.velocity)
    #print(f"Number of different velocities: {len(note_set)}")
    return (len(note_set) > 7)

#checks if the meter is in 4/4 timing
def correct_meter(midi):  
    four_four = False
    for time_sig_change in midi.time_signature_changes:
        if time_sig_change.numerator == 4 and time_sig_change.denominator == 4:
            four_four = True
        else:
            four_four = False
            break
    #print(f"Four four time signature: {four_four}")
    return four_four

def get_family_set(pm):
    """Returns a set of instrument families present in the PrettyMIDI object."""
    families = set()
    for inst in pm.instruments:
        if inst.is_drum:
            families.add("Drums")
        else:
            for key in midi_instrument_mapping_for_writing.keys():
                if inst.program in midi_instrument_mapping_for_writing[key]:
                    families.add(key)
    return families



# Go though each file
# Create pm object out of midi file
# merge the instruments
# Create a new pm object with the merged instruments
# write the file
# def create_merged_midi(directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     #file_count = sum(len(files) for _, _, files in os.walk(directory))  # Get the number of files
#     #with tqdm(total=file_count) as pbar:
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             try:
#                 if file.endswith('.mid'):
#                     print(f"Processing file {file}")
#                     pm = pretty_midi.PrettyMIDI(os.path.join(root, file))
                    
#                     #Verify there are at least four instruments we will use in the file
#                     # Check instrument families
#                     families = get_family_set(pm)
#                     if len(families) < 4:
#                         del pm
#                         continue

#                     if (correct_meter(pm) and correct_velocity(pm)):
#                         instruments_to_merge = []
#                         merged_instruments = []
#                         for key in midi_instrument_mapping.keys():
#                             instruments_to_merge = get_instruments(pm, midi_instrument_mapping[key])
                            
#                             if len(instruments_to_merge) > 0:
#                                 merged_instruments.append(merge_instruments(instruments_to_merge, key))

#                             #Drums are handled differently in midi, so we don't use the dictionary above
#                         drum_instrument_to_merge = pretty_midi.Instrument(
#                             program=0,
#                             is_drum=True,
#                             name="Drums"
#                         )
#                         #If the instrument is a drum, we add the notes to the new drum instrument
#                         #After adding all the notes, we can append to the merged instrument list
#                         #This handles cases where the midi file has multiple drum instruments like "kick","snare", etc..
#                         for instrument in pm.instruments:
#                             if instrument.is_drum:
#                                 drum_instrument_to_merge.notes.extend(instrument.notes)
                            
#                         if len(drum_instrument_to_merge.notes) > 0:
#                             merged_instruments.append(drum_instrument_to_merge)
                        
#                         write_new_midi_from_instruments(merged_instruments, file, output_directory, pm)
#                         print(f"Success {file}")
#                     else:
#                         pass
#                     ## Added this July 10 2025 - 23:44
#                     ##. Getting memory issues
#                     del pm
#                     del instruments_to_merge
#                     del merged_instruments
#                     gc.collect()
#             except:
#                 print(f"Error processing file {file}")
#                 continue

def create_merged_midi(directory, output_directory, batch_size=100):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    midi_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))

    total_files = len(midi_files)
    print(f"Found {total_files} MIDI files to process...")

    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch = midi_files[batch_start:batch_end]
        print(f"\nProcessing batch {batch_start}–{batch_end - 1}...")

        for file_path in tqdm(batch):
            try:
                file = os.path.basename(file_path)
                print(f"Processing file {file}")
                pm = pretty_midi.PrettyMIDI(file_path)
                
                # Filter: Must have 4+ usable instrument families
                families = get_family_set(pm)
                if len(families) < 4:
                    del pm
                    continue

                if not (correct_meter(pm) and correct_velocity(pm)):
                    del pm
                    continue

                # Merge instruments by family
                merged_instruments = []
                for key in midi_instrument_mapping.keys():
                    instrs = get_instruments(pm, midi_instrument_mapping[key])
                    if instrs:
                        merged_instruments.append(merge_instruments(instrs, key))

                # Merge drum instruments manually
                drum_instrument = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
                for inst in pm.instruments:
                    if inst.is_drum:
                        drum_instrument.notes.extend(inst.notes)
                if drum_instrument.notes:
                    merged_instruments.append(drum_instrument)

                # Save merged MIDI file
                write_new_midi_from_instruments(merged_instruments, file, output_directory, pm)

                # Memory cleanup
                del pm
                del merged_instruments
                gc.collect()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

#This removes small intervals from the midi file, which can cause issues rewriting the midi file
def remove_small_intervals(midi):
    min_interval = 1
    intervals = []
    epsilon = 1e-3
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end - note.start <= epsilon:
                print("error")
                instrument.notes.remove(note)
            else:
            # print("ok")
                #intervals.append(note.end - note.start)
                #min_interval = min(min_interval, note.end - note.start)
                pass

# Creates separate midi files for each instrument in the midi file
# The instrument names are hardcoded, so it only works for the Lakh MIDI dataset
# Inputs: midi_dir (directory containing the merged midi files),
#         midi_file (name of the midi file),
#         new_dir (directory to save the new midi files)
# Outputs: new midi files with the instrument names appended to the file name
# Example: if the input midi file is 'example.mid', the output files will be:
#          'example_piano.mid', 'example_guitar.mid', 'example_bass.mid', etc.
def create_separate_midi_tracks_lakh(midi_dir):

    

    #file_count = sum(len(files) for _, _, files in os.walk(midi_dir))  # Get the number of files

    #with tqdm(total=file_count) as pbar:
    for root, dirs, files in os.walk(midi_dir):
        for midi_file in files:
            if midi_file.endswith('.mid'):
                try:
                    #Create Pretty Midi object
                    pm = pretty_midi.PrettyMIDI(midi_dir + '/' + midi_file)
                    
                    #Get tempo changes to add to new midi files
                    tempo_times, tempi = pm.get_tempo_changes()

                    #Get midi file name to create new directory for the individual instrument files - this helps with creating the hvo sequences
                    midi_file_name = midi_file.split('.mid')[0]
                    os.makedirs(midi_dir + "/" + midi_file_name)

                    # Remove the small intervals from the midi file - these were causing errors
                    remove_small_intervals(pm)

                    # Set time signature for the midi files below
                    ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)
                    
                    
                    for instrument in pm.instruments:

                        if instrument.name == 'Piano':
                            instrument_piano = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_piano.time_signature_changes.append(ts)
                            instrument_piano.instruments.append(instrument)
                            instrument_piano.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_piano.mid')
                            print(f"Writing {midi_file_name}_piano.mid")
                    
                        if instrument.name == 'Guitar':
                            instrument_guitar = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_guitar.time_signature_changes.append(ts)
                            instrument_guitar.instruments.append(instrument)
                            instrument_guitar.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_guitar.mid')
                            print(f"Writing {midi_file_name}_guitar.mid")

                        if instrument.name == 'Bass':
                            instrument_bass = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_bass.time_signature_changes.append(ts)
                            instrument_bass.instruments.append(instrument)
                            instrument_bass.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_bass.mid')
                            print(f"Writing {midi_file_name}_bass.mid")

                        if instrument.name == 'Drums':
                            instrument_drum = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_drum.time_signature_changes.append(ts)
                            instrument_drum.instruments.append(instrument)
                            instrument_drum.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_drum.mid')
                            print(f"Writing {midi_file_name}_drum.mid")
                            

                        if instrument.name == 'Brass':
                            instrument_brass = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_brass.time_signature_changes.append(ts)
                            instrument_brass.instruments.append(instrument)
                            instrument_brass.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_brass.mid')
                            print(f"Writing {midi_file_name}_brass.mid")

                        if instrument.name == 'Percussion':
                            instrument_sfx = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_sfx.time_signature_changes.append(ts)
                            instrument_sfx.instruments.append(instrument)
                            instrument_sfx.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_percussion.mid')
                            print(f"Writing {midi_file_name}_percussion.mid")

                        if instrument.name == 'Percussive':
                            instrument_sfx = pretty_midi.PrettyMIDI(initial_tempo=tempi[0] if len(tempi) > 0 else 120.0)
                            instrument_sfx.time_signature_changes.append(ts)
                            instrument_sfx.instruments.append(instrument)
                            instrument_sfx.write(midi_dir + "/" + midi_file_name + '/' + midi_file_name + '_percussive.mid')
                            print(f"Writing {midi_file_name}_percussive.mid")

                except:
                    print(f"Error processing file {midi_file}")
                    continue


# 1. Go through list of instruments and find associated midi file - need file name and instrument
# 2. Create hvo sequence for each midi file
# 3. combine hvo sequences into one
# 4. save the sequence

# def create_hvo_sequence_lakh(instrument_list, midi_path, midi_file):

#     #Create Mapping for the MIDI values in the tracks. Since these are pitched, we make a mapping from [0, 127]
#     mapping_array = list(range(0,128))
#     PITCH_MAPPING = {
#         'Instrument': mapping_array
#     }

#     hs_individual_streams_list = []

#     max_length = 0
#     #Load the four midi instrument files into HVO Sequences
#     for instrument in instrument_list:
#         hs_individual_streams_list.append(
#             HVO_Sequence.midi_to_hvo_sequence(
#                 filename= midi_path + '/' + midi_file + '_' + instrument + '.mid',
#                 drum_mapping=PITCH_MAPPING,
#                 beat_division_factors=[4]))
        
#     #Find the longest HVO Sequence
#     for i in range(4):
#         max_length = max(max_length, hs_individual_streams_list[i].hvo.shape[0])

#     #Adjust all HVO Sequences so they have the same length as the longest
#     for i in range(4):
#         hs_individual_streams_list[i].adjust_length(max_length)

# pass





    # def plot_instrument_histogram(directory):
    #     instrument_types_for_hist = []
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             pm_hist = pretty_midi.PrettyMIDI(os.path.join(root, file))
    #             for instrument in pm_hist.instruments:
    #                 instrument_types_for_hist.append(instrument.name)
    #     plt.hist(instrument_types_for_hist, edgecolor='black')
    #     plt.xticks(rotation=90)
    #     plt.show()

def main(params):
    NUM_PARAMS = 2
    if len(params) != NUM_PARAMS:
        print("Main function takes only 2 parameters: input directory and output directory")
        sys.exit(1)
    instrument_list = [['bass', 'drums', 'guitar', 'piano'], ['bass', 'brass', 'drums', 'guitar'], ['bass', 'brass', 'drums', 'piano'], ['brass', 'drums', 'guitar', 'piano'], ['bass', 'brass', 'guitar', 'piano']]
    create_merged_midi(params[0], params[1])
    create_separate_midi_tracks_lakh(params[1])

if __name__ == "__main__":
    main(sys.argv[1:])
