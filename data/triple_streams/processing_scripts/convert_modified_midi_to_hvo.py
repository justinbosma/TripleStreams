""" from hvo_sequence import midi_to_hvo_sequence
from hvo_sequence.custom_dtypes import Metadata
import os

def convert_lakh_instrument_midi_to_hvo(midi_dir, output_dir):
    #Create Mapping for the MIDI values in the tracks. Since these are pitched, we make a mapping from [0, 127]
    mapping_array = list(range(0,128))
    PITCH_MAPPING = {
    'Instrument': mapping_array
    }
    os.makedirs('output_dir', exist_ok=True)
    for root, dirs, files in os.walk('lakh_midi/lmd_merged'):
        for file in files:
            if file.endswith('.mid'):

                #We will use these values for metatdata
                name_remove_mid = file.split('.')[0]

                file_main_name = name_remove_mid.split('_')[0]

                file_instrument = name_remove_mid.split('_')[1]

                file_dir = output_dir + '/' + file_main_name
                os.makedirs(file_dir, exist_ok=True)

                hvo_seq = midi_to_hvo_sequence(
                        filename= os.path.join(root, file),
                        drum_mapping=PITCH_MAPPING,
                        beat_division_factors=[4])

                metadata_first_bar = Metadata({
                    'filename': file,
                    'main_file': file_main_name,
                    'instrument': file_instrument,
                    'source': 'Lakh MIDI'})

                hvo_seq.metadata = metadata_first_bar
                hvo_seq.save(os.path.join(file_dir, name_remove_mid))
                

 """


import os
from hvo_sequence import midi_to_hvo_sequence
from hvo_sequence.custom_dtypes import Metadata
from multiprocessing import Pool, cpu_count

# Define mapping once
mapping_array = list(range(0, 128))
PITCH_MAPPING = {'Instrument': mapping_array}

def process_midi_file(args):
    file_path, output_dir = args
    try:
        file = os.path.basename(file_path)
        name_remove_mid = file.split('.')[0]
        file_main_name = name_remove_mid.split('_')[0]
        file_instrument = name_remove_mid.split('_')[1]
        file_dir = os.path.join(output_dir, file_main_name)
        os.makedirs(file_dir, exist_ok=True)

        hvo_seq = midi_to_hvo_sequence(
            filename=file_path,
            drum_mapping=PITCH_MAPPING,
            beat_division_factors=[4]
        )

        metadata = Metadata({
            'filename': file,
            'main_file': file_main_name,
            'instrument': file_instrument,
            'source': 'Lakh MIDI'
        })

        hvo_seq.metadata = metadata
        hvo_seq.save(os.path.join(file_dir, name_remove_mid))

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def get_all_midi_files(midi_dir):
    midi_files = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

def convert_lakh_instrument_midi_to_hvo(midi_dir, output_dir, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    midi_files = get_all_midi_files(midi_dir)

    # Prepare list of (file_path, output_dir) tuples
    task_args = [(file_path, output_dir) for file_path in midi_files]

    # Use specified or default number of workers
    num_workers = num_workers or cpu_count()

    print(f"Processing {len(midi_files)} MIDI files with {num_workers} workers...")

    with Pool(num_workers) as pool:
        pool.map(process_midi_file, task_args)

def process_groove_midi_file(args):
    file_path, output_dir = args
    try:
        file = os.path.basename(file_path)
        print(f"[{os.getpid()}] Processing {file}")
        name_remove_mid = file.split('.')[0]
        split_name = file.split('_')
        save_name = split_name[0] + '_' + split_name[1] + '_' + split_name[2] + '_' + split_name[3] + '_' + split_name[4] + '_' + split_name[5]
        file_dir = os.path.join(output_dir, save_name)
        os.makedirs(file_dir, exist_ok=True)

        hvo_seq = midi_to_hvo_sequence(
            filename=file_path,
            drum_mapping=PITCH_MAPPING,
            beat_division_factors=[4]
        )

        metadata = Metadata({
            'track_number': split_name[0],
            'genre': split_name[1],
            'bpm': split_name[2],
            'meter': split_name[4],
            'grouping': split_name[5],
            'member': split_name[6].split('.')[0],
            'source': 'GrooveMIDI'
        })

        hvo_seq.metadata = metadata
        hvo_seq.save(os.path.join(file_dir, name_remove_mid))

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


def convert_groove_midi_to_hvo(midi_dir, output_dir, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    midi_files = get_all_midi_files(midi_dir)

    # Prepare list of (file_path, output_dir) tuples
    task_args = [(file_path, output_dir) for file_path in midi_files]

    # Use specified or default number of workers
    num_workers = num_workers or cpu_count()

    print(f"Processing {len(midi_files)} MIDI files with {num_workers} workers...")

    with Pool(num_workers) as pool:
        pool.map(process_groove_midi_file, task_args)
def main():
    midi_dir = "/Users/justinbosma/Desktop/TripleStreams/groove_midi/groove_merged"
    output_dir = "/Users/justinbosma/Desktop/TripleStreams/groove_midi/groove_hvo"
    convert_groove_midi_to_hvo(midi_dir, output_dir, 8)

if __name__ == "__main__":
    main()
    