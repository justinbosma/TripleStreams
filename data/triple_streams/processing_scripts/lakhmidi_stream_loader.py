from random import random
import os
import tqdm
import itertools
import pickle
import bz2
import math
from hvo_sequence.io_helpers import load_HVO_Sequence_from_file
from hvo_sequence.hvo_seq import HVO_Sequence

grouping_types = ["AllAvailableStreams"]

def get_loaded_hvos_text_description(loaded_hvos, title="Lack MIDI"):
    text = f'{title}\n\n'
    text += "Dictionary Structure:\n"

    for grouping_type, hvo_dict in loaded_hvos.items():
        text += f"{grouping_type}:\n"
        cnt = 0
        for subfolder_name, subfolder_data in hvo_dict.items():
            text += (f"    |-------> {subfolder_name} ({len(subfolder_data['filenames'])} files) \n")
            text += (f"    |          |-------> {subfolder_name} \n")
            text += (f"    |                     |--> 'filenames:' \n")

            for i, filename in enumerate(subfolder_data["filenames"]):
                text += (f"    |                           |-------> {filename} \n")

            text += (f"    |                     |--> 'hvo_sequences:'\n")
            for i, hvo_seq in enumerate(subfolder_data["hvo_sequences"]):
                text += (
                    f"    |                           |-------> {hvo_seq.hvo.shape[0]} steps, i.e. {hvo_seq.hvo.shape[0] // 16} bars.\n")

            text += (f"    |                     |--> 'compiled_single_hvos:'\n")
            text += (
                f"    |                               |-------> {loaded_hvos[grouping_type][subfolder_name]['compiled_single_hvos']}\n")

            text += (f"    |                     |--> 'split_n_bar_phrases:'\n")
            text += (
                f"    |                               |-------> {loaded_hvos[grouping_type][subfolder_name]['split_n_bar_phrases']}\n")

            text += (f"    |-------> ...\n")
            text += ("\n")

            cnt += 1
            if cnt > 5:  # Limit to first 5 subfolders for readability
                break

    return text


def compile_four_hvo_streams_into_single_hvo(seperate_hvo_seqs_, filenames_):
    voice_mapping = {
        'stream_0': [36],
        'stream_1': [38],
        'stream_2': [42],
        'stream_3': [46]
    }

    time_sigs = seperate_hvo_seqs_[-1].time_signatures
    tempos = seperate_hvo_seqs_[-1].tempos

    # max length of all HVO sequences

    single_compiled_hvo_seq = HVO_Sequence(
        beat_division_factors=[4],
        drum_mapping=voice_mapping
    )

    for time_sig in time_sigs:
        single_compiled_hvo_seq.add_time_signature(time_step=time_sig.time_step, numerator=time_sig.numerator,
                                                   denominator=time_sig.denominator)
    for tempo in tempos:
        single_compiled_hvo_seq.add_tempo(time_step=tempo.time_step, qpm=tempo.qpm)

    max_length = max(hvo_seq.hvo.shape[0] for hvo_seq in seperate_hvo_seqs_)
    single_compiled_hvo_seq.adjust_length(max_length)

    metadata_ = {}
    metadata_.update(seperate_hvo_seqs_[-1].metadata)
    metadata_.update(
        {f'stream_{ix}': filenames_[ix].split("_")[-1].replace(".hvo", "") for ix in range(len(filenames_))})
    n_streams = single_compiled_hvo_seq.hvo.shape[-1] // 3
    for ix, hvo_stream_seq_ in enumerate(seperate_hvo_seqs_):
        hvo_stream_file_name = filenames_[ix]
        hvo_stream_seq_type = filenames_[ix].split("_")[-1].replace(".hvo", "")
        metadata_[f'stream_{ix}'] = hvo_stream_seq_type
        single_compiled_hvo_seq.hvo[:, ix] = hvo_stream_seq_.hvo[:, 0]
        single_compiled_hvo_seq.hvo[:, ix + n_streams] = hvo_stream_seq_.hvo[:, 1]
        single_compiled_hvo_seq.hvo[:, ix + n_streams * 2] = hvo_stream_seq_.hvo[:, 2]

    single_compiled_hvo_seq.metadata.update(metadata_)

    return single_compiled_hvo_seq


def get_combinations_of_four_hvo_streams(seperate_hvo_seqs_, filenames_):
    seperate_hvo_seqs_four_streams = []
    filenames_four_streams = []
    # get all combinations of four HVO sequences (without repetition)
    for combination in itertools.combinations(zip(seperate_hvo_seqs_, filenames_), 4):
        hvo_streams_combination, filenames_combination = zip(*combination)
        if len(hvo_streams_combination) == 4:
            seperate_hvo_seqs_four_streams.append(list(hvo_streams_combination))
            filenames_four_streams.append(list(filenames_combination))
    return seperate_hvo_seqs_four_streams, filenames_four_streams


def split_into_n_bar_phrases(hvo_seq, n_bars=2, hop_size=16):
    """
    Split the HVO sequence into n-bar phrases with a hop size.
    Returns a list of HVO sequences.
    """
    n_steps_per_bar = hvo_seq.grid_maker.n_steps_per_beat * 4  # Assuming 4/4 time signature
    n_steps_per_phrase = n_steps_per_bar * n_bars

    phrases = []

    for start in range(0, hvo_seq.hvo.shape[0] - n_steps_per_phrase + 1, hop_size):
        end = start + n_steps_per_phrase
        phrase_hvo = hvo_seq.copy_empty()
        phrase_hvo.adjust_length(n_steps_per_phrase)
        phrase_hvo.hvo[:n_steps_per_phrase, :] = hvo_seq.hvo[start:end, :]
        phrase_hvo.metadata.update({'start_bar': start // n_steps_per_bar})
        phrases.append(phrase_hvo)

    return phrases


def process_batch(batch_subfolders, batch_num, root_dir):
    """Process a batch of subfolders and save to a unique file"""
    print(f"\n=== Processing Batch {batch_num + 1}/60 ===")
    print(f"Processing {len(batch_subfolders)} subfolders in this batch")

    # Initialize structure for this batch
    loaded_hvos = {grouping_type: {} for grouping_type in grouping_types}

    # Create subfolder dictionary for this batch
    subfolder_dict = {grouping_type: batch_subfolders for grouping_type in grouping_types}

    # Initialize the loaded_hvos structure for this batch
    for grouping_type, subfolders in subfolder_dict.items():
        for subfolder in subfolders:
            subfolder_name = subfolder.replace(f"_{grouping_type}", "")
            loaded_hvos[grouping_type].update({subfolder_name: {
                "filenames": [],
                "hvo_sequences": [],
                "compiled_single_hvos": [],
                "split_n_bar_phrases": None
            }})

    # Load HVO sequences for this batch
    for grouping_type, subfolders in subfolder_dict.items():
        print(f"Loading {grouping_type} for batch {batch_num + 1}")
        for subfolder in tqdm.tqdm(subfolders, desc=f"Batch {batch_num + 1} - Loading"):
            subfolder_name = subfolder.replace(f"_{grouping_type}", "")
            hvo_files = [f for f in os.listdir(os.path.join(root_dir, subfolder)) if f.endswith('.hvo')]

            for hvo_file in hvo_files:
                hvo_seq = load_HVO_Sequence_from_file(os.path.join(root_dir, subfolder, hvo_file))
                loaded_hvos[grouping_type][subfolder_name]["filenames"].append(hvo_file)
                loaded_hvos[grouping_type][subfolder_name]["hvo_sequences"].append(hvo_seq)

    # Adjust all loaded HVO sequences to the same length
    for grouping_type, hvo_dict in loaded_hvos.items():
        for subfolder_name, subfolder_data in hvo_dict.items():
            if subfolder_data["hvo_sequences"]:  # Check if there are sequences
                max_length = max(hvo_seq.hvo.shape[0] for hvo_seq in subfolder_data["hvo_sequences"])
                for i, hvo_seq in enumerate(subfolder_data["hvo_sequences"]):
                    hvo_seq.adjust_length(max(max_length, 32))

    # Compile HVO streams for this batch
    for grouping_type, hvo_dict in loaded_hvos.items():
        for subfolder_name, subfolder_data in tqdm.tqdm(hvo_dict.items(), desc=f"Batch {batch_num + 1} - Compiling"):
            filenames = subfolder_data["filenames"]
            seperate_hvo_seqs = subfolder_data["hvo_sequences"]
            if len(seperate_hvo_seqs) >= 4:  # Need at least 4 sequences for combinations
                grouped_hvos, grouped_filenames = get_combinations_of_four_hvo_streams(seperate_hvo_seqs, filenames)
                loaded_hvos[grouping_type][subfolder_name]["compiled_single_hvos"] = []
                for i in range(len(grouped_hvos)):
                    seperate_hvo_seqs_ = grouped_hvos[i]
                    filenames_ = grouped_filenames[i]
                    compiled_single_hvo = compile_four_hvo_streams_into_single_hvo(seperate_hvo_seqs_, filenames_)
                    loaded_hvos[grouping_type][subfolder_name]["compiled_single_hvos"].append(compiled_single_hvo)

    # Split into n-bar phrases for this batch
    for grouping_type, hvo_dict in loaded_hvos.items():
        for subfolder_name, subfolder_data in tqdm.tqdm(hvo_dict.items(), desc=f"Batch {batch_num + 1} - Splitting"):
            loaded_hvos[grouping_type][subfolder_name]["split_n_bar_phrases"] = []
            compiled_single_hvos = subfolder_data["compiled_single_hvos"]
            for compiled_single_hvo in compiled_single_hvos:
                split_n_bar_phrases = split_into_n_bar_phrases(compiled_single_hvo, n_bars=2, hop_size=16)
                loaded_hvos[grouping_type][subfolder_name]["split_n_bar_phrases"].extend(split_n_bar_phrases)

    # Save this batch to a unique compressed file
    os.makedirs("data/triple_streams/split_2bars", exist_ok=True)
    batch_filename = f"data/triple_streams/split_2bars/lmd_batch_{batch_num + 1:02d}.pkl.bz2"
    with bz2.BZ2File(batch_filename, "wb") as f:
        pickle.dump(loaded_hvos, f)

    # Save the description text for this batch
    description_filename = f"data/triple_streams/split_2bars/lmd_batch_{batch_num + 1:02d}_description.txt"
    with open(description_filename, "w") as f:
        f.write(get_loaded_hvos_text_description(loaded_hvos, title=f"Batch {batch_num + 1} - Lack MIDI"))

    print(f"Batch {batch_num + 1} saved to {batch_filename}")
    print(f"Batch {batch_num + 1} description saved to {description_filename}")

    return loaded_hvos


# Main execution
if __name__ == "__main__":
    root_dir = "data/triple_streams/lmd_merged_hvo"

    # Get all subfolders
    all_subfolders = [f for f in os.listdir(root_dir) if "_" not in f]
    print(f"Total subfolders found: {len(all_subfolders)}")

    # Divide into 60 equal batches
    batch_size = math.ceil(len(all_subfolders) / 60)
    batches = [all_subfolders[i:i + batch_size] for i in range(0, len(all_subfolders), batch_size)]

    print(f"Dividing into {len(batches)} batches with batch size: {batch_size}")
    for i, batch in enumerate(batches):
        print(f"Batch {i + 1}: {len(batch)} subfolders")

    # Process each batch
    for batch_num, batch_subfolders in enumerate(batches):
        try:
            process_batch(batch_subfolders, batch_num, root_dir)
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {e}")
            continue

    print("\nAll batches processed successfully!")
    print("Output files:")
    for i in range(len(batches)):
        print(f"  - lmd_batch_{i + 1:02d}.pkl.bz2")
        print(f"  - lmd_batch_{i + 1:02d}_description.txt")