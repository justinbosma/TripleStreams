import math
from multiprocessing import Pool
import os
import pickle, bz2
from tqdm import tqdm
import sys



def load_dataset(file_path):
    with bz2.BZ2File(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# Sorts the list of hvo sequences and finds the one with the lowest amount of hits, the highest amount of hits, 1/4 the max of hits, and 3/4 max of hits
# Input: dictionary from LMD track
# Output: new dictionary of same structure, but with split_n_bar_phrases updated with the best four hvo sequences
def filter_best_four(track_dict):
    new_dict = {}
    sorting_list = []
    splits = track_dict['split_n_bar_phrases']
    for i, hvo in enumerate(splits):
        if sum(sum(hvo.hits)) > 0:
            sum_with_index = [i, sum(sum(hvo.hits))]
            sorting_list.append(sum_with_index)
        #else:
            #print("there were no hits")
    sorted_hvo = sorted(sorting_list, key=lambda x: x[1])

    left_fourth = int(math.floor(len(sorted_hvo)/4))
    right_fourth = int(math.ceil(len(sorted_hvo)*(3/4)))
    if len(sorted_hvo) > 0:
        new_dict['split_n_bar_phrases'] = [splits[sorted_hvo[0][0]], splits[sorted_hvo[left_fourth][0]], splits[sorted_hvo[right_fourth][0]], splits[sorted_hvo[len(sorted_hvo) - 1][0]]]
        new_dict['filenames'] = track_dict['filenames']
        new_dict['hvo_sequences'] = track_dict['hvo_sequences']
        new_dict['compiled_single_hvos'] = track_dict['compiled_single_hvos']

    return new_dict

# Goes through each track in the dictionary and runs the filter_best_four function on each
# Input: Dictionary (first level) with track name values
# output: New Dictionary with new dictionaries for each track with updated best hvo values
def get_hvo_sequence_from_track_dictionary(dict):
    new_dict = {}
    for track in dict['AllAvailableStreams']:
        print(track)
        track_dict = dict['AllAvailableStreams'][track]
        new_dict[track] = filter_best_four(track_dict)
    return new_dict
    

def run_all(pkl_path):
 

    print("Processing:", pkl_path)
    try:
        dataset = load_dataset(pkl_path)
        filtered_dict = get_hvo_sequence_from_track_dictionary(dataset)

        # Build output path with `_filtered` suffix
        os.mkdir('data/triple_streams/split_2bars/lmd_top_four')
        base_name = os.path.basename(pkl_path).replace('.pkl.bz2', '')
        output_name = f"{base_name}_filtered.pkl.bz2"
        output_path = os.path.join('data/triple_streams/split_2bars/lmd_top_four', output_name)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the filtered dictionary
        with bz2.BZ2File(output_path, 'wb') as f:
            pickle.dump({'AllAvailableStreams': filtered_dict}, f)

        print(f"Saved filtered file: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return False


def main(dataset_dir):
    pkl_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith('.pkl.bz2')
    ]

    with Pool(processes=5) as pool:
        list(tqdm(pool.imap(run_all, pkl_files), total=len(pkl_files)))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Please only pass the data directory")
        sys.exit(1)
    main(sys.argv[1])