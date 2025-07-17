import os
root_dir = "data/triple_streams/bongosero_hvo_sequences_single_voices"

# find all folders in the root directory
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# each folder there are files like 10834_attempt_num_0_*.hvo
# in some cases there are more than one attempts.
# get the first folder which has more than one attempt
# number of attemps is number of files divided by 8
# dont rename yet
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.hvo')]
    num_attempts = len(files) // 8
    if num_attempts > 1:
        print(f"Folder: {folder}, Number of attempts: {num_attempts}")



new_dir = "data/triple_streams/bongosero_hvo_sequences_single_voices_organized_by_attempts"
# create new directory if it does not exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.hvo')]
    num_attempts = len(files) // 8
    for attempt in range(num_attempts):
        new_folder_name = f"{folder}_attempt{attempt}"
        os.makedirs(os.path.join(new_dir, new_folder_name), exist_ok=True)


for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.hvo')]
    num_attempts = len(files) // 8
    for attempt in range(num_attempts):
        # find all files that have _attempt_num_{attempt}_ in their name
        attempt_files = []
        for f in files:
            # only keep those that have both.hvo, left.hvo, and right.hvo, flattened.hvo
            if f"attempt_num_{attempt}_" in f:
                if any(x in f for x in ['both.hvo', 'left.hvo', 'right.hvo', 'flattened.hvo']):
                    # if the file has all four, add it to the list
                    attempt_files.append(f)
        print(attempt_files)

        new_filenames = [f.replace(f"{folder}_attempt_num_{attempt}_", "") for f in attempt_files]

        # copy files to new directory with new names
        for old_name, new_name in zip(attempt_files, new_filenames):
            old_path = os.path.join(folder_path, old_name)
            new_path = os.path.join(new_dir, f"{folder}_attempt{attempt}", new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

