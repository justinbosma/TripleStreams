# TripleStreams
## TripleStreams is a real-time rhythm accompaniment system based on the GrooveTransformer system.
## It takes as an input a user's MIDI pattern and outputs three related MIDI rhythmic patterns

# Model Training
## The data used to train the model is in HVO representation and can be found in the directory "split_2bars"
## To create your own dataset, we've included the links for the original datasets and scripts for data preprocessing

# LAKH Dataset
## The LAKH dataset can be downloaded from here: http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
## The dataset is quite large. You can modify the method create_merged_midi in lakh_midi.py by adding a limit to the files like this "for file in files[0:30]:"
## How it works:
## The file lakh_midi.py is used to go through each midi file in the LAKH dataset, merge the instruments into their 'families', and split out each instrument type into a separate midi file.
## An example of the merging is combining piano, electric piano, grand piano, etc into an instrument labled "Piano". This allows us to focus on the role of pianos in the tracks
## The script filters out midi files that are not in 4/4 timing and files where the velocity does not change at least 7 times
## To run the script in the terminal type "python lakh_midi.py full_lakh_dir merged_dir" where lakh_dir is the directory of the LAKH dataset and merged_dir is the directory to create the new midi files in.

# GrooveMIDI
## GrooveMIDI is a dataset consisting of 
## The dataset can be obtained from this link: https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
## To run the script in the terminal run 

# Candombe
## The candombe annotations are provided in the repository in the csv files in the candombe_annotations directory. There are twelve csv files in total, each representing a recorded session. For creating the HVO sequences, we use the columns with labels "_hit, _vel, and _off" which give the hit, velocity, and offset time for each drum. C is the chico, P is piano, and R1 is the first repique. The groove is created by flattenning all the drums into a single stream. If any of the drums have a hit, we add a hit to the groove. The groove velocity is the maximum velocity from the three drums. The offset is randomly selected from the three drums.
## To run the preprocessing type "python candombe.py annotations_dir output_dir" where annotation_dir is the location of the csv files and output_dir is the desired output directory for the HVO sequences.
## More information about the recordings, and Candombe in general, can be found at https://candombeando.uy/

# El Bongosero
## The dataset can be downloaded here: https://elbongosero.github.io/
## T

# TapTamDrum
## The TapTamDrum dataset can be downloaded here: https://taptamdrum.github.io/

# hvo_sequence
A Piano Roll Representation for Drums Similar to Sequence Representations in Magenta's GrooVAE

w