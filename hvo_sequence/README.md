# LAKH Dataset
## The file lakh_midi.py is used to go through each midi file in the LAKH dataset, merge the instruments into their 'families', and split out each instrument type into a separate midi file.
# Say something about the families we merge to.....!!!
## The script filters out midi files that are not in 4/4 timing and files where the velocity does not change at least 7 times
## To run the script in the terminal type "python lakh_midi.py full_lakh_dir merged_dir" where lakh_dir is the directory of the LAKH dataset and merged_dir is the directory to create the new midi files in


# hvo_sequence
A Piano Roll Representation for Drums Similar to Sequence Representations in Magenta's GrooVAE
