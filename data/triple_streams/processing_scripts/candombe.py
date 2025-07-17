import sys
from hvo_sequence import HVO_Sequence
import csv
import os
import numpy as np
import pandas as pd
import random
from hvo_sequence.custom_dtypes import Metadata

Four_Voices = {
        "voice_1": [36],
        "voice_2": [37],
        "voice_3": [38],
        "voice_4": [39]
    }





def create_hvo_from_annotation(directory, file):
    
    df = pd.read_csv(os.path.join(directory, file))
    # create an instance of HVO_Sequence
   
    hvo_seq = HVO_Sequence(
        beat_division_factors=[4],                 # select the subdivision of the beat
        drum_mapping=Four_Voices
    )
   
    #Create metadata for the HVO sequence. We will use file name, date of recording, and dataset
    metadata_first_bar = Metadata({
    'filename': file,
    'date_of_recording': 'Jul 2025',
    'source': 'Dataset Candombe with offsets'})
    
    hvo_seq.metadata = metadata_first_bar

    # Add time_signature
    hvo_seq.add_time_signature(time_step=0, numerator=4, denominator=4)
    # Add tempo
    hvo_seq.add_tempo(time_step=0, qpm=120)
    #Make sure length is divisble by 32
    hvo_length = int(len(df["R1_hit"]) - len(df["R1_hit"])%32)
    #Create empty hvo seq
    hvo_seq.zeros(hvo_length)

    for i in range(hvo_length):
        if df['C_hit'][i] > 0:
            hvo_seq.hits[i][0] = 1
            hvo_seq.velocities[i][0] = df['C_vel'][i]
            hvo_seq.offsets[i][0] = df['C_off'][i]
        if df['P_hit'][i] > 0:
            hvo_seq.hits[i][1] = 1
            hvo_seq.velocities[i][1] = df['P_vel'][i]
            hvo_seq.offsets[i][1] = df['P_off'][i]
        if df['R1_hit'][i] > 0:
            hvo_seq.hits[i][2] = 1
            hvo_seq.velocities[i][2] = df['R1_vel'][i]
            hvo_seq.offsets[i][2] = df['R1_off'][i]
        if(df['C_hit'][i] or df['P_hit'][i] or df['R1_hit'][i]) > 0:
            hvo_seq.hits[i][3] = 1
            hvo_seq.velocities[i][3] = max(df['C_vel'][i], df['P_vel'][i], df['R1_vel'][i])
            hvo_seq.offsets[i][3] = random.choice([df['C_off'][i], df['P_off'][i], df['R1_off'][i]])

    return hvo_seq

def create_and_save_all_hvo(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    hvo_list = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            try:
                file_name = file.split('.')[0]
               
                hvo_seq = create_hvo_from_annotation(input_directory, file)
              
                hvo_seq.save(output_directory + '/' + file_name)
          
                #hvo_list.append(hvo_seq)
            except:
                print(f"Error: Cannot process {file}")
                pass

def main(params):
    NUM_PARAMS = 2
    if len(params) != NUM_PARAMS:
        print("Main function takes only 2 parameters: input directory and output directory")
        sys.exit(1)
    create_and_save_all_hvo(params[0], params[1])

if __name__ == "__main__":
    main(sys.argv[1:])
