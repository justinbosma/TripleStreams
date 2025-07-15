import os
import requests
import zipfile
import sys
from hvo_sequence import HVO_Sequence
from hvo_sequence.custom_dtypes import Metadata
from dataset_and_API.src import BongoDrumCollection



## Make code that takes in a list of lists
## allows user to select four pre-determined combinations - e.g. left, right, both, kick



def make_hvo_from_list(instrument_list, attempt, dataset_dir, output_dir, attempt_num, user):
    #Check if output directory exists and create it if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #create the subdirectory if it doesn't exist
    sub_dir = output_dir + '/' + str(user.user_id)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    print(attempt_num)
    
    #location variable to keep track of which of the 4 streams we are creating
    location = 0

    for name in instrument_list:
        #for instrument_list in instrument_lists:
        # print('user_id_' + str(user.user_id))
        # print('attempt_num_' + str(attempt_num))
        # print(name)
        # print(attempt.user_level_of_musical_experience)
        # print(attempt.user_exhibion_rating)
        #We set each voice to a Midi Mapping value for Drums
        FOUR_VOICES = {
            "voice_1": [36],
            "voice_2": [37],
            "voice_3": [38],
            "voice_4": [39]
        }

        #Create new HVO Sequence with beat division of 4 and set Drum Mapping to Four_Voices Mapping
        hvo_seq_four_voices = HVO_Sequence(
            beat_division_factors=[4],
            drum_mapping=FOUR_VOICES)
        
        
        #set tempo
        hvo_seq_four_voices.add_tempo(
            time_step=0,
            qpm=attempt.attempt_tempo
        )

        #Add a time signature of 4/4 for the HVO Sequence
        hvo_seq_four_voices.add_time_signature(
            time_step=0,
            numerator=4,
            denominator=4
        )

        #Create metadata for hvo
        metadata_first_bar = Metadata({
            'self_assesment': attempt.self_assessment,
            'genre': attempt.genre,
            'tempo': attempt.attempt_tempo,
            'experience': attempt.user_level_of_musical_experience,
            'instrument': name})
        
        #assign the metadata to the hvo
        hvo_seq_four_voices.metadata = metadata_first_bar

        #Load HVO from Bongosero Collection
        hvo = attempt.load_drums_with_bongos_hvo_sequence(drum_source=dataset_dir)

        #Set Four Voice HVO to all zeros
        hvo_seq_four_voices.zeros(len(hvo.hvo))
        #if location > 4:
            #raise ValueError("Only four instruments allowed in the list. This was meant to be handled elsewhere, but I messed up. Sorry! - J")
        if name.lower() == 'left':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][0]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][0]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][0]
        elif name.lower() == 'right':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][1]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][1]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][1]
        elif name.lower() == 'both':
            for i in range(len(hvo.hvo)):
                if(hvo.hits[i][0] == 1 and hvo.hits[i][1] == 1):
                    hvo_seq_four_voices.hits[i][location] = 1
                    hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][0], hvo.velocities[i][1])
                    hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][0], hvo.offsets[i][1])
                else:
                    hvo_seq_four_voices.hits[i][location] = 0
                    hvo_seq_four_voices.velocities[i][location] = 0
                    hvo_seq_four_voices.offsets[i][location] = 0
        elif name.lower() == 'kick':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][2]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][2]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][2]
        elif name.lower() == 'snare':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][3]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][3]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][3]
        elif name.lower() == 'hh_closed':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][4]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][4]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][4]
        elif name.lower() == 'hh_open':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][5]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][5]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][5]
        elif name.lower() == 'tom_lo':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][6]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][6]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][6]
        elif name.lower() == 'tom_mid':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][7]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][7]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][7]
        elif name.lower() == 'tom_hi':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][8]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][8]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][8]
        elif name.lower() == 'crash':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][9]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][9]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][9]
        elif name.lower() == 'ride':
            for i in range(len(hvo.hvo)):
                hvo_seq_four_voices.hits[i][location] = hvo.hits[i][10]
                hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][10]
                hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][10]
        elif name.lower() == 'toms':
            for i in range(len(hvo.hvo)):
                if(hvo.hits[i][6] == 1 or hvo.hits[i][7] == 1 or hvo.hits[i][8] == 1):
                    hvo_seq_four_voices.hits[i][location] = 1
                    hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][6], hvo.velocities[i][7], hvo.velocities[i][8])
                    hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][6], hvo.offsets[i][7], hvo.offsets[i][8])
        elif name.lower() == 'hihats':
            for i in range(len(hvo.hvo)):
                if(hvo.hits[i][4] == 1 or hvo.hits[i][5] == 1):
                    hvo_seq_four_voices.hits[i][location] = 1
                    hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][4], hvo.velocities[i][5])
                    hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][4], hvo.offsets[i][5])
        elif name.lower() == 'cymbals':
            for i in range(len(hvo.hvo)):
                if(hvo.hits[i][9] == 1 or hvo.hits[i][10] == 1):
                    hvo_seq_four_voices.hits[i][location] = 1
                    hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][9], hvo.velocities[i][10])
                    hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][9], hvo.offsets[i][10])
        elif name.lower() == 'flattened':
            for i in range(len(hvo.hvo)):
                if(hvo.hits[i][2] == 1 or hvo.hits[i][3] == 1 or hvo.hits[i][4] == 1 or hvo.hits[i][5] == 1 or hvo.hits[i][6] == 1 or hvo.hits[i][7] == 1 or hvo.hits[i][8] == 1 or hvo.hits[i][9] == 1 or hvo.hits[i][10] == 10):
                    hvo_seq_four_voices.hits[i][location] = 1
                    hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][2], hvo.velocities[i][3], hvo.velocities[i][4], hvo.velocities[i][5], hvo.velocities[i][6], hvo.velocities[i][7], hvo.velocities[i][8], hvo.velocities[i][9], hvo.velocities[i][10])
                    hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][2], hvo.offsets[i][3], hvo.offsets[i][4], hvo.offsets[i][5], hvo.offsets[i][6], hvo.offsets[i][7], hvo.offsets[i][8], hvo.offsets[i][9], hvo.offsets[i][10])
        else:
            raise ValueError('Incorrect name given. Please look at readme for correct names to use.')
        
        #Update the location of the hvo sequence
        #location += 1
        
        #Save hvo sequence
        full_name = str(user.user_id) + '_attempt_num_' + str(attempt_num) + '_' + name + '.hvo'
        hvo_seq_four_voices.save(sub_dir + '/' + full_name)


import os
import multiprocessing as mp

# def process_user(args):
#     user, dataset_dir, output_dir, expert_level, rating, instrument_lists = args
#     attempt_num = 0
#     for attempt in user:
#         print(f"[{os.getpid()}] Processing {user.user_id} - Attempt {attempt_num}")
#         if attempt.user_level_of_musical_experience >= expert_level and attempt.user_exhibion_rating >= rating:
#             make_hvo_from_list(instrument_lists, attempt, dataset_dir, output_dir, attempt_num, user)
#         attempt_num += 1

def process_user(args):
    user, dataset_dir, output_dir, expert_level, rating, instrument_lists = args
    for attempt_num, attempt in enumerate(user.attempts):  # use enumerate!
        print(f"[{os.getpid()}] Processing {user.user_id} - Attempt {attempt_num}")
        if attempt.user_level_of_musical_experience >= expert_level and attempt.user_exhibion_rating >= rating:
            make_hvo_from_list(instrument_lists, attempt, dataset_dir, output_dir, attempt_num, user)


def move_through_collection(dataset_dir, output_dir, expert_level, rating, instrument_lists, n_workers):
    collection = BongoDrumCollection(dataset_dir + '/SavedSessions.bz2')

    # Prepare arguments for multiprocessing
    args_list = [
        (user, dataset_dir, output_dir, expert_level, rating, instrument_lists)
        for user in collection[:10]
    ]

    # Use multiprocessing to handle users in parallel
    with mp.Pool(processes=n_workers) as pool:
        pool.map(process_user, args_list)

#Params: Bool: Download dataset, dataset_dir, output_dir, expert_level, rating, instrument_lists
def main(params):

    #list_instruments = [['left', 'right', 'both', 'flattened'], ['left', 'right', 'snare', 'kick'], ['left', 'right', 'toms', 'kick'], ['left', 'right', 'both', 'toms'], ['left', 'right', 'both', 'hihats']]
    list_instruments = ['flattened', 'snare', 'kick', 'left', 'right', 'both', 'toms', 'hihats']
    hvo_dir = 'dataset_and_API/data/'
    output_dir = 'bongosero/hvo_sequences_single_voices'
    move_through_collection(hvo_dir, output_dir, 2, 2, list_instruments, n_workers=8)

if __name__ == "__main__":
    main(sys.argv[1:])


# Old Code

# def move_through_collection(dataset_dir, output_dir, expert_level, rating, instrument_lists):
#     #collection = BongoDrumCollection('dataset_and_API/data/SavedSessions.bz2')
#     collection = BongoDrumCollection(dataset_dir + '/SavedSessions.bz2')
#     for user in collection[:10]:
#         print('user')
#         print(user)
#         attempt_num = 0
#         for attempt in user:
#             if(attempt.user_level_of_musical_experience >= expert_level and attempt.user_exhibion_rating >= rating):
#                 make_hvo_from_list(instrument_lists, attempt, dataset_dir, output_dir, attempt_num, user)
#             attempt_num += 1



# Old Main function
    # NUM_PARAMS = 6
    # if len(params) != NUM_PARAMS:
    #     print("Main function takes only 6 parameters: Bool to download data, data dir, output dir, expert level, rating, instrument lists")
    #     sys.exit(1)
    # if(params[0]):
    #     #download(params[1])
    #     pass
    #list_of_test_instruments = [['left', 'right', 'both', 'kick'], ['left', 'right', 'snare', 'kick'], ['left', 'right', 'toms', 'kick']]
    #move_through_collection('dataset_and_API/data', 'bongosero', 2, 2, list_of_test_instruments)


    ##This was creating an hvo sequence with four voices, but in the other examples we are using single voices so we can switch what the groove is
    ##
    ##--------------------------------
    # def make_hvo_from_list(instrument_lists, attempt, dataset_dir, output_dir, attempt_num, user):
    # #Check if output directory exists and create it if it does not exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # #create the subdirectory if it doesn't exist
    # sub_dir = output_dir + '/' + str(user.user_id)
    # if not os.path.exists(sub_dir):
    #     os.makedirs(sub_dir)

    # for instrument_list in instrument_lists:
    #     print('user_id_' + str(user.user_id))
    #     print('attempt_num_' + str(attempt_num))
    #     print(instrument_list)
    #     print(attempt.user_level_of_musical_experience)
    #     print(attempt.user_exhibion_rating)
    #     #We set each voice to a Midi Mapping value for Drums
    #     FOUR_VOICES = {
    #         "voice_1": [36],
    #         "voice_2": [37],
    #         "voice_3": [38],
    #         "voice_4": [39]
    #     }

    #     #Create new HVO Sequence with beat division of 4 and set Drum Mapping to Four_Voices Mapping
    #     hvo_seq_four_voices = HVO_Sequence(
    #         beat_division_factors=[4],
    #         drum_mapping=FOUR_VOICES)
        
        
    #     #set tempo
    #     hvo_seq_four_voices.add_tempo(
    #         time_step=0,
    #         qpm=attempt.attempt_tempo
    #     )

    #     #Add a time signature of 4/4 for the HVO Sequence
    #     hvo_seq_four_voices.add_time_signature(
    #         time_step=0,
    #         numerator=4,
    #         denominator=4
    #     )

    #     #Create metadata for hvo
    #     metadata_first_bar = Metadata({
    #         'self_assesment': attempt.self_assessment,
    #         'genre': attempt.genre,
    #         'tempo': attempt.attempt_tempo,
    #         'experience': attempt.user_level_of_musical_experience,
    #         'instrument_order': instrument_list})
        
    #     #assign the metadata to the hvo
    #     hvo_seq_four_voices.metadata = metadata_first_bar

    #     #Load HVO from Bongosero Collection
    #     hvo = attempt.load_drums_with_bongos_hvo_sequence(drum_source=dataset_dir)

    #     #Set Four Voice HVO to all zeros
    #     hvo_seq_four_voices.zeros(len(hvo.hvo))
        
    #     #location variable to keep track of which of the 4 streams we are creating
    #     location = 0

    #     for name in instrument_list:
    #         if location > 4:
    #             raise ValueError("Only four instruments allowed in the list. This was meant to be handled elsewhere, but I messed up. Sorry! - J")
    #         if name.lower() == 'left':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][0]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][0]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][0]
    #         elif name.lower() == 'right':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][1]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][1]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][1]
    #         elif name.lower() == 'both':
    #             for i in range(len(hvo.hvo)):
    #                 if(hvo.hits[i][0] == 1 and hvo.hits[i][1] == 1):
    #                     hvo_seq_four_voices.hits[i][location] = 1
    #                     hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][0], hvo.velocities[i][1])
    #                     hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][0], hvo.offsets[i][1])
    #                 else:
    #                     hvo_seq_four_voices.hits[i][location] = 0
    #                     hvo_seq_four_voices.velocities[i][location] = 0
    #                     hvo_seq_four_voices.offsets[i][location] = 0
    #         elif name.lower() == 'kick':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][2]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][2]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][2]
    #         elif name.lower() == 'snare':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][3]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][3]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][3]
    #         elif name.lower() == 'hh_closed':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][4]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][4]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][4]
    #         elif name.lower() == 'hh_open':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][5]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][5]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][5]
    #         elif name.lower() == 'tom_lo':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][6]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][6]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][6]
    #         elif name.lower() == 'tom_mid':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][7]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][7]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][7]
    #         elif name.lower() == 'tom_hi':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][8]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][8]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][8]
    #         elif name.lower() == 'crash':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][9]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][9]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][9]
    #         elif name.lower() == 'ride':
    #             for i in range(len(hvo.hvo)):
    #                 hvo_seq_four_voices.hits[i][location] = hvo.hits[i][10]
    #                 hvo_seq_four_voices.velocities[i][location] = hvo.velocities[i][10]
    #                 hvo_seq_four_voices.offsets[i][location] = hvo.offsets[i][10]
    #         elif name.lower() == 'toms':
    #             for i in range(len(hvo.hvo)):
    #                 if(hvo.hits[i][6] == 1 or hvo.hits[i][7] == 1 or hvo.hits[i][8] == 1):
    #                     hvo_seq_four_voices.hits[i][location] = 1
    #                     hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][6], hvo.velocities[i][7], hvo.velocities[i][8])
    #                     hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][6], hvo.offsets[i][7], hvo.offsets[i][8])
    #         elif name.lower() == 'hihats':
    #             for i in range(len(hvo.hvo)):
    #                 if(hvo.hits[i][4] == 1 or hvo.hits[i][5] == 1):
    #                     hvo_seq_four_voices.hits[i][location] = 1
    #                     hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][4], hvo.velocities[i][5])
    #                     hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][4], hvo.offsets[i][5])
    #         elif name.lower() == 'cymbals':
    #             for i in range(len(hvo.hvo)):
    #                 if(hvo.hits[i][9] == 1 or hvo.hits[i][10] == 1):
    #                     hvo_seq_four_voices.hits[i][location] = 1
    #                     hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][9], hvo.velocities[i][10])
    #                     hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][9], hvo.offsets[i][10])
    #         elif name.lower() == 'flattened':
    #             for i in range(len(hvo.hvo)):
    #                 if(hvo.hits[i][2] == 1 or hvo.hits[i][3] == 1 or hvo.hits[i][4] == 1 or hvo.hits[i][5] == 1 or hvo.hits[i][6] == 1 or hvo.hits[i][7] == 1 or hvo.hits[i][8] == 1 or hvo.hits[i][9] == 1 or hvo.hits[i][10] == 10):
    #                     hvo_seq_four_voices.hits[i][location] = 1
    #                     hvo_seq_four_voices.velocities[i][location] = max(hvo.velocities[i][2], hvo.velocities[i][3], hvo.velocities[i][4], hvo.velocities[i][5], hvo.velocities[i][6], hvo.velocities[i][7], hvo.velocities[i][8], hvo.velocities[i][9], hvo.velocities[i][10])
    #                     hvo_seq_four_voices.offsets[i][location] = max(hvo.offsets[i][2], hvo.offsets[i][3], hvo.offsets[i][4], hvo.offsets[i][5], hvo.offsets[i][6], hvo.offsets[i][7], hvo.offsets[i][8], hvo.offsets[i][9], hvo.offsets[i][10])
    #         else:
    #             raise ValueError('Incorrect name given. Please look at readme for correct names to use.')
            
    #         #Update the location of the hvo sequence
    #         location += 1
            
    #         #Save hvo sequence
    #         name = '_'.join(instrument_list)
    #         full_name = str(user.user_id) + '_attempt_num_' + str(attempt_num) + '_' + name + '.hvo'
    #         hvo_seq_four_voices.save(sub_dir + '/' + full_name)


# def download(dir):
#     url = "https://elbongosero.github.io/assets/dataset_and_API.zip"
#     zip_path = dir
#     extract_dir = dir

#     # Download the ZIP file
#     response = requests.get(url)
#     with open(zip_path, 'wb') as f:
#         f.write(response.content)

#     # Unzip the file
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_dir)

#     # Optional: remove ZIP file after extraction
#     os.remove(zip_path)
