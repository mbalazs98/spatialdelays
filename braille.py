import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class braille_dataset:
    def __init__(self):
        self.threshold = 1
        self.nb_input_copies = 1
        self.time_bin_size = 4
        # Load data and parameters
        file_dir_data = '../data/reading_braille_data/'
        file_type = 'data_braille_letters_th_'
        file_thr = str(self.threshold)
        file_name = file_dir_data + file_type + file_thr + '.pkl'

        # load data
        self.load_and_extract(file_name, letter_written=letters)



    def load_and_extract(self, file_name, letter_written=letters):


        data_dict = pd.read_pickle(file_name)
        # Extract data
        data = []
        labels = []
        bins = 1000  # ms conversion
        # loop over all trials
        for i, sample in enumerate(data_dict['events']):
            ids = []
            times = []
            # loop over sensors (taxel)
            for taxel in range(len(sample)):
                
                # loop over On and Off channels
                for event_type in range(len(sample[taxel])):
                    if sample[taxel][event_type]:
                        indx = bins*(np.array(sample[taxel][event_type]))
                        indx = np.array((indx/self.time_bin_size).round(), dtype=int)
                        for ind in indx:
                            for copies in range(self.nb_input_copies):
                                ids.append(taxel + event_type * 12 + 24 * copies)
                                times.append(ind)
            data.append((np.array(times), np.array(ids)))
            labels.append(letter_written.index(data_dict['letter'][i]))

        # create 80/20 train/test
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.20, shuffle=True, stratify=labels)


        self.x_train_braille = x_train
        self.y_train_braille  = y_train
        self.x_test_braille  = x_test 
        self.y_test_braille  = y_test



    
