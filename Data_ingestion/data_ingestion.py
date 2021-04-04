import pandas as pd
import bz2
import _pickle as cPickle

class data_getter:

    def data_load(self, file):
        '''
                    Description: This method loads the data from the file and convert into a pandas dataframe.
                    Output: Returns a Dataframes, which is our data for training.
                    On Failure: Raise Exception.
        '''
        try:
            data = pd.read_csv(file,encoding = "ISO-8859-1",error_bad_lines=False)
            return data
        except Exception as e:
            raise e

    # Load any compressed pickle file
    def decompress_pickle(self,file):
        '''
                           Description: This method loads the compressed pickle files from pickle file folder.
                           Output: Returns the model defined in compressed pickle files.
                           On Failure: Raise Exception.
        '''
        try:
            data = bz2.BZ2File(file, 'rb')
            data = cPickle.load(data)
            return data
        except Exception as e:
            raise e

