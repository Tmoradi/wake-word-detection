import os 
from glob import glob 
import pandas as pd



def create_labels_csv(src_path:str) -> None: 
    ''' Here we are going to be cuarting a csv of our labels 
        for the data based on the data directory structure.

        args:
            src_path(str): that path of the whole data directory; when using glob 
                            we'll be able to get a list of all mel spectrograms 
                            for our dataset.    
    '''
    labels_df = pd.DataFrame()
    # list_relative_paths =  glob(r"data\**\**\*.png")
    list_relative_paths = glob(src_path)
    labels_df['paths'] = list_relative_paths
    labels_df['paths'] = labels_df.apply(lambda x: x.replace(r"\\\\",r"\\"))
    labels_df['label'] = 0
    labels_df.loc[(labels_df.paths.str.contains('c1')),'label'] = 1

    train_df = labels_df.loc[labels_df.paths.str.contains('train')].reset_index()
    test_df = labels_df.loc[~labels_df.paths.str.contains('train')].reset_index()

    train_df.to_csv(os.path.join('data','train_labels.csv'))
    test_df.to_csv(os.path.join('data','test_labels.csv'))

if __name__ == "__main__":
    create_labels_csv(src_path=r"data\**\**\*.png")