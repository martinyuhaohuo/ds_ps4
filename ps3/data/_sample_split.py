import hashlib

import numpy as np


def hash_int(ID):
    """Convert a numerical/string ID into a hash interger

    Parameters
    ----------
    ID : string or interger
        id value

    Returns
    -------
    interger
        the hash interger representation of the ID
    """
    string_id = str(ID)
    return int(hashlib.md5(string_id.encode()).hexdigest(),16)

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # why we do not use a method such that: first assign hash interger representation to each ID
    # then sort these has intergers to get exactly first 80% of observation as train set, left 20% as test set
    # this is because sorting by hash number is time consuming as scale of the dataset increases
    # on the other hand, although split by modulo > 20 method only guarantee that in expectation 20% of data fall to test set
    # it is fast and efficient in large scale (as data scale increases, test set proportition closer and closer to 20%)
    df["hash"] = df[id_column].apply(hash_int)
    df["sample"] = df["hash"].apply(lambda x: "test" if x%100 >= training_frac*100 else "train")
    df = df.drop("hash", axis=1)
    return df
