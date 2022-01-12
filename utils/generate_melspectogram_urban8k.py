import progressbar
import time
import os
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa # for sound processing
import DataCollection as dc # local module

data = pd.read_csv("../store/dataset/UrbanSound8K/metadata/UrbanSound8K.csv")
data.head()

###data = data.sort_values(by=['fold', 'classID',"fsID"], ascending=[True, True, True])
dataset = np.zeros(shape=(data.shape[0], 2), dtype=object)
dataset.shape  # dataset is the array in which value will be saved

bar = progressbar.ProgressBar(maxval=data.shape[0], widgets=[
                              progressbar.Bar('$', '||', '||'), ' ', progressbar.Percentage()])
bar.start()
for i in range(data.shape[0]):

    fullpath, class_id = dc.path_class(data, data.slice_file_name[i])
    try:
        X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.melspectrogram(
            y=X, sr=sample_rate).T, axis=0)
    except Exception:
        print("Error encountered while parsing file: ", fullpath)
        mfccs, class_id = None, None
    feature = mfccs
    label = class_id
    dataset[i, 0], dataset[i, 1] = feature, label

    bar.update(i+1)

np.save("dataset_melspectrogram", dataset, allow_pickle=True)

l = np.load("../store/dataset/melspectrogram_urban8k.npy")
print(l.shape)