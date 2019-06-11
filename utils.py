import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import os
from pydub import AudioSegment
import numpy as np
import random
from itertools import permutations
import numpy as np
import random
import sys
import io
import os
import glob
import IPython

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def process_sliding_window(flac_file, processed_directory_path,filename):
    window_size = 1600
    step_size = 800
    actual_idx = 0
    amount_of_files = 0
    new_base_directory = processed_directory_path +filename[:-5]+"/"
    new_base_path = (new_base_directory+filename)[:-5]

    if not os.path.exists(new_base_directory):
        	os.makedirs(new_base_directory)

    while(len(flac_file) > actual_idx + window_size):
        flac_file[actual_idx:actual_idx+window_size].export(new_base_path+'_'+str(amount_of_files) + '.wav',format='wav')
        amount_of_files += 1
        actual_idx += step_size
    if (actual_idx + window_size - len(flac_file) > window_size/2 and len(flac_file) > window_size):
        flac_file[-window_size:].export(new_base_path+'_'+str(amount_of_files) + '.wav',format='wav')
        

def make_processed_dataset(libri_speech_path,processed_libri_speech_path):
    dev_path = libri_speech_path + "dev-clean/"
    processed_dev_path = processed_libri_speech_path + "dev-clean/"
    voice_count = 0
    for voice in (os.listdir(dev_path)):
        voice_path = dev_path + voice + "/"
        for chapter in (os.listdir(voice_path)):
            chapter_path = voice_path + chapter + "/"
            for filename in os.listdir(chapter_path):
                processed_directory_path = processed_dev_path + voice + "/" + chapter + "/" 
                if filename.endswith("flac"):
                    flac_file = AudioSegment.from_file(chapter_path + filename, "flac")
                    process_sliding_window(flac_file,processed_directory_path,filename)
        voice_count += 1
        print ("actual progress", str (int(voice_count/len(os.listdir(dev_path))*100)), ('%'))
        
def process_16_dataset_main():
    dataset_base_path = "./"
    libri_speech_path = dataset_base_path + "LibriSpeech/"
    processed_libri_speech_path = dataset_base_path + "16_LibriSpeech/"
    make_processed_dataset(libri_speech_path, processed_libri_speech_path)


def make_numpy_XY(libri_speech_path):
    dev_path = libri_speech_path + "dev-clean/"
    voice_count = 0

    X = []
    Y = []
    for voice in (os.listdir(dev_path)):
        voice_path = dev_path + voice + "/"
        count = 0     
        X_5 = []
        for chapter in (os.listdir(voice_path)):
            chapter_path = voice_path + chapter + "/"
            for uterrance in os.listdir(chapter_path):
              uterrance_path = chapter_path + uterrance + "/"
              for filename in os.listdir(uterrance_path):
                if filename.endswith("wav"):
                    count += 1
                    wav_file = graph_spectrogram(uterrance_path + filename)
                    X_5.append(wav_file)
                if (count % 5 == 0 and count != 0):
                    X.append(X_5)
                    X_5 = []
                    Y.append(voice)
                if (count >= 100):
                    break
            if count>=100:
                break
        voice_count+=1
        print ("actual progress", str (int(voice_count/len(os.listdir(dev_path))*100)), ('%'))
    return X,Y

def process_numpy_XY():
    dataset_base_path = "./"
    processed_libri_speech_path = dataset_base_path + "16_LibriSpeech/"
    X,Y = make_numpy_XY(processed_libri_speech_path)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def reshape_X(X):
  examples = X.shape[0]
  dimension = X.shape[1]
  time = X.shape[2]
  freq = X.shape[3]
  new_X = np.zeros((dimension,examples,time,freq))
  for m in range(examples):
    for d in range(dimension):
      new_X[d][m] = X[m][d]
  return new_X
    
    
def load_numpy_XY():
    X_path = './preprocessed/X_total.npy'
    Y_path = './preprocessed/Y_total.npy'
    X = np.load(X_path)
    Y = np.load(Y_path)
    return X,Y

def process_dataset():
    print ('processing dataset:')
    process_16_dataset_main()
    print ('----------------')
    print ('Making XY:')
    X,Y = process_numpy_XY()
    return X,Y

def split_dataset(X,Y,portion):
    X_train, X_test = [], []
    Y_train, Y_test = [], []
    for i in range (X.shape[0]):
        if np.random.random() < portion:
            X_train.append(X[i])
            Y_train.append(Y[i])
        else:
            X_test.append(X[i])
            Y_test.append(Y[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_train, Y_train, X_test, Y_test
        
def save_np_file(path,file):
    np.save(path,file)

def load_np_file(path):
    return np.load(path)

