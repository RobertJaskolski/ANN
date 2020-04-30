# Import (Tutaj umieszczamy wszystkie biblioteki których będziemy używać)
import librosa
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# DataLoading (Tutaj wczytamy nasze pliki dźwiękowe z danej ścieżki,
# a nastepnie przekonwertujemy to do danych zrozumiałych dla komputera i zapiszemy w ppliku .cvs)
def dataLoading(csvFilename):
    # Przygotowanie nagłówka naszego pliku .csv (Nazwa oraz Stosowane funkcje)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(csvFilename, 'w', newline='')
    with  file:
        writer = csv.writer(file)
        writer.writerow(header)
    # Dźwięki które będziemy wykrywać
    sounds = 'Baby-cry Chainsaw Clock-tick Dog-bark Fire-crackling Helicopter Person-sneeze Rain Rooster Sea-waves'.split()
    for s in sounds:
        for filename in os.listdir(f"./ESC-10/{s}"):
            # Tutaj sobie wczytamy konkretny dźwięk i przekonwertujemy go za pomocą librosy
            songname = f'./ESC-10/{s}/{filename}'
            y, sr = librosa.load(songname, mono=True)

            # Tutaj zaczynamy zapisywanie naszych funkcji
            # rmse
            rmse = librosa.feature.rms(y=y)
            # chroma
            stft = librosa.feature.chroma_stft(y=y, sr=sr)
            # spectral
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            bawi = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolo = librosa.feature.spectral_rolloff(y=y, sr=sr)
            # zero crossing rate
            zero = librosa.feature.zero_crossing_rate(y)
            # mfcc
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            # Teraz będziemy to wszystko układać w stringu a na końcu zapiszemy do naszego arkusza
            toAppend = f'{filename} {np.mean(stft)} {np.mean(rmse)} {np.mean(cent)} {np.mean(bawi)} {np.mean(rolo)} '
            toAppend += f'{np.mean(zero)}'
            for result in mfcc:
                toAppend += f' {np.mean(result)}'
            toAppend += f' {s}'
            file = open(csvFilename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(toAppend.split())
def forOneAudio():
    pass # Tutaj będziemy dodawać jedno audio dla sprawdzenia
def nn():
    # Załadowanie danych oraz przygotowanie ich do naszej sieci ANN
    data = pd.read_csv('data.csv')
    data.head()
    data = data.drop(['filename'], axis=1)
    soundList = data.iloc[:,-1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(soundList)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    # X - są to dane do uczenia naszej sieci a y - to nr dźwięków od 0-9
    # Podział jest prosty do trenowania 80% czyli 320 wektorów a do testowania 80 wektorów
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Tutaj określamy nasze ANN

    model = Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    classifier = model.fit(X_train,
                           y_train,
                           epochs=80,
                           batch_size=128)

    return model


# Funkcja główna
if __name__ == '__main__':
    # dataLoading("data.csv")
    network = nn()
