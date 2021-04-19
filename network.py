import numpy as np
from keras import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

if __name__ == '__main__':
    # Wczytywanie danych z pliku csv
    dataframe = pd.read_csv(filepath_or_buffer="textures_data.csv", sep=',')
    data = dataframe.to_numpy()

    # Wyodrębnić wektory cech do macierzy X
    X = data[:, :-1].astype('float')
    # X = np.asarray(X).astype(np.float32)
    # Wyodrębnić etykiety kategorii do wektora y
    y = data[:, -1]
    # etykiety na int
    label_encoder = LabelEncoder()
    integer_encoder = label_encoder.fit_transform(y)
    # int na 1 z n
    onehot_encoder = OneHotEncoder()
    integer_encoder = integer_encoder.reshape(len(integer_encoder), 1)
    onehot_encoder = onehot_encoder.fit_transform(integer_encoder)
    # podzial na czesc treningowa i testowa; część treningowa (70%); testowa (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoder, test_size=0.3)

    # Tworzenie modelu sieci neuronowe typu Sequential
    # Dodać dwie warstwy typu Dense. W pierwszej warstwie ustawić liczbę perceptronów na 10,
    # funkcja aktywacji – sigmoid, wymiar wejściowy – input_dim = 72 (tyle, ile jest cech w zbiorze danych).
    # W drugiej warstwie ustawić liczbę neuronów na 3 (tyle ile jest klas) oraz funkcję aktywacji na softmax
    model = Sequential()
    model.add(Dense(10, input_dim=72, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    # Skompilować model z funkcją straty categorical_crossentropy
    # oraz algorytmem optymalizacji – sgd (=stochastic gradient descent).
    # Dodać accuracy jako metrykę wyliczaną w trakcie uczenia.
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.summary()

    # Uczenie sieci – wywołać na rzecz modelu funkcję fit
    # X_train = np.asarray(X_train).astype('int64')
    model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)
    # Przekonwertować wektor y_test oraz y_pred do kodowania całkowitego
    y_pred = model.predict(X_test)
    y_pred_int = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    # Na podstawie otrzymanych wektorów etykiet całkowitoliczbowych wyliczyć macierz pomyłek
    cm = confusion_matrix(y_test_int, y_pred_int)
    print(f'{cm}')
    pass
