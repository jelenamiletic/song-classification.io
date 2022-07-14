import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC


def genre_splitter(genre):
    result = genre.copy()
    result = result.str.split(" ", 1)
    for i in range(len(result)):
        if len(result[i]) > 1:
            result[i] = [result[i][1]]
    return result.str.join('')


def dedupe_genres(genre_column):
    while max((genre_column.str.split(" ", 1)).str.len()) > 1:
        genre_column = genre_splitter(genre_column)
    return genre_column


def load_data(path: str):
    data = pd.read_csv(path)

    data.columns = ['index', 'title', 'artist', 'genre', 'year', 'bpm', 'energy', 'danceability', 'db', 'liveness',
                    'valence', 'duration', 'acousticness', 'speechiness', 'popularity']

    # ne zelimo ovi podaci da uticu na predikcije
    data.drop(['artist', 'index', 'title'], inplace=True, axis=1)

    data = data.dropna()

    # gledamo da li su neke kolone neravnomerne
    data.hist(bins=20, figsize=(15, 15))
    # plt.show()

    # izbacujemo neadekvatne kolone
    data.drop(['db', 'liveness', 'acousticness', 'speechiness'], inplace=True, axis=1)

    y = data['genre']

    y = dedupe_genres(y)

    X = data.drop('genre', axis=1)

    return train_test_split(X, y, test_size=0.33)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('data/Spotify-2000.csv')

    std_scaler = StandardScaler()
    X_scaled_train = std_scaler.fit_transform(X_train)
    svm_clf = OneVsRestClassifier(LinearSVC(C=1, loss="hinge", random_state=1, max_iter=10_000))
    svm_clf.fit(X_scaled_train, y_train)

