import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings(
        "ignore", category=DeprecationWarning)
warnings.filterwarnings(
        "ignore", category=ConvergenceWarning)


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
    # data.hist(bins=20, figsize=(15, 15))
    # plt.show()
    # izbacujemo neadekvatne kolone
    data.drop(['liveness', 'acousticness', 'speechiness'], inplace=True, axis=1)
    y = data['genre']
    y = dedupe_genres(y)

    X = data.drop('genre', axis=1)

    return train_test_split(X, y, test_size=0.33)


def svm(X_train, y_train):
    std_scaler = StandardScaler()
    X_scaled_train = std_scaler.fit_transform(X_train)
    svm_clf = OneVsRestClassifier(LinearSVC(C=0.01, loss="hinge", random_state=1))
    svm_clf.fit(X_scaled_train, y_train)
    preds = svm_clf.predict(X_scaled_train)
    print(classification_report(y_train, preds))

def logistic_regression(X_train, y_train):
    ovr_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=1))
    ovr_clf.fit(X_train, y_train)
    y_test_pred = ovr_clf.predict(X_test)
    confusion_matrix(y_test, y_test_pred)
    print(accuracy_score(y_test, y_test_pred))

def random_forest_classifier(X_train, y_train):
    rnd_clf = RandomForestClassifier(n_estimators=25, max_leaf_nodes=16, n_jobs=-1, random_state=1)
    rnd_clf.fit(X_train, y_train)
    ypred = rnd_clf.predict(X_test)
    print(accuracy_score(y_test, ypred))



def grid_search(X_train, y_train):
    SVCpipe = Pipeline([('scale', StandardScaler()),
                        ('SVC', LinearSVC())])

    param_grid = {'SVC__C': np.arange(0.01, 100, 10)}
    linearSVC = GridSearchCV(SVCpipe, param_grid, cv=5, return_train_score=True)
    linearSVC.fit(X_train, y_train)
    print(linearSVC.best_params_)  # najbolji za C je: {'SVC__C': 0.01}

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('data/Spotify-2000.csv')
    # svm(X_train, y_train)
    #logistic_regression(X_train,y_train)
    #random_forest_classifier(X_train,y_train)
    pca = PCA(0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    log_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, penalty="l2", C=1, random_state=1))
    rnd_clf = RandomForestClassifier(random_state=1)
    svm_clf = OneVsRestClassifier(LinearSVC(C=0.01, loss="hinge", random_state=1))
    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, ypred))

