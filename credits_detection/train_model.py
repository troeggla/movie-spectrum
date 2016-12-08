import cv2
import numpy as np
import sys

from argparse import ArgumentParser
from glob import glob
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from time import time


def process_frame(frame):
    frame = cv2.resize(frame, (100, frame.shape[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return np.mean(frame, axis=0)


def get_frames_from_file(infile):
    cap = cv2.VideoCapture(infile)

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        yield process_frame(frame)

    cap.release()


def get_dataset(files, target, subsample=None):
    X = []

    for f in files:
        for frame in get_frames_from_file(f):
            X.append(frame)

    X = np.array(X)
    rows = X.shape[0]

    if subsample:
        indices = np.random.choice(rows, subsample)
        rows = subsample
        X = X[indices]

    return X, np.full((rows, 1), target, dtype=int)


def crossvalidate(model, X, y):
    scores = cross_val_score(model, X, y, cv=KFold(shuffle=True), scoring="f1")

    print "score statistics:", scores.mean(), scores.std() * 2
    print "individual scores:", scores


def train_and_dump_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)
    print "score:", model.score(X_test, y_test)

    print "dumping model..."
    dump(model, open("model.p", "wb"))


def main(cross_validate=False):
    start = time()
    print "loading data..."

    credits_X, credits_y = get_dataset(glob("./credits/*.mov"), 1, 5000)
    print "credits done:", credits_X.shape, credits_y.shape

    content_X, content_y = get_dataset(glob("./content/*.mov"), 0, 5000)
    print "content done:", content_X.shape, content_y.shape

    print "took", time() - start, "sec"

    X = np.vstack((credits_X, content_X))
    y = np.vstack((credits_y, content_y)).ravel()

    model = RandomForestClassifier()

    if cross_validate:
        print "training model with cross-validation..."
        crossvalidate(model, X, y)
    else:
        print "training model..."
        train_and_dump_model(model, X, y)

    print "DONE"
    print "took", time() - start, "sec"


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Train a machine learning classifier to detect credit
        sequences in video files and dump the generated model to file. If
        cross-validation is activated, no model will be dumped.

        The training data needs to be stored as MOV files in the folders
        credits/ and content/ respectively.
        """
    )

    parser.add_argument(
        "--cv",
        action="store_true", default=False,
        help="Perform cross-validation and do not dump model"
    )

    args = parser.parse_args()
    main(cross_validate=args.cv)
