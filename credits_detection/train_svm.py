import cv2
import numpy as np

from argparse import ArgumentParser
from glob import glob
from pickle import dump
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from time import time
from util import get_dataset


def setup_model():
    svm = load(open("credits_detection/models/svm.p", "r"))

    def is_credit(frame):
        frame = process_frame(frame)
        return svm.predict([frame]) == [[1]]

    return is_credit


def process_frame(frame):
    frame = cv2.resize(frame, (100, frame.shape[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)

    col_mean = np.mean(frame, axis=0)
    col_mean = col_mean - np.min(col_mean)

    if np.max(col_mean) > 0:
        col_mean = col_mean / np.max(col_mean)

    return col_mean


def crossvalidate(model, X, y):
    scores = cross_val_score(model, X, y, cv=KFold(shuffle=True), scoring="f1")

    print "score statistics:", scores.mean(), scores.std() * 2
    print "individual scores:", scores


def train_and_dump_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)
    print "score:", model.score(X_test, y_test)

    print "dumping model..."
    dump(model, open("models/svm.p", "wb"))

    print "testing on unseen data..."
    credits_X, credits_y = get_dataset(
        ["credits/john_wick.mov"], 1,
        process_frame
    )
    print "data loaded:", credits_X.shape, credits_y.shape

    print "score:", model.score(credits_X, credits_y)


def main(cross_validate=False):
    start = time()
    print "loading data..."

    credits_X, credits_y = get_dataset(
        ["credits/mad_max.mov", "credits/under_the_skin.mov"], 1,
        process_frame, 10000
    )
    print "credits done:", credits_X.shape, credits_y.shape

    content_X, content_y = get_dataset(
        glob("./content/*.mov"), 0,
        process_frame, 10000
    )
    print "content done:", content_X.shape, content_y.shape
    print "took", time() - start, "sec"

    X = np.vstack((credits_X, content_X))
    y = np.vstack((credits_y, content_y)).ravel()

    model = SVC()

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
