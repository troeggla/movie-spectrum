import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from pickle import dump


def get_frames_from_file(infile):
    cap = cv2.VideoCapture(infile)

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        frame = cv2.resize(frame, (100, frame.shape[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_colors = []

        for i in xrange(frame.shape[1]):
            mean_colors.append(np.mean(frame[:, i]))

        yield mean_colors

    cap.release()


def get_dataset(files, target):
    X = []

    for f in files:
        for frame in get_frames_from_file(f):
            X.append(frame)

    X = np.array(X)
    rows = X.shape[0]

    return X, np.full((rows, 1), target, dtype=int)


def main():
    credits_X, credits_y = get_dataset(
        ["credits/mad_max.mov", "credits/under_the_skin.mov"],
        1
    )

    print "credits done:", credits_X.shape, credits_y.shape

    content_X, content_y = get_dataset(
        ["content/mad_max.mov", "content/under_the_skin.mov"],
        0
    )

    print "content done:", content_X.shape, content_y.shape

    X = np.vstack((credits_X, content_X))
    y = np.vstack((credits_y, content_y)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print "training model..."

    model = svm.SVC()
    model.fit(X_train, y_train)

    print "score:", model.score(X_test, y_test)
    print "dumping model..."

    dump(model, open("model.p", "wb"))


if __name__ == "__main__":
    main()
