import cv2
import numpy as np


def get_frames_from_file(infile):
    cap = cv2.VideoCapture(infile)

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        yield frame

    cap.release()


def get_dataset(files, target, process_frame, subsample=None):
    X = []

    for f in files:
        for frame in get_frames_from_file(f):
            X.append(process_frame(frame))

    X = np.array(X)
    rows = X.shape[0]

    if subsample:
        indices = np.random.choice(rows, subsample)
        rows = subsample
        X = X[indices]

    return X, np.full((rows, 1), target, dtype=int)
