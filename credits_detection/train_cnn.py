import cv2
import numpy as np

from glob import glob
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from time import time


def process_frame(frame):
    frame = cv2.resize(frame, (30, 30))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if np.max(frame) > 0:
        frame = frame / float(np.max(frame))

    frame = frame[..., np.newaxis]

    return frame


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


def build_model():
    model = Sequential()

    model.add(Convolution2D(20, 5, 5, input_shape=(30, 30, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dropout(0.1))

    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01),
        metrics=["accuracy"]
    )

    return model


def train_and_dump_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    print "training model..."
    model.fit(X_train, y_train, batch_size=128, nb_epoch=15, verbose=1)

    print "evaluating..."
    loss, accuracy = model.evaluate(
        X_test, y_test,
        batch_size=128, verbose=1
    )
    print "accuracy: {:.2f}%".format(accuracy * 100)

    print("dumping weights to file...")
    model.save_weights("models/cnn.h5", overwrite=True)


def main():
    start = time()
    print "loading data..."

    credits_X, credits_y = get_dataset([
        "credits/mad_max.mov",
        "credits/under_the_skin.mov"
    ], 1, 20000)
    print "credits done:", credits_X.shape, credits_y.shape

    content_X, content_y = get_dataset(glob("./content/*.mov"), 0, 20000)
    print "content done:", content_X.shape, content_y.shape

    print "took", time() - start, "sec"

    X = np.vstack((credits_X, content_X))
    y = np.vstack((credits_y, content_y)).ravel()

    model = build_model()
    train_and_dump_model(model, X, y)

    print "testing on unseen data..."
    credits_X, credits_y = get_dataset(["credits/john_wick.mov"], 1)
    print "data loaded:", credits_X.shape, credits_y.shape

    credits_y = np_utils.to_categorical(credits_y, 2)
    print "score:", model.evaluate(credits_X, credits_y, batch_size=128)


if __name__ == "__main__":
    main()
