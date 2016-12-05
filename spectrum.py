import cv2
import numpy as np
import sys

from collections import namedtuple
from math import sin, pi
from pickle import load
from sklearn import svm

Color = namedtuple("Color", ["red", "green", "blue"])
model = load(open("credits_detection/model.p", "r"))


def process_frame_for_prediction(frame):
    frame = cv2.resize(frame, (100, frame.shape[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_colors = []

    for i in xrange(frame.shape[1]):
        mean_colors.append(np.mean(frame[:, i]))

    return mean_colors


def is_credit(frame):
    frame = process_frame_for_prediction(frame)

    if model.predict([frame]) == [[1]]:
        return True

    return False


def gradient(img, length=0.25):
    rows, cols = img.shape[:2]
    output = np.copy(img)

    def grad_function(x):
        y = sin(pi/2 * x/(rows * length))
        return y if x <= rows * length else 1.0

    grad_function = np.vectorize(grad_function)

    vec = grad_function(np.arange(0, rows)).reshape(-1, 1)
    gradient = vec * vec[::-1]

    for j in xrange(cols):
        output[:, j] = output[:, j] * gradient

    return output


def mean_color(frame):
    red_pixels = frame[:, :, 2]
    green_pixels = frame[:, :, 1]
    blue_pixels = frame[:, :, 0]

    return Color(
        red=int(round(np.mean(red_pixels))),
        green=int(round(np.mean(green_pixels))),
        blue=int(round(np.mean(blue_pixels)))
    )


def main(infile):
    cap = cv2.VideoCapture(infile)
    frames_processed, frames_dropped = 0, 0
    frame_averages = []

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        frames_processed += 1

        if is_credit(frame):
            frames_dropped += 1
            continue

        frame_average = mean_color(frame)
        frame_averages.append(frame_average)

        if frames_processed % 1000 == 0:
            print "Processed", frames_processed, "frames", frame_average
            print "Dropped", frames_dropped, "frames"

    cap.release()

    height = 2000
    width = 8000

    outimg = np.zeros(
        (height, len(frame_averages), 3),
        np.uint8
    )

    for i, frame_color in enumerate(frame_averages):
        outimg[:, i, 0] = frame_color.blue
        outimg[:, i, 1] = frame_color.green
        outimg[:, i, 2] = frame_color.red

    outimg = cv2.resize(outimg, (width, height))
    with_gradient = gradient(outimg, 0.4)

    cv2.imwrite("spectrum.png", outimg)
    cv2.imwrite("spectrum_vignette.png", with_gradient)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "USAGE:", sys.argv[0], "[video_file]"
        sys.exit(1)

    main(sys.argv[1])
