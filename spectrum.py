import cv2
import numpy as np
import sys
from collections import namedtuple
from math import sin, pi

Color = namedtuple("Color", ["red", "green", "blue"])


def gradient(img, length=0.25):
    rows, cols = img.shape[:2]
    output = np.copy(img)

    def grad_function(x):
        y = sin(pi/2 * x/(rows * length))
        return y if x <= rows * length else 1.0

    grad_function = np.vectorize(grad_function)

    vec = grad_function(np.arange(0, rows)).reshape(-1, 1)
    rev_vec = vec[::-1]

    for j in xrange(cols):
        output[:, j] = output[:, j] * vec
        output[:, j] = output[:, j] * rev_vec

    return output


def mean_color(frame):
    red_pixels = frame[:, :, 0]
    green_pixels = frame[:, :, 1]
    blue_pixels = frame[:, :, 2]

    return Color(
        int(round(np.mean(red_pixels))),
        int(round(np.mean(green_pixels))),
        int(round(np.mean(blue_pixels)))
    )


def main(infile):
    cap = cv2.VideoCapture(infile)
    frames_processed = 0
    frame_averages = []

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        frame_average = mean_color(frame)
        frame_averages.append(frame_average)

        frames_processed += 1
        if frames_processed % 1000 == 0:
            print "Processed", frames_processed, "frames", frame_average

    cap.release()

    height = 2000
    width = 8000

    outimg = np.zeros(
        (height, len(frame_averages), 3),
        np.uint8
    )

    for i, frame_color in enumerate(frame_averages):
        outimg[:, i, 0] = frame_color.red
        outimg[:, i, 1] = frame_color.green
        outimg[:, i, 2] = frame_color.blue

    outimg = cv2.resize(outimg, (width, height))
    cv2.imwrite("spectrum.png", outimg)
    cv2.imwrite("spectrum_vignette.png", gradient(outimg, 0.4))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "USAGE:", sys.argv[0], "[video_file]"
        sys.exit(1)

    main(sys.argv[1])
