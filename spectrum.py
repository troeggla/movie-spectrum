import cv2
import numpy as np

from argparse import ArgumentParser
from collections import namedtuple
from math import sin, pi
from pickle import load

from credits_detection.train_model import process_frame

Color = namedtuple("Color", ["red", "green", "blue"])
model = load(open("credits_detection/model.p", "r"))


def is_credit(frame):
    frame = process_frame(frame)

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


def main(infile, dimensions=(8000, 2000), remove_credits=True):
    cap = cv2.VideoCapture(infile)
    frames_processed, frames_dropped = 0, 0
    frame_averages = []

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        frames_processed += 1

        if remove_credits and is_credit(frame):
            frames_dropped += 1
        else:
            frame_average = mean_color(frame)
            frame_averages.append(frame_average)

        if frames_processed % 1000 == 0:
            print "Processed", frames_processed, "frames", frame_average

            if remove_credits:
                print "Dropped", frames_dropped, "frames"

    cap.release()

    width, height = dimensions
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

    cv2.imwrite("spectrum.png", outimg, (cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 9))
    cv2.imwrite(
        "spectrum_vignette.png",
        with_gradient,
        (cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 9)
    )

    cv2.destroyAllWindows()

    print "\nDONE\n"

    if remove_credits:
        print "Processed %d frames, dropped %d (%.2f%%)" % (
            frames_processed,
            frames_dropped,
            frames_dropped / float(frames_processed) * 100
        )
    else:
        print "Processed %d frames" % (frames_processed)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Generate a colour spectrum image from a video file by
        taking the average colour of each frame an adding it as a stripe of
        that coulour to the output image."""
    )

    parser.add_argument(
        "-r", "--remove-credits",
        action="store_true", default=False,
        help="Remove credits from generated image"
    )

    def dimensions(dim):
        dim = dim.split("x")[:2]

        if len(dim) != 2:
            raise ValueError()

        return tuple(map(int, dim))

    parser.add_argument(
        "-d", "--dimensions",
        default="8000x2000", type=dimensions,
        help="Generate an image with custom dimensions (default 8000x2000)"
    )

    parser.add_argument(
        "video_file", type=str,
        help="The video file to process"
    )

    args = parser.parse_args()

    main(
        args.video_file,
        dimensions=args.dimensions,
        remove_credits=args.remove_credits
    )
