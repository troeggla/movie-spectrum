import cv2
import numpy as np

from argparse import ArgumentParser
from collections import namedtuple
from credits_detection import train_svm, train_cnn
from math import sin, pi
from pickle import load

Color = namedtuple("Color", ["blue", "green", "red"])
Red = Color(red=255, green=0, blue=0)


def model_selector(modelname=None):
    if modelname == "svm":
        is_credit = train_svm.setup_model()
    elif modelname == "cnn":
        is_credit = train_cnn.setup_model()
    else:
        def is_credit(_):
            return False

    return is_credit


def gradient(img, length=0.25):
    rows, cols = img.shape[:2]
    output = np.copy(img)

    def grad_function(x):
        y = sin(pi/2 * x/(rows * length))
        return y if x <= rows * length else 1.0

    grad_function = np.vectorize(grad_function)

    vec = grad_function(np.arange(0, rows)).reshape(-1, 1)
    gradient = vec * vec[::-1]

    return output * gradient[..., np.newaxis]


def mean_color(frame):
    def int_mean(arr):
        return int(round(np.mean(arr)))

    return Color(
        int_mean(frame[:, :, 0]),
        int_mean(frame[:, :, 1]),
        int_mean(frame[:, :, 2])
    )


def get_frames(infile):
    cap = cv2.VideoCapture(infile)

    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        yield frame

    cap.release()


def write_to_files(frame_averages, dimensions, outname="spectrum"):
    width, height = dimensions

    frame_averages = np.array(frame_averages, dtype=np.uint8)
    outimg = np.tile(frame_averages, (height, 1, 1))

    outimg = cv2.resize(outimg, (width, height))
    cv2.imwrite(
        outname + ".png",
        outimg,
        (cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 9)
    )

    with_gradient = gradient(outimg, 0.4)
    cv2.imwrite(
        outname + "_vignette.png",
        with_gradient,
        (cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 9)
    )


def main(infile, dimensions=(8000, 2000), credit_detector=model_selector()):
    frames_processed, frames_dropped = 0, 0
    frame_averages = []

    for frame in get_frames(infile):
        frames_processed += 1

        if credit_detector(frame):
            frames_dropped += 1
        else:
            frame_averages.append(mean_color(frame))

        if frames_processed % 1000 == 0:
            print "Processed %d frames (%d dropped)" % (
                frames_processed, frames_dropped
            )

    write_to_files(frame_averages, dimensions)

    print "\nDONE\n"
    print "Processed %d frames, dropped %d (%.2f%%)" % (
        frames_processed,
        frames_dropped,
        frames_dropped / float(frames_processed) * 100
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Generate a colour spectrum image from a video file by
        taking the average colour of each frame an adding it as a stripe of
        that coulour to the output image."""
    )

    parser.add_argument(
        "-r", "--remove-credits",
        choices=["svm", "cnn"],
        help="""Remove credits from generated image. This argument requires a
        string parameter specifying the model to use. Either 'svm' or 'cnn' are
        valid options."""
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
        credit_detector=model_selector(args.remove_credits)
    )
