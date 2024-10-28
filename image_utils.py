import numpy as np
import cv2
import math

import geometry


class Camera:
    def __init__(self, width, height, focal_length=None):
        if focal_length is None:
            self.focal_length = min(width, height)
        else:
            self.focal_length = focal_length
        self.center_x = width / 2
        self.center_y = height / 2
        self.width = int(width)
        self.height = int(height)

    def calculate_camera_intrinsics_matrix(self):
        # standard camera intrinsics matrix to 3d project points to image
        return np.array(
            [
                [self.focal_length, 0, self.center_x],
                [0, self.focal_length, self.center_y],
                [0, 0, 1],
            ]
        )

    def project_points(self, extrinsics, points_3d_in):
        # project 3d points using given extrinsics
        # shape of points_3d_in can be anything as long as size of last dim is 3
        points_3d_cam_world = extrinsics(points_3d_in)
        assert points_3d_cam_world.shape[-1] == 3
        original_shape = points_3d_cam_world.shape
        points_3d_cam_world = np.reshape(points_3d_cam_world, [-1, 3])

        K = self.calculate_camera_intrinsics_matrix()
        points_2d = points_3d_cam_world @ np.transpose(K)
        points_2d = points_2d[:, 0:2] / points_2d[:, 2:3]

        points_2d_out = np.reshape(points_2d, list(original_shape[:-1]) + [2])
        return points_2d_out

    def calculate_delta_translation(self, x_px_shift, y_px_shift, z):
        # how much delta x and delta y need to be to shift
        # image by x_px_shift and y_px_shift
        return np.array(
            [x_px_shift * z / self.focal_length, y_px_shift * z / self.focal_length, 0]
        )


def video_generator(video_path, max_frames=100000):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened() and i < max_frames:
        ret, frame = cap.read()
        if ret:
            yield frame
            i += 1
        else:
            cap.release()
            return


def find_lines_in_img(img, lattice_size=19, debug=False):
    # given an image find the lines in the image
    # use search parameters to eliminate lines that aren't part of a grid pattern

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[0:2]
    blur_kernel_size = int(min(h, w) // 400)
    blur_kernel_size += blur_kernel_size % 2 - 1
    threshold, thresh_im = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    canny_thresh_high = int(min(threshold * 2, 200))
    canny_thresh_low = int(threshold / 2)
    canny_aperture_size = 3
    minLineLength = min(h, w) // (lattice_size)
    maxLineGap = min(h, w) // (lattice_size)
    if debug:
        print("blur kernel size", blur_kernel_size)
        print("canny thresh high", canny_thresh_high)
        print("canny thresh low", canny_thresh_low)
        print("canny aperture size", canny_aperture_size)
        print("hough min line length", minLineLength)
        print("hough max line gap", maxLineGap)
    if blur_kernel_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(
        gray, canny_thresh_low, canny_thresh_high, apertureSize=canny_aperture_size
    )

    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        0.5 * np.pi / 180,  # Angle resolution in radians
        threshold=minLineLength * 3,  # Min number of votes for valid line
        minLineLength=minLineLength,  # Min allowed length of line
        maxLineGap=maxLineGap,  # Max allowed gap between line for joining them
    )
    lines = np.squeeze(np.array(lines), 1)
    lines = np.array(lines)
    assert lines.ndim == 2

    return geometry.Lines2D(lines)
