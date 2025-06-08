import cv2
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose3D:
    def __init__(
        self,
        translation,
        x_rot=None,
        y_rot=None,
        z_rot=None,
        rotation=None,
        degrees=True,
    ):
        # exactly one of rotation matrix or (z,y,x) euler angles must be provided
        if rotation is None:
            assert z_rot is not None
            assert y_rot is not None
            assert x_rot is not None
            r = R.from_euler("ZYX", [z_rot, y_rot, x_rot], degrees=degrees)
            rotation = R.as_matrix(r)
        else:
            assert z_rot is None
            assert y_rot is None
            assert x_rot is None
            assert np.allclose(np.matmul(rotation, np.transpose(rotation)), np.eye(3))
            assert np.allclose(np.matmul(np.transpose(rotation), rotation), np.eye(3))

        assert rotation.shape == (3, 3)
        self.rotation = rotation
        assert translation.shape == (3,)
        self.translation = translation

    def __str__(self):
        parameters_string = [
            f"\tz_rot: {self.z_rot_degrees} degrees",
            f"\ty_rot: {self.y_rot_degrees} degrees",
            f"\tz_rot: {self.x_rot_degrees} degrees",
            f"\tx: {self.x}",
            f"\ty: {self.y}",
            f"\tz: {self.z}",
        ]
        return "\n".join(parameters_string)

    def __call__(self, points_3d_in):
        # transform 3d points by pose and preserve shape
        # points_3d_in can be any shape but with length 3 in last dim
        assert points_3d_in.shape[-1] == 3

        original_shape = points_3d_in.shape
        points_3d_in = np.reshape(points_3d_in, [-1, 3])

        points_3d_out = np.matmul(points_3d_in, np.transpose(self.rotation))
        points_3d_out += self.translation

        return np.reshape(points_3d_out, original_shape)

    @property
    def x_rot_degrees(self):
        # return x euler angle in degrees
        r = R.from_matrix(self.rotation)
        z_rot, y_rot, x_rot = r.as_euler("ZYX")
        return x_rot * 180 / math.pi

    @property
    def y_rot_degrees(self):
        # return y euler angle in degrees
        r = R.from_matrix(self.rotation)
        z_rot, y_rot, x_rot = r.as_euler("ZYX")
        return y_rot * 180 / math.pi

    @property
    def z_rot_degrees(self):
        # return z euler angle in degrees
        r = R.from_matrix(self.rotation)
        z_rot, y_rot, x_rot = r.as_euler("ZYX")
        return z_rot * 180 / math.pi

    @property
    def x(self):
        # return translation x component in meters
        return self.translation[0].item()

    @property
    def y(self):
        # return translation y component in meters
        return self.translation[1].item()

    @property
    def z(self):
        # return translation z component in meters
        return self.translation[2].item()


class Lines2D:
    def __init__(self, lines):
        # lines is a Nx4 np array consisting of 2 line segment endpoints
        self.x1 = lines[:, 0:1]
        self.x2 = lines[:, 2:3]
        self.y1 = lines[:, 1:2]
        self.y2 = lines[:, 3:4]

    def __len__(self):
        return self.x1.shape[0]

    @staticmethod
    def combine(obj1, obj2):
        # combine two Lines2D objects
        return Lines2D(np.concatenate([obj1.lines, obj2.lines]))

    @staticmethod
    def get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return (float("inf"), float("inf"))
        return (x / z, y / z)

    @property
    def lines(self):
        return np.concatenate([self.x1, self.y1, self.x2, self.y2], axis=1)

    @property
    def pt1(self):
        # first point in the line segment
        return np.concatenate([self.x1, self.y1], axis=1)

    @property
    def pt2(self):
        # second point in the line segment
        return np.concatenate([self.x2, self.y2], axis=1)

    @property
    def line_angle(self):
        # angle of the line relative to the standard image xy axes
        # result is in radians
        angle = np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
        return angle % math.pi

    @property
    def hough_theta(self):
        # angle of the line orthogonal to the line going through origin
        # this is different from line_angle by 90 degrees
        # result is in radians
        theta = np.arctan2(-self.x1 + self.x2, self.y1 - self.y2)
        theta = theta % math.pi  # make sure angle is in first 2 quadrants
        return theta

    @property
    def hough_r(self):
        # distance between line and origin
        r1 = self.x1 * np.cos(self.hough_theta) + self.y1 * np.sin(self.hough_theta)
        r2 = self.x2 * np.cos(self.hough_theta) + self.y2 * np.sin(self.hough_theta)
        assert np.allclose(r1, r2)
        return r1

    @property
    def line_lengths(self):
        # length of line segment is distance between two points
        return np.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)

    def get_indices(self, indices):
        # select a subset of the lines by indices
        # output is a new Lines2D object
        if indices.shape == (self.x1.shape[0], 1):
            indices = np.reshape(indices, [-1])
        assert indices.shape == (self.x1.shape[0],)
        return Lines2D(self.lines[indices, :])

    def adjust_reference_frame(self, transform_matrix, translation):
        # transform the lines from one set of axes to another set
        # this is a 2D rotation and 2D translation
        # output is a new set of Lines2D object
        transformed_pts1 = self.pt1 @ np.transpose(transform_matrix) + translation
        transformed_pts2 = self.pt2 @ np.transpose(transform_matrix) + translation
        return Lines2D(np.concatenate([transformed_pts1, transformed_pts2], axis=1))

    def build_line_matching_image(self, img_width, img_height, thickness=None):
        # create a new image for visualizing the lines
        min_img_dim = min(img_width, img_height)
        assert thickness > 0
        img = np.zeros((img_height, img_width), dtype=float)
        img = self.overlay_lines(img, color=(1, 1, 1), thickness=thickness)
        return DistributionImage2D(img)

    def filter_lines_by_line_angle(self, z_rot, angle_threshold):
        valid_lines = np.isclose(
            self.line_angle % math.pi, z_rot % math.pi, atol=angle_threshold
        )
        return self.get_indices(valid_lines)

    def filter_parallel_lines_by_grid_size(
        self, grid_size_distance, grid_center_hough_r, tolerance_frac=0.1
    ):
        is_potentially_lattice_line = np.isclose(
            (self.hough_r - grid_center_hough_r) % grid_size_distance,
            0,
            atol=grid_size_distance * tolerance_frac,
        )
        return self.get_indices(is_potentially_lattice_line)

    def overlay_lines(self, img_overlay, color=(0, 0, 0), thickness=2, **kwargs):
        # given an image draw lines on it and return the annotated image
        assert thickness > 0
        debug_img = np.copy(img_overlay)

        for pt1, pt2 in zip(self.pt1, self.pt2):
            cv2.line(
                debug_img,
                pt1.astype(int),
                pt2.astype(int),
                color=color,
                thickness=thickness,
                **kwargs,
            )

        return debug_img


class Distribution1D:
    def __init__(self, bin_centers, distribution_values, is_symmetric=False):
        # input to this is basically the output of a np.hist
        # do not modify the contents of this class as it caches the fft

        assert len(bin_centers) == len(distribution_values)
        self.bin_centers = bin_centers
        self.distribution_values = distribution_values
        self.is_symmetric = is_symmetric
        self.fft = None

    def calculate_fft(self):
        # use cached fft if possible
        if self.fft is None:
            self.fft = np.fft.fft(self.distribution_values)
        return self.fft

    def __len__(self):
        return len(self.distribution_values)

    @staticmethod
    def calculate_correlation(distribution1, distribution2):
        # calculate circular correlation by using fft
        # make sure there is enough padding to not overlap
        fft1 = distribution1.calculate_fft()
        fft2 = distribution2.calculate_fft()
        if distribution1.is_symmetric or distribution2.is_symmetric:
            correlation_scores = np.real(np.fft.ifft(fft1 * fft2))
        else:
            correlation_scores = np.real(np.fft.ifft(fft1 * fft2[::-1]))
        return correlation_scores

    @staticmethod
    def triangle_filter(filter_length, triangle_center, triangle_width):
        # generate an isoceles triangle distribution/filter
        # triangle center is the index of where the triangle should be centered
        # triangle width is the size of the base of triangle

        triangle_base_left = int((triangle_width - 1) // 2)
        triangle_base_right = int(triangle_width - triangle_base_left - 1)
        assert (
            triangle_center + triangle_base_right < filter_length
            or triangle_center - triangle_base_left >= 0
        )
        left_indices = np.linspace(
            -triangle_base_left + triangle_center,
            triangle_center,
            num=triangle_base_left + 1,
            endpoint=True,
        )
        right_indices = np.linspace(
            triangle_center,
            triangle_center + triangle_base_right,
            num=triangle_base_right + 1,
            endpoint=True,
        )
        left_indices = (left_indices % filter_length).astype(int)
        right_indices = (right_indices % filter_length).astype(int)
        triangle_filter = np.zeros([filter_length], dtype=float)
        left_vals = np.linspace(start=0, stop=1, num=len(left_indices), endpoint=True)
        right_vals = np.linspace(start=1, stop=0, num=len(right_indices), endpoint=True)
        triangle_filter[left_indices] = left_vals
        triangle_filter[right_indices] = right_vals
        return triangle_filter

    def display(self, bin_centers=True, **kwargs):
        # debug function to show display the distribution using plt
        if bin_centers:
            plt.plot(self.bin_centers, self.distribution_values, **kwargs)
        else:
            plt.plot(self.distribution_values, **kwargs)


class DistributionImage2D:
    def __init__(self, img):
        assert img.ndim == 2
        self.img = img
        self.fft2 = None

    def resize_img(self, scale):
        # scale is ratio to increase or decrease by
        resized_img = cv2.resize(
            self.img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        return DistributionImage2D(resized_img)

    def calculate_padded_fft(self):
        # if there is a cached fft use that
        # else calculate fft
        # padding is so that there is no overlap in circular fft
        if self.fft2 is None:
            padding_h = self.img.shape[0]
            padding_w = self.img.shape[1]
            padded_img = np.pad(
                self.img, ((padding_h, padding_h), (padding_w, padding_w))
            )
            self.fft2 = np.fft.fft2(padded_img)
        return self.fft2

    @staticmethod
    def match_images(img1, img2):
        # correlation of two images
        # output is the x and y location and correlation value of best correlation
        assert img1.img.shape == img2.img.shape
        original_shape = img1.img.shape

        fft1 = img1.calculate_padded_fft()
        fft2 = img2.calculate_padded_fft()
        normalization_factor = math.sqrt(
            np.sum(np.abs(img1.img)) * np.sum(np.abs(img2.img))
        )
        match_scores = (
            np.real(np.fft.ifft2(fft1 * np.flip(fft2, (0, 1)))) / normalization_factor
        )

        best_match = np.amax(match_scores)

        y_px_shift, x_px_shift = np.unravel_index(
            np.argmax(match_scores), match_scores.shape
        )

        if x_px_shift > original_shape[1]:
            x_px_shift -= fft1.shape[1]
            assert x_px_shift > -original_shape[1]
        if y_px_shift > original_shape[0]:
            y_px_shift -= fft1.shape[0]
            assert y_px_shift > -original_shape[0]
        return (x_px_shift.item(), y_px_shift.item()), best_match.item()
