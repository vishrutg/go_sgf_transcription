import cv2
import itertools

import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.signal

import geometry
import image_utils


class Lattice2D:
    def __init__(
        self,
        img,
        lines,
        camera,
        are_original_lines=True,
        lattice_size=19,
        lattice_grid_width_mm=22,
        lattice_grid_height_mm=23.7,
        lattice_line_diameter_mm=1,
        lattice_width_mm=424.4,
        lattice_height_mm=454.5,
        lattice_stone_diameter_mm=22.5,
    ):
        # vertical and horizontal lines are objects of geometry.Lines2D
        # assumed to be square lattice of num_lattice_lines x num_lattice_lines

        # step 1 estimate z_rot and separate into vertical and horizontal lines
        self.z_rot = None
        self.z_rot_matching_score = None
        self.vertical_lines = None
        self.horizontal_lines = None
        # step 2 estimate grid size and center line hough r
        self.vertical_lines_grid_size_px = None
        self.horizontal_lines_grid_size_px = None
        self.vertical_center_line_hough_r = None
        self.horizontal_center_line_hough_r = None
        # step 3 estimate center lattice point and directions
        self.center_lattice_pt_px = None
        self.lattice_vertical_step = None
        self.lattice_horizontal_step = None
        # step 4 estimate vertical_line_tilt and horizontal_line_tilt and update grid size
        self.vertical_line_tilt = None
        self.horizontal_line_tilt = None
        self.x_rot = None
        self.y_rot = None
        self.z = None
        # estimate full camera extrinsics
        self.x = None
        self.y = None
        self.initial_camera_extrinsics = None
        self.initial_camera_extrinsics_match_score = None

        self.num_lattice_lines = lattice_size
        self.img = img
        self.lines = lines
        self.camera = camera
        self.are_original_lines = are_original_lines

        assert lattice_size % 2 == 1
        assert lattice_grid_height_mm > lattice_grid_width_mm

        self.lattice_grid_width = lattice_grid_width_mm / 1000
        self.lattice_grid_height = lattice_grid_height_mm / 1000
        self.lattice_line_diameter = lattice_line_diameter_mm / 1000
        self.lattice_width = lattice_width_mm / 1000
        self.lattice_height = lattice_height_mm / 1000
        self.lattice_stone_diameter = lattice_stone_diameter_mm / 1000
        self.lattice_3d = self.generate_lattice_3d()
        self.corners_3d = self.generate_corners_3d()

    @staticmethod
    def from_initial_camera_extrinsics(camera, initial_camera_extrinsics, **kwargs):
        lattice = Lattice2D(img=None, lines=None, camera=camera, **kwargs)
        lattice.x_rot = initial_camera_extrinsics.x_rot
        lattice.y_rot = initial_camera_extrinsics.y_rot
        lattice.z_rot = initial_camera_extrinsics.z_rot
        lattice.x = initial_camera_extrinsics.x
        lattice.y = initial_camera_extrinsics.y
        lattice.z = initial_camera_extrinsics.z
        lattice.initial_camera_extrinsics = initial_camera_extrinsics
        return lattice

    def generate_lattice_3d(
        self,
    ):
        # array of playable points on the board
        # output is np array of shape (self.num_lattice_lines, self.num_lattice_lines, 3)
        idx = (self.num_lattice_lines - 1) // 2
        horizontal_coords = np.arange(-idx, idx + 1) * (self.lattice_grid_width)
        vertical_coords = np.arange(-idx, idx + 1) * (self.lattice_grid_height)
        points_3d = [[[x, y, 0] for y in vertical_coords] for x in horizontal_coords]
        points_3d = np.array(points_3d)
        assert points_3d.shape == (self.num_lattice_lines, self.num_lattice_lines, 3)
        return points_3d

    def generate_corners_3d(
        self,
    ):
        # array of corner points on the board
        # output is np array of shape (2, 2, 3)
        idx = (self.num_lattice_lines - 1) // 2
        horizontal_coords = [-self.lattice_width / 2, self.lattice_width / 2]
        vertical_coords = [-self.lattice_height / 2, self.lattice_height / 2]
        points_3d = [[[x, y, 0] for y in vertical_coords] for x in horizontal_coords]
        points_3d = np.array(points_3d)
        assert points_3d.shape == (2, 2, 3)
        return points_3d

    def generate_lattice_lines(self, lattice_points_2d):
        # input is projections of the lattice 3d points to 2d
        # with shape (self.num_lattice_lines, self.num_lattice_lines, 2)
        # output is a set of lines in Lines2D object
        assert lattice_points_2d.shape == (
            self.num_lattice_lines,
            self.num_lattice_lines,
            2,
        )
        lattice_lines = []
        for i in range(self.num_lattice_lines):
            line1_start = lattice_points_2d[i, 0]
            line1_end = lattice_points_2d[i, -1]
            lattice_lines.append(np.concatenate([line1_start, line1_end]))
            line2_start = lattice_points_2d[0, i]
            line2_end = lattice_points_2d[-1, i]
            lattice_lines.append(np.concatenate([line2_start, line2_end]))
        assert len(lattice_lines) == 2 * self.num_lattice_lines
        lattice_lines = np.array(lattice_lines)
        return geometry.Lines2D(lattice_lines)

    def overlay_grid_with_camera_extrinsics(
        self, camera, camera_extrinsics, img, difficulty=0.5
    ):
        lattice_points_2d = camera.project_points(camera_extrinsics, self.lattice_3d)
        lattice_lines = self.generate_lattice_lines(lattice_points_2d)

        thickness = (
            self.lattice_line_diameter
            * camera.focal_length
            / camera_extrinsics.z
            / difficulty
        )
        thickness = int(math.ceil(thickness))
        assert thickness > 0

        debug_img = lattice_lines.overlay_lines(
            np.copy(img), color=(0, 0, 0), thickness=thickness
        )
        return debug_img

    def filtered_detected_lines_img_by_camera_extrinsics(
        self, camera, camera_extrinsics, detected_lines, difficulty=0.5
    ):
        angle_threshold = 5 * math.pi / 180
        z_rot = camera_extrinsics.z_rot_degrees / 180 * math.pi

        height_lattice_lines = detected_lines.filter_lines_by_line_angle(
            z_rot, angle_threshold
        )
        width_lattice_lines = detected_lines.filter_lines_by_line_angle(
            z_rot + math.pi / 2, angle_threshold
        )
        lattice_lines = geometry.Lines2D.combine(
            height_lattice_lines, width_lattice_lines
        )

        thickness = (
            self.lattice_line_diameter
            * camera.focal_length
            / camera_extrinsics.z
            / difficulty
        )
        thickness = int(math.ceil(thickness))
        assert thickness > 0

        filtered_det_lines_img = lattice_lines.build_line_matching_image(
            img_width=camera.width, img_height=camera.height, thickness=thickness
        )
        return filtered_det_lines_img

    def camera_extrinsics_match_score(
        self, camera, camera_extrinsics, filtered_det_lines_img, difficulty=0.5
    ):
        thickness = (
            self.lattice_line_diameter
            * camera.focal_length
            / camera_extrinsics.z
            / difficulty
        )
        thickness = int(math.ceil(thickness))
        assert thickness > 0

        lattice_points_2d = camera.project_points(camera_extrinsics, self.lattice_3d)
        ref_lattice_lines = self.generate_lattice_lines(lattice_points_2d)
        ref_lines_img = ref_lattice_lines.build_line_matching_image(
            img_width=camera.width, img_height=camera.height, thickness=thickness
        )

        match_score = np.mean(ref_lines_img.img * filtered_det_lines_img.img)
        return match_score

    def adjust_reference_frame(self, z_rot=0, translation=np.array([0, 0])):
        transform_matrix = np.array(
            [[np.cos(z_rot), -np.sin(z_rot)], [np.sin(z_rot), np.cos(z_rot)]]
        )
        transformed_vertical_lines = self.vertical_lines.adjust_reference_frame(
            transform_matrix=transform_matrix, translation=translation
        )
        transformed_horizontal_lines = self.horizontal_lines.adjust_reference_frame(
            transform_matrix=transform_matrix, translation=translation
        )
        transformed_lines = self.lines.adjust_reference_frame(
            transform_matrix=transform_matrix, translation=translation
        )
        # todo copy all other attributes
        transformed_lattice = Lattice2D(
            img=None,
            lines=transformed_lines,
            camera=self.camera,
            are_original_lines=False,
        )
        transformed_lattice.z_rot = z_rot + self.z_rot
        transformed_lattice.vertical_lines = transformed_vertical_lines
        transformed_lattice.horizontal_lines = transformed_horizontal_lines
        transformed_lattice.z_rot_matching_score = self.z_rot_matching_score
        transformed_lattice.vertical_lines_grid_size_px = (
            self.vertical_lines_grid_size_px
        )
        transformed_lattice.horizontal_lines_grid_size_px = (
            self.horizontal_lines_grid_size_px
        )
        transformed_lattice.vertical_center_line_hough_r = None
        transformed_lattice.horizontal_center_line_hough_r = None

        return transformed_lattice

    @staticmethod
    def calculate_hough_r_distribution(lines, max_r, granularity):
        weight = lines.line_lengths
        weight /= np.sum(weight)
        # the range is -max_r to +max_r for padding needed in circular conv
        r_hist, bin_edges = np.histogram(
            lines.hough_r,
            bins=int(2 * max_r * granularity + 1),
            range=(-max_r, max_r),
            weights=weight,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        r_distribution = geometry.Distribution1D(
            bin_centers, r_hist, is_symmetric=False
        )
        return r_distribution

    @staticmethod
    def filter_lines_by_z_rot(lines, z_rot, angle_threshold):
        theta = lines.line_angle % math.pi
        valid_lines = np.isclose(theta, z_rot % math.pi, atol=angle_threshold)
        return lines.get_indices(valid_lines)

    @staticmethod
    def filter_lines_by_grid_size(
        parallel_lines, grid_size_px, grid_center_hough_r, tolerance_frac=0.1
    ):
        is_potentially_lattice_line = np.isclose(
            (parallel_lines.hough_r - grid_center_hough_r) % grid_size_px,
            0,
            atol=grid_size_px * tolerance_frac,
        )
        return parallel_lines.get_indices(is_potentially_lattice_line)

    def estimate_lattice_z_rot(self, filter_granularity, filter_size_degrees):
        filter_size = filter_size_degrees * math.pi / 180

        hough_theta = self.lines.hough_theta
        weight = self.lines.line_lengths
        weight /= np.sum(weight)

        # although hough theta is going to be between 0 to math.pi
        # hough_theta_hist range is -math.pi to math.pi
        # this is because of circular conv (fft) needs to be padded

        hough_theta_hist, bin_edges = np.histogram(
            hough_theta,
            bins=filter_granularity,
            range=(-math.pi, math.pi),
            weights=weight,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hough_theta_distribution = geometry.Distribution1D(
            bin_centers=bin_centers, distribution_values=hough_theta_hist
        )

        # generate a filter that matches at 0, math.pi/2, math.pi, 3*math.pi/2
        match_filter = np.zeros_like(hough_theta_hist)
        filter_bins = int(filter_size / (2 * math.pi) * filter_granularity)
        for i in range(4):
            match_filter += geometry.Distribution1D.triangle_filter(
                len(hough_theta_hist), (i * len(hough_theta_hist)) // 4, filter_bins
            )
        match_filter = geometry.Distribution1D(
            bin_centers=bin_centers - bin_centers[0],
            distribution_values=match_filter,
            is_symmetric=True,
        )

        correlation_scores = geometry.Distribution1D.calculate_correlation(
            hough_theta_distribution, match_filter
        )
        best_hough_theta = bin_centers[np.argmax(correlation_scores)] % math.pi

        valid_lines1 = self.lines.filter_lines_by_line_angle(
            z_rot=best_hough_theta + math.pi / 2, angle_threshold=filter_size / 2
        )
        valid_lines2 = self.lines.filter_lines_by_line_angle(
            z_rot=best_hough_theta, angle_threshold=filter_size / 2
        )

        if math.pi / 4 < best_hough_theta % math.pi < 3 * math.pi / 4:
            # best hough theta is pointing roughly down
            horizontal_lines = valid_lines1
            vertical_lines = valid_lines2
        else:
            horizontal_lines = valid_lines2
            vertical_lines = valid_lines1

        self.vertical_lines = vertical_lines
        self.horizontal_lines = horizontal_lines
        self.z_rot = best_hough_theta % (math.pi / 2)
        self.z_rot_matching_score = np.max(correlation_scores)

    def estimate_lattice_scale_px(
        self,
        grid_size_px_search_values,
        tolerance_frac,
        granularity,
        max_r,
        use_priors=False,
    ):

        def _generate_parallel_line_matching_kernel(filter_length, filter_grid_size):
            tolerance = int(filter_grid_size * tolerance_frac * granularity)
            assert tolerance > 0
            match_filter = np.zeros(filter_length, dtype=float)
            for i in np.linspace(
                -(self.num_lattice_lines - 1) // 2,
                (self.num_lattice_lines - 1) // 2,
                num=self.num_lattice_lines,
                endpoint=True,
            ):
                match_filter += geometry.Distribution1D.triangle_filter(
                    filter_length, i * filter_grid_size * granularity, tolerance
                )
            return geometry.Distribution1D(
                np.arange(filter_length)
                + filter_length // 2 % filter_length
                - filter_length // 2,
                match_filter,
                is_symmetric=True,
            )

        def calculate_correlation_score_matrix(
            r_distribution, grid_size_px_search_values, prior=None
        ):
            filters = [
                _generate_parallel_line_matching_kernel(
                    filter_length=len(r_distribution), filter_grid_size=grid_size
                )
                for grid_size in grid_size_px_search_values
            ]
            correlation_scores = [
                geometry.Distribution1D.calculate_correlation(r_distribution, filter)
                for filter in filters
            ]
            if prior is not None:
                correlation_scores = [
                    correlation_score_i * prior_i
                    for (correlation_score_i, prior_i) in zip(prior, correlation_scores)
                ]
            best_indices = [
                np.argmax(correlation) for correlation in correlation_scores
            ]
            best_score = [
                correlation[idx]
                for idx, correlation in zip(best_indices, correlation_scores)
            ]
            best_r = [r_distribution.bin_centers[idx] for idx in best_indices]
            return np.array(best_score), np.array(best_r)

        vertical_lines_hough_r_distribution = Lattice2D.calculate_hough_r_distribution(
            self.vertical_lines, max_r, granularity
        )
        horizontal_lines_hough_r_distribution = (
            Lattice2D.calculate_hough_r_distribution(
                self.horizontal_lines, max_r, granularity
            )
        )

        if use_priors:
            output = self.calculate_lattice_grid_size_and_center_point_priors(
                hough_r_distribution1=vertical_lines_hough_r_distribution.bin_centers,
                grid_scale_px_distribution1=grid_size_px_search_values,
                hough_r_distribution2=horizontal_lines_hough_r_distribution.bin_centers,
                grid_scale_px_distribution2=grid_size_px_search_values,
            )
            self.vertical_center_line_hough_r_prior = output[0]
            self.horizontal_center_line_hough_r_prior = output[1]
            self.grid_size_consistency_prior = output[2]
        else:
            self.grid_size_consistency_prior = 1

        best_scores1, best_center_vertical_line_hough_r = (
            calculate_correlation_score_matrix(
                vertical_lines_hough_r_distribution,
                grid_size_px_search_values,
                self.vertical_center_line_hough_r_prior if use_priors else None,
            )
        )
        best_scores2, best_center_horizontal_line_hough_r = (
            calculate_correlation_score_matrix(
                horizontal_lines_hough_r_distribution,
                grid_size_px_search_values,
                self.horizontal_center_line_hough_r_prior if use_priors else None,
            )
        )

        grid_size_scores = self.grid_size_consistency_prior * np.outer(
            best_scores1, best_scores2
        )
        best_grid_size1_idx, best_grid_size2_idx = np.unravel_index(
            np.argmax(grid_size_scores), grid_size_scores.shape
        )

        self.vertical_lines_grid_size_px = grid_size_px_search_values[
            best_grid_size1_idx
        ]
        self.horizontal_lines_grid_size_px = grid_size_px_search_values[
            best_grid_size2_idx
        ]
        self.vertical_center_line_hough_r = best_center_vertical_line_hough_r[
            best_grid_size1_idx
        ]
        self.horizontal_center_line_hough_r = best_center_horizontal_line_hough_r[
            best_grid_size2_idx
        ]

        self.vertical_lines = self.vertical_lines.filter_parallel_lines_by_grid_size(
            self.vertical_lines_grid_size_px,
            self.vertical_center_line_hough_r,
            tolerance_frac=tolerance_frac,
        )
        self.horizontal_lines = (
            self.horizontal_lines.filter_parallel_lines_by_grid_size(
                self.horizontal_lines_grid_size_px,
                self.horizontal_center_line_hough_r,
                tolerance_frac=tolerance_frac,
            )
        )
        best_total_score = (
            grid_size_scores[best_grid_size1_idx, best_grid_size2_idx]
            * best_scores1[best_grid_size1_idx]
            * best_scores2[best_grid_size2_idx]
        )
        self.best_grid_scale_px_matching_score = best_total_score

    def calculate_lattice_grid_size_and_center_point_priors(
        self,
        hough_r_distribution1,
        grid_scale_px_distribution1,
        hough_r_distribution2,
        grid_scale_px_distribution2,
        z_rot=None,
    ):
        # grid_scale_px is list of possible grid sizes in pixels
        # hough_r_distribution is list of possible hough r values for center lattice line
        # hough_r_distribution1,
        # grid_scale_px1,

        # w is img_width
        # h is img_height

        assert self.are_original_lines
        if z_rot is None:
            z_rot = self.z_rot
        w = self.camera.width
        h = self.camera.height

        if z_rot > math.pi / 2:
            output = self.calculate_lattice_grid_size_and_center_point_priors(
                hough_r_distribution2,
                grid_scale_px_distribution2,
                hough_r_distribution1,
                grid_scale_px_distribution1,
                z_rot=z_rot - math.pi / 2,
            )
            return output[1], output[0], output[2]

        assert z_rot < math.pi / 2

        # calculate constraints of largest possible rotated rectangle inside image
        # mat is not a rotation matrix
        mat = np.array([[np.sin(z_rot), np.cos(z_rot)], [np.cos(z_rot), np.sin(z_rot)]])
        if w < h:
            rh, rw = np.linalg.inv(mat) @ np.array([w, h])
            a = (w - rw * np.cos(z_rot)) * np.cos(z_rot)
            b = a * rh / rw
            c = a * np.tan(z_rot)
            # center_r should be in: [-c, -c + rh] x [a, a+rw]
            center_r_dist_params = np.array([[-c, -c + rh], [a, a + rw]])
        else:
            rw, rh = np.linalg.inv(mat) @ np.array([h, w])
            a = (h - rh * np.cos(z_rot)) * np.cos(z_rot)
            b = a * rw / rh
            c = a * np.tan(z_rot)
            # center_r should be in: [a, a+rw] x [-c, -c + rh]
            center_r_dist_params = np.array([[a, a + rw], [-c, -c + rh]])

        def _calculate_prob(center_r, center_r_dist_params, grid_scale):
            center_r_dist_params = np.tile(
                np.expand_dims(center_r_dist_params, 1), [1, len(grid_scale)]
            )
            assert center_r_dist_params.shape == (2, len(grid_scale))

            center_r_lower_bound = (
                center_r_dist_params[0, :]
                + (self.num_lattice_lines - 1) / 2 * grid_scale
            )
            center_r_upper_bound = (
                center_r_dist_params[1, :]
                - (self.num_lattice_lines - 1) / 2 * grid_scale
            )
            center_r_lower_bound = np.tile(
                np.expand_dims(center_r_lower_bound, 1), [1, len(center_r)]
            )
            center_r_upper_bound = np.tile(
                np.expand_dims(center_r_upper_bound, 1), [1, len(center_r)]
            )

            assert center_r_lower_bound.shape == (len(grid_scale), len(center_r))
            assert center_r_upper_bound.shape == (len(grid_scale), len(center_r))

            center_r = np.tile(np.expand_dims(center_r, 0), [len(grid_scale), 1])
            assert center_r.shape == center_r_lower_bound.shape

            is_valid = np.logical_and(
                center_r > center_r_lower_bound, center_r < center_r_upper_bound
            ).astype(float)
            assert is_valid.shape == center_r_lower_bound.shape
            return is_valid

        prob1 = _calculate_prob(
            hough_r_distribution1,
            center_r_dist_params[0, :],
            grid_scale_px_distribution1,
        )
        prob2 = _calculate_prob(
            hough_r_distribution2,
            center_r_dist_params[1, :],
            grid_scale_px_distribution2,
        )

        # calculate grid size consistentcy
        consistency_prior = scipy.stats.norm(
            np.expand_dims(grid_scale_px_distribution1, 1),
            np.expand_dims(grid_scale_px_distribution1, 1) / 10,
        )
        consistency_prior = consistency_prior.pdf(
            np.expand_dims(grid_scale_px_distribution2, 0)
        )
        assert consistency_prior.shape == (
            len(grid_scale_px_distribution1),
            len(grid_scale_px_distribution2),
        )

        return prob1, prob2, consistency_prior

    def estimate_lattice_center_point(self):
        rotation_matrix = np.array(
            [
                [np.cos(self.z_rot), -np.sin(self.z_rot)],
                [np.sin(self.z_rot), np.cos(self.z_rot)],
            ]
        )
        center_pt_hough_r = np.array(
            [self.vertical_center_line_hough_r, self.horizontal_center_line_hough_r]
        )

        self.center_lattice_pt_px = rotation_matrix @ center_pt_hough_r
        self.lattice_vertical_step = (
            rotation_matrix[0, :] * self.vertical_lines_grid_size_px
        )
        self.lattice_horizontal_step = (
            rotation_matrix[1, :] * self.horizontal_lines_grid_size_px
        )

    def estimate_lattice_tilt(
        self,
        max_r,
        tolerance_frac=0.25,
        granularity=4,
    ):
        # rotate lines by -z_rot to center them
        # vertical lines should be approximately x = C * scale for C in [-(num_lattice_lines-1)/2, (num_lattice_lines-1)/2]
        # horizontal lines should be approximately y = C * scale for C in [-(num_lattice_lines-1)/2, (num_lattice_lines-1)/2]
        origin_lattice = self.adjust_reference_frame(
            translation=-self.center_lattice_pt_px
        ).adjust_reference_frame(z_rot=-self.z_rot)
        board_center_to_edge_px = (
            np.array(
                [self.vertical_lines_grid_size_px, self.horizontal_lines_grid_size_px]
            )
            * (self.num_lattice_lines - 1)
            / 2
        )

        # when grid scales  of top left lattice are calculated
        # they are going to be the hough values at the top and left edges of the board
        top_left_lattice = origin_lattice.adjust_reference_frame(
            translation=board_center_to_edge_px
        )
        # when hough grid scales  of bottom right lattice are calculated
        # they are going to be the hough values at the bottom right and edges of the board
        bottom_right_lattice = origin_lattice.adjust_reference_frame(
            z_rot=math.pi, translation=board_center_to_edge_px
        )

        min_scale = int(
            min(self.vertical_lines_grid_size_px, self.horizontal_lines_grid_size_px)
            * 0.8
        )
        max_scale = int(
            max(self.vertical_lines_grid_size_px, self.horizontal_lines_grid_size_px)
            * 1.2
        )
        grid_size_px_search_values = np.linspace(
            min_scale, max_scale, num=int(10 * (max_scale - min_scale))
        )

        kwargs = {
            "grid_size_px_search_values": grid_size_px_search_values,
            "tolerance_frac": tolerance_frac,
            "granularity": granularity,
            "max_r": max_r,
            "use_priors": False,
        }

        top_left_lattice.estimate_lattice_scale_px(**kwargs)
        bottom_right_lattice.estimate_lattice_scale_px(**kwargs)

        self.top_edge_grid_size = top_left_lattice.vertical_lines_grid_size_px
        self.bottom_edge_grid_size = bottom_right_lattice.vertical_lines_grid_size_px
        self.left_edge_grid_size = top_left_lattice.horizontal_lines_grid_size_px
        self.right_edge_grid_size = bottom_right_lattice.horizontal_lines_grid_size_px

        def get_angle_from_scales(
            board_center_to_edge_px, left_side_scale, right_side_scale, focal_length
        ):
            pt1 = np.array([board_center_to_edge_px, right_side_scale])
            pt2 = np.array([-board_center_to_edge_px, left_side_scale])
            pt3 = np.array([0, 0])
            pt4 = np.array([1, 0])
            epipole_distance, _ = geometry.Lines2D.get_intersect(pt1, pt2, pt3, pt4)
            x_inv_z = epipole_distance / focal_length
            angle = np.atan(focal_length / epipole_distance)
            return angle, epipole_distance

        self.vertical_line_tilt, self.vertical_lines_epipole = get_angle_from_scales(
            board_center_to_edge_px[0],
            left_side_scale=self.left_edge_grid_size,
            right_side_scale=self.right_edge_grid_size,
            focal_length=self.camera.focal_length,
        )
        self.horizontal_line_tilt, self.horizontal_lines_epipole = (
            get_angle_from_scales(
                board_center_to_edge_px[1],
                left_side_scale=self.top_edge_grid_size,
                right_side_scale=self.bottom_edge_grid_size,
                focal_length=self.camera.focal_length,
            )
        )

        # update vertical and horizontal line grid size
        self.vertical_lines_grid_size_px = (
            self.top_edge_grid_size + self.bottom_edge_grid_size
        ) / 2
        self.horizontal_lines_grid_size_px = (
            self.left_edge_grid_size + self.right_edge_grid_size
        ) / 2

        if self.vertical_lines_grid_size_px > self.horizontal_lines_grid_size_px:
            # vertical lines are farther away means camera is rotated 90 degrees
            self.z_rot += math.pi / 2
            self.y_rot = self.horizontal_line_tilt
            self.x_rot = self.vertical_line_tilt
            z1 = (
                self.camera.focal_length
                * self.lattice_grid_height
                / self.vertical_lines_grid_size_px
            )
            z2 = (
                self.camera.focal_length
                * self.lattice_grid_width
                / self.horizontal_lines_grid_size_px
            )
            self.z = (z1 + z2) / 2
        else:
            self.x_rot = self.horizontal_line_tilt
            self.y_rot = self.vertical_line_tilt
            z1 = (
                self.camera.focal_length
                * self.lattice_grid_height
                / self.horizontal_lines_grid_size_px
            )
            z2 = (
                self.camera.focal_length
                * self.lattice_grid_width
                / self.vertical_lines_grid_size_px
            )
            self.z = (z1 + z2) / 2

    def estimate_camera_extrinsics(self, line_matching_frac):

        extrinsics = geometry.Pose3D(
            z_rot=self.z_rot,
            x_rot=self.x_rot,
            y_rot=self.y_rot,
            translation=np.array([0.0, 0.0, self.z]),
            degrees=False,
        )
        thickness = int(
            math.ceil(self.horizontal_lines_grid_size_px * line_matching_frac)
        )
        assert thickness > 0

        lattice_points_2d = self.camera.project_points(extrinsics, self.lattice_3d)
        ref_lattice_lines = self.generate_lattice_lines(lattice_points_2d)

        ref_lines_img = ref_lattice_lines.build_line_matching_image(
            img_width=self.camera.width,
            img_height=self.camera.height,
            thickness=thickness,
        )
        filtered_det_lines = geometry.Lines2D.combine(
            self.vertical_lines, self.horizontal_lines
        )
        det_lines_img = filtered_det_lines.build_line_matching_image(
            img_width=self.camera.width,
            img_height=self.camera.height,
            thickness=thickness,
        )

        # from x and y pixel offsets calculate the translation x and y
        match_results = geometry.DistributionImage2D.match_images(
            ref_lines_img, det_lines_img
        )
        (x_px_shift, y_px_shift), self.initial_camera_extrinsics_match_score = (
            match_results
        )

        extrinsics.translation += self.camera.calculate_delta_translation(
            -x_px_shift, -y_px_shift, extrinsics.translation[2]
        )
        self.x = extrinsics.translation[0]
        self.y = extrinsics.translation[1]
        self.initial_camera_extrinsics = extrinsics

    def estimate_initial_parameters(self, debug=False):
        z_rot_filter_granularity = 3600
        z_rot_matching_threshold_degrees = 5
        max_lattice_grid_px_size = (
            min(self.camera.width, self.camera.height) / self.num_lattice_lines
        )
        grid_size_px_search_size = min(100, int(max_lattice_grid_px_size))
        lattice_scale_tolerance_frac = 0.2
        lattice_scale_filter_sampling_rate = 4
        lattice_max_hough_r = max(self.camera.width, self.camera.height) * 2
        debug_image_line_thickness = min(self.camera.width, self.camera.height) // (
            10 * self.num_lattice_lines
        )

        if debug:
            # show the detected lines
            debug_img = np.copy(self.img)
            debug_img = self.lines.overlay_lines(
                debug_img, color=(0, 0, 255), thickness=debug_image_line_thickness
            )
            cv2.imshow("detected lines", debug_img)

        # step 1 estimate z rotation of lattice and separate vertical lines and horizontal lines
        self.estimate_lattice_z_rot(
            filter_granularity=z_rot_filter_granularity,
            filter_size_degrees=z_rot_matching_threshold_degrees,
        )
        if debug:
            # show the detected lines
            print("z_rot:", self.z_rot * 180 / math.pi, "degrees")
            print("z_rot matching score:", self.z_rot_matching_score)
            debug_img = np.copy(self.img)
            debug_img = self.vertical_lines.overlay_lines(
                debug_img, color=(255, 0, 0), thickness=debug_image_line_thickness
            )
            debug_img = self.horizontal_lines.overlay_lines(
                debug_img, color=(0, 255, 0), thickness=debug_image_line_thickness
            )
            cv2.imshow("detected lines filtered by angle", debug_img)

        # step 2 estimate the grid size of the lattice
        grid_size_px_search_values = np.linspace(
            max_lattice_grid_px_size / 2,
            max_lattice_grid_px_size,
            num=grid_size_px_search_size,
        )
        self.estimate_lattice_scale_px(
            grid_size_px_search_values,
            tolerance_frac=lattice_scale_tolerance_frac,
            granularity=lattice_scale_filter_sampling_rate,
            max_r=lattice_max_hough_r,
            use_priors=True,
        )
        if debug:
            print(
                "vertical lines grid size", self.vertical_lines_grid_size_px, "pixels"
            )
            print(
                "horizontal lines grid size",
                self.horizontal_lines_grid_size_px,
                "pixels",
            )
            print("vertical lines hough r", self.vertical_center_line_hough_r, "pixels")
            print(
                "horizontal lines hough r",
                self.horizontal_center_line_hough_r,
                "pixels",
            )
            print("scale matching score", self.best_grid_scale_px_matching_score)
            debug_img = np.copy(self.img)
            debug_img = self.vertical_lines.overlay_lines(
                debug_img, color=(255, 0, 0), thickness=debug_image_line_thickness
            )
            debug_img = self.horizontal_lines.overlay_lines(
                debug_img, color=(0, 255, 0), thickness=debug_image_line_thickness
            )
            cv2.imshow("detected lines filtered by angle and scale", debug_img)

        self.estimate_lattice_center_point()
        if debug:
            debug_img = np.copy(self.img)
            debug_img = self.vertical_lines.overlay_lines(
                debug_img, color=(255, 0, 0), thickness=debug_image_line_thickness
            )
            debug_img = self.horizontal_lines.overlay_lines(
                debug_img, color=(0, 255, 0), thickness=debug_image_line_thickness
            )
            debug_img = cv2.circle(
                debug_img,
                self.center_lattice_pt_px.astype(int),
                radius=debug_image_line_thickness,
                color=(0, 0, 255),
                thickness=-1,
            )
            for i in np.linspace(
                -(self.num_lattice_lines - 1) // 2,
                (self.num_lattice_lines - 1) // 2,
                num=self.num_lattice_lines,
                endpoint=True,
            ):
                debug_img = cv2.circle(
                    debug_img,
                    (self.center_lattice_pt_px + i * self.lattice_vertical_step).astype(
                        int
                    ),
                    radius=debug_image_line_thickness,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                debug_img = cv2.circle(
                    debug_img,
                    (
                        self.center_lattice_pt_px + i * self.lattice_horizontal_step
                    ).astype(int),
                    radius=debug_image_line_thickness,
                    color=(0, 0, 255),
                    thickness=-1,
                )

            cv2.imshow("detected lines with estimate axis", debug_img)

        self.estimate_lattice_tilt(
            max_r=lattice_max_hough_r,
            tolerance_frac=lattice_scale_tolerance_frac,
            granularity=lattice_scale_filter_sampling_rate // 2,
        )
        if debug:
            print(
                "top edge vertical lines grid size", self.top_edge_grid_size, "pixels"
            )
            print(
                "bottom edge vertical lines grid size",
                self.bottom_edge_grid_size,
                "pixels",
            )
            print(
                "left edge horizontal lines grid size",
                self.left_edge_grid_size,
                "pixels",
            )
            print(
                "right edge horizontal lines grid size",
                self.right_edge_grid_size,
                "pixels",
            )
            print("vertical lines epipole", self.vertical_lines_epipole, "pixels")
            print("horizontal lines epipole", self.horizontal_lines_epipole, "pixels")
            print(
                "vertical line tilt", self.vertical_line_tilt * 180 / math.pi, "degrees"
            )
            print(
                "horizontal line tilt",
                self.horizontal_line_tilt * 180 / math.pi,
                "degrees",
            )
            print("x_rot", self.x_rot * 180 / math.pi, "degrees")
            print("y_rot", self.y_rot * 180 / math.pi, "degrees")

        self.estimate_camera_extrinsics(line_matching_frac=lattice_scale_tolerance_frac)
        if debug:
            print(
                "camera extrinsics match score",
                self.initial_camera_extrinsics_match_score,
            )
            print("estimated camera extrinsics:\n", str(self.initial_camera_extrinsics))
            debug_img = np.copy(self.img)
            recalculated_lattice_points_2d = self.camera.project_points(
                self.initial_camera_extrinsics, self.lattice_3d
            )
            recalculated_lines = self.generate_lattice_lines(
                recalculated_lattice_points_2d
            )
            debug_img = recalculated_lines.overlay_lines(
                debug_img, color=(0, 0, 0), thickness=debug_image_line_thickness
            )
            cv2.imshow("extrinsics matching", debug_img)

    def refine_camera_extrinsics_brute_force(
        self,
        camera,
        filtered_det_lines_img,
        num_x_bins,
        num_y_bins,
        num_z_bins,
        num_x_rot_bins,
        num_y_rot_bins,
        num_z_rot_bins,
        z_rot_search_range_degrees=2,
        z_search_range_mm=10,
        x_search_range_mm=10,
        y_search_range_mm=10,
        debug=False,
    ):

        z_rot_search_range = z_rot_search_range_degrees * math.pi / 180

        x_rot_vals = np.linspace(
            0,
            self.initial_camera_extrinsics.x_rot_degrees,
            num_x_rot_bins,
            endpoint=True,
        )
        x_rot_vals = x_rot_vals * math.pi / 180
        y_rot_vals = np.linspace(
            0,
            self.initial_camera_extrinsics.y_rot_degrees,
            num_y_rot_bins,
            endpoint=True,
        )
        y_rot_vals = y_rot_vals * math.pi / 180
        z_rot_vals = np.linspace(
            -z_rot_search_range, z_rot_search_range, num_z_rot_bins, endpoint=True
        )
        z_rot_vals += self.initial_camera_extrinsics.z_rot_degrees * math.pi / 180

        x_vals = np.linspace(
            -x_search_range_mm, x_search_range_mm, num_x_bins, endpoint=True
        )
        x_vals = x_vals / 1000 + self.initial_camera_extrinsics.x
        y_vals = np.linspace(
            -y_search_range_mm, y_search_range_mm, num_y_bins, endpoint=True
        )
        y_vals = y_vals / 1000 + self.initial_camera_extrinsics.y
        z_vals = np.linspace(
            -z_search_range_mm, z_search_range_mm, num_z_bins, endpoint=True
        )
        z_vals = z_vals / 1000 + self.initial_camera_extrinsics.z

        search_space_iterator = itertools.product(
            z_rot_vals, y_rot_vals, x_rot_vals, x_vals, y_vals, z_vals
        )

        if debug:
            print("z rot search values:", z_rot_vals * 180 / math.pi)
            print("y rot search values:", y_rot_vals * 180 / math.pi)
            print("x rot search values:", x_rot_vals * 180 / math.pi)
            print("x search values:", x_vals)
            print("y search values:", y_vals)
            print("z search values:", z_vals)

        best_camera_extrinsics = self.initial_camera_extrinsics
        best_match_score = self.camera_extrinsics_match_score(
            camera=camera,
            camera_extrinsics=self.initial_camera_extrinsics,
            filtered_det_lines_img=filtered_det_lines_img,
        )
        for i, (z_rot, y_rot, x_rot, x, y, z) in enumerate(search_space_iterator):
            camera_extrinsics = geometry.Pose3D(
                z_rot=z_rot,
                y_rot=y_rot,
                x_rot=x_rot,
                translation=np.array([x, y, z]),
                degrees=False,
            )
            match_score = self.camera_extrinsics_match_score(
                camera=camera,
                camera_extrinsics=camera_extrinsics,
                filtered_det_lines_img=filtered_det_lines_img,
            )
            if debug:

                def angle_to_string(angle):
                    return str(round(angle * 180 / math.pi))

                def distance_to_string(distance):
                    return str(round(distance, 3))

                print(
                    i,
                    angle_to_string(z_rot),
                    angle_to_string(y_rot),
                    angle_to_string(x_rot),
                    distance_to_string(x),
                    distance_to_string(y),
                    distance_to_string(z),
                    round(match_score, 3),
                )
            if match_score > best_match_score:
                best_match_score = match_score
                best_camera_extrinsics = camera_extrinsics

        self.camera_extrinsics = best_camera_extrinsics
        return best_camera_extrinsics

    def estimate_obstruction_map(
        self, camera, filtered_det_lines_img, dilation_ratio=0.25
    ):
        filtered_det_lines_img = filtered_det_lines_img.img
        lattice_points_2d = camera.project_points(
            self.camera_extrinsics, self.lattice_3d
        )
        assert lattice_points_2d.shape == (
            self.num_lattice_lines,
            self.num_lattice_lines,
            2,
        )
        lattice_points_2d = np.reshape(lattice_points_2d, [-1, 2])
        apparent_stone_size = (
            self.lattice_stone_diameter * camera.focal_length / self.camera_extrinsics.z
        )
        kernel_values = np.linspace(
            -apparent_stone_size / 2,
            apparent_stone_size / 2,
            num=int(math.ceil(apparent_stone_size)),
        )

        obstruction_map = []
        for i, pt in enumerate(lattice_points_2d):
            line_visible = 0.0
            for j in kernel_values:
                for k in kernel_values:
                    test_pt = np.round(pt + np.array([j, k])).astype(int)
                    line_visible_jk = filtered_det_lines_img[test_pt[1], test_pt[0]]
                    if line_visible_jk > line_visible:
                        line_visible = line_visible_jk
            obstruction_map.append(1 - line_visible)
        assert len(obstruction_map) == self.num_lattice_lines**2
        obstruction_map = np.reshape(
            np.array(obstruction_map), [self.num_lattice_lines, self.num_lattice_lines]
        )

        dilation_kernel = np.array(
            [
                [dilation_ratio, 1, dilation_ratio],
                [1, 1, 1],
                [dilation_ratio, 1, dilation_ratio],
            ]
        )
        dilation_kernel /= np.sum(dilation_kernel)
        obstruction_map = scipy.signal.convolve2d(
            obstruction_map, dilation_kernel, mode="same"
        )

        obstruction_map = np.minimum(np.maximum(obstruction_map, 1e-4), 1 - 1e-4)
        return obstruction_map

    def save_parameters(self, path):
        parameters = {
            "num_lattice_lines": self.num_lattice_lines,
            "lattice_grid_width_mm": self.lattice_grid_width * 1000,
            "lattice_grid_height_mm": self.lattice_grid_height * 1000,
            "lattice_line_diameter_mm": self.lattice_line_diameter * 1000,
            "lattice_width_mm": self.lattice_width * 1000,
            "lattice_height_mm": self.lattice_height * 1000,
            "lattice_stone_diameter_mm": self.lattice_stone_diameter * 1000,
            "camera_width": self.camera.width,
            "camera_height": self.camera.height,
            "camera_focal_length": self.camera.focal_length,
            "x_rot": self.camera_extrinsics.x_rot_degrees,
            "y_rot": self.camera_extrinsics.y_rot_degrees,
            "z_rot": self.camera_extrinsics.z_rot_degrees,
            "x": self.camera_extrinsics.x,
            "y": self.camera_extrinsics.y,
            "z": self.camera_extrinsics.z,
        }
        np.save(path, parameters)

    @staticmethod
    def from_parameters(path):
        parameters = np.load(path, allow_pickle=True).item()
        camera = image_utils.Camera(
            width=parameters["camera_width"],
            height=parameters["camera_height"],
            focal_length=parameters["camera_focal_length"],
        )
        lattice = Lattice2D(
            img=None,
            lines=None,
            camera=camera,
            lattice_size=parameters["num_lattice_lines"],
            lattice_grid_width_mm=parameters["lattice_grid_width_mm"],
            lattice_grid_height_mm=parameters["lattice_grid_height_mm"],
            lattice_line_diameter_mm=parameters["lattice_line_diameter_mm"],
            lattice_width_mm=parameters["lattice_width_mm"],
            lattice_height_mm=parameters["lattice_height_mm"],
            lattice_stone_diameter_mm=parameters["lattice_stone_diameter_mm"],
        )
        lattice.x_rot = parameters["x_rot"] * math.pi / 180
        lattice.y_rot = parameters["y_rot"] * math.pi / 180
        lattice.z_rot = parameters["z_rot"] * math.pi / 180
        lattice.x = parameters["x"]
        lattice.y = parameters["y"]
        lattice.z = parameters["z"]
        camera_extrinsics = geometry.Pose3D(
            translation=np.array([lattice.x, lattice.y, lattice.z]),
            x_rot=lattice.x_rot,
            y_rot=lattice.y_rot,
            z_rot=lattice.z_rot,
            degrees=False,
        )
        lattice.initial_camera_extrinsics = camera_extrinsics
        lattice.camera_extrinsics = camera_extrinsics
        return lattice


def estimate_video_lattice(board_size, camera, img_generator, **kwargs):
    best_match_score = 0
    best_extrinsics = None
    best_lattice = None
    average_det_lines_img = None
    for i, img in enumerate(img_generator):
        detected_lines = image_utils.find_lines_in_img(img, lattice_size=board_size)
        lattice = Lattice2D(
            img=img, lines=detected_lines, camera=camera, lattice_size=board_size
        )
        lattice.estimate_initial_parameters(debug=False)
        filtered_det_lines_img = (
            lattice.filtered_detected_lines_img_by_camera_extrinsics(
                camera, lattice.initial_camera_extrinsics, detected_lines
            )
        )
        match_score = lattice.camera_extrinsics_match_score(
            camera, lattice.initial_camera_extrinsics, filtered_det_lines_img
        )
        if average_det_lines_img is None:
            assert i == 0
            average_det_lines_img = filtered_det_lines_img
        else:
            average_det_lines_img = geometry.DistributionImage2D(
                (i * average_det_lines_img.img + filtered_det_lines_img.img) / (i + 1)
            )
        if match_score > best_match_score:
            best_match_score = match_score
            best_extrinsics = lattice.initial_camera_extrinsics
            best_lattice = lattice

    best_lattice.refine_camera_extrinsics_brute_force(
        camera, average_det_lines_img, **kwargs
    )

    return best_lattice


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lattice_estimation_parameters_path", type=str, default=None)
    parser.add_argument(
        "--train_lattice_estimation", action="store_true", default=False
    )
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--img_downsample", type=float, default=1)
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--num_x_bins", default=3, type=int)
    parser.add_argument("--num_y_bins", default=3, type=int)
    parser.add_argument("--num_z_bins", default=3, type=int)
    parser.add_argument("--num_x_rot_bins", default=3, type=int)
    parser.add_argument("--num_y_rot_bins", default=3, type=int)
    parser.add_argument("--num_z_rot_bins", default=3, type=int)
    parser.add_argument("--x_search_range_mm", default=20, type=float)
    parser.add_argument("--y_search_range_mm", default=20, type=float)
    parser.add_argument("--z_search_range_mm", default=10, type=float)
    parser.add_argument("--z_rot_search_range_degrees", default=3, type=float)

    args = parser.parse_args()

    if args.img_path is not None:
        test_img = cv2.imread(args.img_path)
        assert test_img is not None
        test_img = cv2.resize(
            test_img, (0, 0), fx=1.0 / args.img_downsample, fy=1.0 / args.img_downsample
        )
        test_camera = image_utils.Camera(
            width=test_img.shape[1], height=test_img.shape[0]
        )
    elif args.video_path is not None:
        cap = cv2.VideoCapture(args.video_path)
        assert cap.isOpened()
        test_camera = image_utils.Camera(
            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.train_lattice_estimation is False:
        lattice = Lattice2D.from_parameters(args.lattice_estimation_parameters_path)
    elif args.img_path is not None:
        detected_lines = image_utils.find_lines_in_img(
            test_img, lattice_size=args.board_size, debug=args.debug
        )
        lattice = Lattice2D(img=test_img, lines=detected_lines, camera=test_camera)
        lattice.estimate_initial_parameters(debug=args.debug)
        cv2.waitKey(0 if args.debug else 30)

        filtered_det_lines_img = (
            lattice.filtered_detected_lines_img_by_camera_extrinsics(
                test_camera, lattice.initial_camera_extrinsics, detected_lines
            )
        )

        print("verifying initial guess camera extrinsics...")
        match_score = lattice.camera_extrinsics_match_score(
            test_camera, lattice.initial_camera_extrinsics, filtered_det_lines_img
        )
        cv2.imshow(
            f"initial matching results: {match_score}",
            lattice.overlay_grid_with_camera_extrinsics(
                test_camera, lattice.initial_camera_extrinsics, test_img, difficulty=1
            ),
        )

        lattice.refine_camera_extrinsics_brute_force(
            test_camera,
            filtered_det_lines_img,
            num_x_bins=args.num_x_bins,
            num_y_bins=args.num_y_bins,
            num_z_bins=args.num_z_bins,
            num_x_rot_bins=args.num_x_rot_bins,
            num_y_rot_bins=args.num_y_rot_bins,
            num_z_rot_bins=args.num_z_rot_bins,
            z_rot_search_range_degrees=args.z_rot_search_range_degrees,
            z_search_range_mm=args.z_search_range_mm,
            x_search_range_mm=args.x_search_range_mm,
            y_search_range_mm=args.y_search_range_mm,
            debug=args.debug,
        )
        # save lattice parameters
        lattice.save_parameters(args.lattice_estimation_parameters_path)

    elif args.video_path is not None:
        img_generator = image_utils.video_generator(args.video_path)
        lattice = estimate_video_lattice(
            args.board_size,
            test_camera,
            img_generator,
            num_x_bins=args.num_x_bins,
            num_y_bins=args.num_y_bins,
            num_z_bins=args.num_z_bins,
            num_x_rot_bins=args.num_x_rot_bins,
            num_y_rot_bins=args.num_y_rot_bins,
            num_z_rot_bins=args.num_z_rot_bins,
            z_rot_search_range_degrees=args.z_rot_search_range_degrees,
            z_search_range_mm=args.z_search_range_mm,
            x_search_range_mm=args.x_search_range_mm,
            y_search_range_mm=args.y_search_range_mm,
        )
        # save lattice parameters
        lattice.save_parameters(args.lattice_estimation_parameters_path)

    if args.img_path is not None:
        print("verifying brute force camera extrinsics...")
        match_score = lattice.camera_extrinsics_match_score(
            test_camera, lattice.camera_extrinsics, filtered_det_lines_img
        )
        cv2.imshow(
            f"brute force matching results: {match_score}",
            lattice.overlay_grid_with_camera_extrinsics(
                test_camera, lattice.camera_extrinsics, test_img, difficulty=1
            ),
        )
        cv2.waitKey(0 if args.debug else 30)
    elif args.video_path is not None:
        img_generator = image_utils.video_generator(args.video_path)
        for img in img_generator:
            cv2.imshow(
                f"matching results",
                lattice.overlay_grid_with_camera_extrinsics(
                    test_camera, lattice.camera_extrinsics, img, difficulty=1
                ),
            )
            cv2.waitKey(0 if args.debug else 30)
