import cv2
import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from move_estimation import Stone


class ColorLightingEstimation:
    def __init__(
        self,
        board_size,
        colors_default_value,
        lighting_default_value=0.9,
        covariance_default_value=0.1,
    ):
        assert 0 <= lighting_default_value <= 1

        self.board_size = board_size
        self.num_colors = colors_default_value.shape[0]

        self.train_observations = None
        self.priors = np.full(
            (self.num_colors,), dtype=float, fill_value=1 / self.num_colors
        )
        self.lighting = np.full(
            (self.board_size, self.board_size),
            fill_value=lighting_default_value,
            dtype=float,
        )
        self.colors = colors_default_value
        self.color_covariances = [
            self.generate_initial_color_covariance(
                color, covariance_default_value=0.1, singularity=lighting_default_value
            )
            for color in self.colors
        ]
        self.color_covariances = np.stack(self.color_covariances)

        self.check_valid()

    def set_train_observations(self, train_observations):
        self.train_observations = train_observations
        self.num_samples = self.train_observations.shape[2]
        self.responsibility = np.full(
            [self.board_size, self.board_size, self.num_samples, self.num_colors],
            dtype=float,
            fill_value=1 / self.num_colors,
        )
        self.check_valid()

    def generate_initial_color_covariance(
        self, color, covariance_default_value, singularity=0.9
    ):
        assert singularity > 0
        assert singularity < 1
        assert np.linalg.norm(color) > 0

        color = np.copy(color)

        color /= np.linalg.norm(color)

        x = np.random.randn(3)
        x -= x.dot(color) * color
        x /= np.linalg.norm(x)
        assert np.isclose(x.dot(color), 0)

        y = np.cross(x, color)
        assert np.isclose(y.dot(y), 1)

        eigenvectors = np.transpose(np.array([color, x, y]))
        assert np.allclose(eigenvectors[:, 0], color)
        assert np.allclose(eigenvectors[:, 1], x)
        assert np.allclose(eigenvectors[:, 2], y)
        eigenvalues = np.array(
            [
                covariance_default_value,
                (1 - singularity) * covariance_default_value,
                (1 - singularity) * covariance_default_value,
            ]
        )

        covariance = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        calculated_eigenvalues = np.linalg.eigvals(covariance)
        assert np.allclose(np.sort(np.linalg.eigvals(covariance)), np.sort(eigenvalues))

        return covariance

    def display_clustering(self):
        self.check_valid()
        observations = np.reshape(self.train_observations, [-1, 3])
        best_index = np.reshape(np.argmax(self.responsibility, axis=-1), [-1])
        colors = self.colors[best_index]

        alpha = 1
        if self.train_observations.size / 3 > 100:
            alpha = 0.5
        if self.train_observations.size / 3 > 1000:
            alpha = 0.25
        if self.train_observations.size / 3 > 10000:
            alpha = 0.125
        if self.train_observations.size / 3 > 100000:
            alpha = 0.125 / 2
        ax = plt.figure().add_subplot(projection="3d")
        ax.scatter(
            xs=observations[:, 0],
            ys=observations[:, 1],
            zs=observations[:, 2],
            c=colors[:, ::-1],
            label="_",
            alpha=alpha,
        )
        cluster_axis_colors = ["r", "g", "b"]

        for color, covariance in zip(self.colors, self.color_covariances):
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            for i in range(3):
                cov_axis_pt = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
                line_pts_3d = np.stack([color, color + cov_axis_pt])
                assert line_pts_3d.shape == (2, 3)
                ax.plot(
                    xs=line_pts_3d[:, 0],
                    ys=line_pts_3d[:, 1],
                    zs=line_pts_3d[:, 2],
                    color=cluster_axis_colors[i],
                    label="_",
                )

        ax.xaxis.set_pane_color((0.5, 0.5, 0.5, 0.0))
        ax.yaxis.set_pane_color((0.5, 0.5, 0.5, 0.0))
        ax.zaxis.set_pane_color((0.5, 0.5, 0.5, 0.0))

        # Make legend, set axes limits and label
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("B")
        ax.set_ylabel("G")
        ax.set_zlabel("R")

        ax.view_init(elev=20.0, azim=0, roll=0)

        plt.show()

    def check_valid_range(self, x):
        assert np.all(x >= 0)
        assert np.all(x <= 1)

    def check_valid(self):
        # check train data
        if self.train_observations is not None:
            self.check_valid_range(self.train_observations)
            assert self.train_observations.shape == (
                self.board_size,
                self.board_size,
                self.num_samples,
                3,
            )

            # check responsibility sums to 1 along cluster axis
            self.check_valid_range(self.responsibility)
            assert np.allclose(np.sum(self.responsibility, axis=-1), 1)
            assert self.responsibility.shape == (
                self.board_size,
                self.board_size,
                self.num_samples,
                self.num_colors,
            )

        # check lighting are in [0,1]
        self.check_valid_range(self.lighting)
        assert self.lighting.shape == (self.board_size, self.board_size)

        # check colors are in [0,1]
        self.check_valid_range(self.colors)
        assert self.colors.shape == (self.num_colors, 3)

        # check priors sum to 1 along cluster axis
        self.check_valid_range(self.priors)
        assert np.allclose(np.sum(self.priors, axis=-1), 1)
        assert self.priors.shape == (self.num_colors,)

        # check covariances are symmetric positive definite
        assert np.allclose(
            np.transpose(self.color_covariances, [0, 2, 1]), self.color_covariances
        )
        assert np.all(np.linalg.eigvals(self.color_covariances) > 0)
        assert self.color_covariances.shape == (self.num_colors, 3, 3)

    def estimate_responsibility(self):
        # update the responsibility array given train_observations, lighting, colors array
        self.check_valid()

        estimates = np.multiply.outer(self.lighting, self.colors)
        self.check_valid_range(estimates)
        assert estimates.shape == (self.board_size, self.board_size, self.num_colors, 3)
        estimates = np.reshape(
            estimates, [self.board_size, self.board_size, 1, self.num_colors, 3]
        )
        estimates = np.tile(estimates, [1, 1, self.num_samples, 1, 1])
        assert estimates.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )

        train_observations = np.expand_dims(self.train_observations, 3)
        assert train_observations.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            1,
            3,
        )

        error = estimates - train_observations
        assert error.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )
        error = np.expand_dims(error, 4)
        assert error.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            3,
        )

        inv_covariances = np.linalg.inv(self.color_covariances)
        assert inv_covariances.shape == (self.num_colors, 3, 3)
        assert np.allclose(
            inv_covariances @ self.color_covariances,
            np.stack([np.eye(3, dtype=float)] * self.num_colors),
        )
        det_covariances = np.linalg.det(self.color_covariances)
        assert det_covariances.shape == (self.num_colors,)

        x = -0.5 * error @ inv_covariances @ np.transpose(error, [0, 1, 2, 3, 5, 4])
        assert x.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            1,
        )
        x = np.reshape(
            x, [self.board_size, self.board_size, self.num_samples, self.num_colors]
        )

        # this ok because we need to normalize along that axis eventually anyway
        x = x - np.max(x, axis=-1, keepdims=True)
        x = np.exp(x)
        x /= np.reshape(np.sqrt(det_covariances), [1, 1, 1, -1])
        x *= np.reshape(self.priors, [1, 1, 1, -1])
        # normalize responsibility so that it is a probability
        self.responsibility = x / np.sum(x, axis=-1, keepdims=True)

        self.check_valid()

    def estimate_priors(self):
        self.check_valid()
        self.priors = np.mean(self.responsibility, axis=(0, 1, 2))
        assert self.priors.shape == (self.num_colors,)
        self.check_valid()

    def estimate_lighting(self):
        # update the lighting array given train_observations, colors, responsibility array
        self.check_valid()

        inv_covariances = np.linalg.inv(self.color_covariances)
        assert inv_covariances.shape == (self.num_colors, 3, 3)
        assert np.allclose(
            inv_covariances @ self.color_covariances,
            np.stack([np.eye(3, dtype=float)] * self.num_colors),
        )

        train_observations = np.expand_dims(self.train_observations, (3, 4))
        train_observations = np.tile(
            train_observations, [1, 1, 1, self.num_colors, 1, 1]
        )
        assert train_observations.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            3,
        )

        numerator = (
            train_observations
            @ inv_covariances
            @ np.reshape(self.colors, [self.num_colors, 3, 1])
        )
        assert numerator.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            1,
        )
        numerator = np.reshape(
            numerator,
            (self.board_size, self.board_size, self.num_samples, self.num_colors),
        )

        denominator = (
            np.reshape(self.colors, [self.num_colors, 1, 3])
            @ inv_covariances
            @ np.reshape(self.colors, [self.num_colors, 3, 1])
        )
        assert denominator.shape == (self.num_colors, 1, 1)
        denominator = np.reshape(denominator, [1, 1, 1, self.num_colors])

        self.lighting = np.sum(numerator * self.responsibility, axis=(2, 3)) / np.sum(
            denominator * self.responsibility, axis=(2, 3)
        )

        # make sure self.lighting is in [0,1] range
        if np.max(self.lighting) > 1:
            lighting_shift = np.max(self.lighting)
            max_lighting_shift = 1 / np.max(self.colors)
            lighting_shift = min(lighting_shift, max_lighting_shift)
            self.colors *= lighting_shift
            self.lighting /= lighting_shift
            # worst case scenario
            self.lighting = np.clip(self.lighting, 0, 1)

        # gaussian smooth lighting because it should be continuous
        self.lighting = cv2.GaussianBlur(
            self.lighting, (5, 5), 0, borderType=cv2.BORDER_REPLICATE
        )

        self.check_valid()

    def estimate_colors(self):
        # update the colors array given train_observations, responsibility, lighting array
        self.check_valid()

        train_observations = np.expand_dims(self.train_observations, 3)
        train_observations = np.tile(train_observations, [1, 1, 1, self.num_colors, 1])
        assert train_observations.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )

        responsibility = np.reshape(
            self.responsibility,
            [self.board_size, self.board_size, self.num_samples, self.num_colors, 1],
        )
        lighting = np.reshape(
            self.lighting, [self.board_size, self.board_size, 1, 1, 1]
        )

        numerator = responsibility * train_observations * lighting
        assert numerator.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )

        denominator = responsibility * (lighting**2)
        assert denominator.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
        )

        self.colors = np.sum(numerator, axis=(0, 1, 2)) / np.sum(
            denominator, axis=(0, 1, 2)
        )
        assert self.colors.shape == (self.num_colors, 3)

        # make sure self.colors is in [0,1] range
        # make sure self.lighting is in [0,1] range
        if np.max(self.colors) > 1:
            lighting_shift = np.max(self.colors)
            max_lighting_shift = 1 / np.max(self.lighting)
            lighting_shift = min(lighting_shift, max_lighting_shift)
            self.colors /= lighting_shift
            self.lighting *= lighting_shift
            # worst case scenario
            self.colors = np.clip(self.colors, 0, 1)

        self.check_valid()

    def estimate_color_covariances(self):
        self.check_valid()

        estimates = np.multiply.outer(self.lighting, self.colors)
        self.check_valid_range(estimates)
        assert estimates.shape == (self.board_size, self.board_size, self.num_colors, 3)

        estimates = np.reshape(
            estimates, [self.board_size, self.board_size, 1, self.num_colors, 3]
        )
        estimates = np.tile(estimates, [1, 1, self.num_samples, 1, 1])
        assert estimates.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )

        train_observations = np.reshape(
            self.train_observations,
            [self.board_size, self.board_size, self.num_samples, 1, 3],
        )

        error = estimates - train_observations
        assert error.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )
        error_col = np.expand_dims(error, 4)
        assert error_col.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            3,
        )
        error_row = np.expand_dims(error, 5)
        assert error_row.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
            1,
        )

        x = error_row @ error_col
        assert x.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
            3,
        )
        assert np.allclose(np.transpose(x, [0, 1, 2, 3, 5, 4]), x)

        numerator = np.expand_dims(self.responsibility, [4, 5]) * x
        assert numerator.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
            3,
        )
        assert np.allclose(np.transpose(numerator, [0, 1, 2, 3, 5, 4]), numerator)

        denominator = np.expand_dims(self.responsibility, [4, 5])
        assert denominator.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            1,
        )
        self.color_covariances = np.sum(numerator, axis=(0, 1, 2)) / np.sum(
            denominator, axis=(0, 1, 2)
        )

        self.check_valid()

    def calculate_training_log_prob(self):
        self.check_valid()

        estimates = np.multiply.outer(self.lighting, self.colors)
        self.check_valid_range(estimates)
        assert estimates.shape == (self.board_size, self.board_size, self.num_colors, 3)
        estimates = np.reshape(
            estimates, [self.board_size, self.board_size, 1, self.num_colors, 3]
        )
        estimates = np.tile(estimates, [1, 1, self.num_samples, 1, 1])
        assert estimates.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )

        train_observations = np.expand_dims(self.train_observations, 3)
        assert train_observations.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            1,
            3,
        )

        error = estimates - train_observations
        assert error.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            3,
        )
        error = np.expand_dims(error, 4)
        assert error.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            3,
        )

        inv_covariances = np.linalg.inv(self.color_covariances)
        assert inv_covariances.shape == (self.num_colors, 3, 3)
        assert np.allclose(
            inv_covariances @ self.color_covariances,
            np.stack([np.eye(3, dtype=float)] * self.num_colors),
        )
        det_covariances = np.linalg.det(self.color_covariances)
        assert det_covariances.shape == (self.num_colors,)

        x = error @ inv_covariances @ np.transpose(error, [0, 1, 2, 3, 5, 4])
        assert np.all(x >= 0)
        assert x.shape == (
            self.board_size,
            self.board_size,
            self.num_samples,
            self.num_colors,
            1,
            1,
        )
        x = np.reshape(
            x, [self.board_size, self.board_size, self.num_samples, self.num_colors]
        )

        x *= -0.5
        x += np.reshape(np.log(self.priors), [1, 1, 1, -1])
        x -= 1.5 * np.log(2 * math.pi)
        x -= 0.5 * np.reshape(np.log(det_covariances), [1, 1, 1, -1])

        # prob doesn't have to be <=1 because it is a pdf value
        prob = np.sum(np.exp(x), axis=-1)
        return np.log(prob)

    def select_log_prob_threshold(self, percentile):
        log_probs = self.calculate_training_log_prob()
        assert log_probs.shape == (self.board_size, self.board_size, self.num_samples)
        self.log_prob_noise = np.percentile(np.reshape(log_probs, [-1]), percentile)
        return self.log_prob_noise

    def estimate_parameters(
        self, max_iter=100, convergence_threshold=0.001, debug=False
    ):

        def _debug_information(x=""):
            print(x)
            print("priors")
            print(self.priors)
            for i, (color, covariance) in enumerate(
                zip(self.colors, self.color_covariances)
            ):
                print("cluster", i)
                print(color)
                print(covariance)
            print("lighting")
            print(np.round(self.lighting, 2))
            self.display_clustering()

        iter = 0
        last_mean_log_prob = np.mean(self.calculate_training_log_prob())

        if debug:
            print("starting mean prob", last_mean_log_prob)
            _debug_information("initialization")

        while iter < max_iter:
            self.estimate_responsibility()
            for i in range(10):
                # the EM algo didn't have a clean solution so need sub iterations
                self.estimate_priors()
                self.estimate_lighting()
                self.estimate_colors()
                self.estimate_color_covariances()
            mean_log_prob = np.mean(self.calculate_training_log_prob())
            if debug:
                _debug_information()
            if mean_log_prob <= last_mean_log_prob * (1 + convergence_threshold):
                break
            last_mean_log_prob = mean_log_prob
            iter += 1

        return mean_log_prob

    def calculate_test_prob(self, board):
        self.check_valid_range(board)
        num_samples = board.shape[2]
        assert board.shape == (self.board_size, self.board_size, num_samples, 3)
        board = np.reshape(board, [self.board_size, self.board_size, num_samples, 1, 3])

        estimates = np.multiply.outer(self.lighting, self.colors)
        self.check_valid_range(estimates)
        assert estimates.shape == (self.board_size, self.board_size, self.num_colors, 3)
        estimates = np.reshape(
            estimates, [self.board_size, self.board_size, 1, self.num_colors, 3]
        )
        estimates = np.tile(estimates, [1, 1, num_samples, 1, 1])
        assert estimates.shape == (
            self.board_size,
            self.board_size,
            num_samples,
            self.num_colors,
            3,
        )

        error_row = np.reshape(
            board - estimates,
            [self.board_size, self.board_size, num_samples, self.num_colors, 1, 3],
        )
        error_col = np.reshape(
            board - estimates,
            [self.board_size, self.board_size, num_samples, self.num_colors, 3, 1],
        )

        inv_covariances = np.linalg.inv(self.color_covariances)
        det_covariances = np.linalg.det(self.color_covariances)

        x = error_row @ inv_covariances @ error_col
        x = np.reshape(
            x, [self.board_size, self.board_size, num_samples, self.num_colors]
        )
        assert np.all(x >= 0)

        x *= -0.5
        x += np.reshape(np.log(self.priors), [1, 1, 1, -1])
        x -= 1.5 * np.log(2 * math.pi)
        x -= 0.5 * np.reshape(np.log(det_covariances), [1, 1, 1, -1])

        log_prob = x
        assert log_prob.shape == (
            self.board_size,
            self.board_size,
            num_samples,
            self.num_colors,
        )

        unnormalized_prob = np.exp(log_prob)
        normalization_factor = np.sum(unnormalized_prob, axis=-1, keepdims=True)
        prob = unnormalized_prob / normalization_factor
        assert log_prob.shape == (
            self.board_size,
            self.board_size,
            num_samples,
            self.num_colors,
        )
        return prob


class CircleSampling:
    def __init__(self, k, r, num_samples=None, offset=None, prior_sigma=0.5):
        assert k > 0
        assert 0 < r < 0.5

        self.k = k
        self.r = r

        if offset is None:
            offset = math.pi / k
        angles = np.array([2 * math.pi * i / k for i in range(k)])
        offset_angle = angles + offset
        self.points = np.stack(
            [
                r * np.cos(offset_angle),
                r * np.sin(offset_angle),
            ],
            axis=1,
        )
        assert self.points.shape == (k, 2)

        self.prior_sigma = prior_sigma
        if num_samples is not None:
            self.approximate_pdf(num_samples=num_samples)

    def _subset_to_code(self, subset):
        code = [2**i for i, x in enumerate(subset) if x]
        return sum(code)

    def _code_to_subset(self, code):
        subset = [(code >> i) & 1 == 1 for i in range(self.k)]
        assert self._subset_to_code(subset) == code, code
        return np.array(subset).astype(int)

    def approximate_pdf(self, num_samples):
        num_samples = int(num_samples)
        samples = np.random.randn(num_samples, 2) * self.prior_sigma
        valid = np.linalg.norm(samples, axis=1) < 1

        samples = np.expand_dims(samples[valid, :], 1)
        points = np.reshape(self.points, [1, self.k, 2])
        distances = np.linalg.norm(samples - points, axis=2)

        codes = np.apply_along_axis(self._subset_to_code, 1, distances <= 1)
        values, counts = np.unique(codes, return_counts=True)
        counts = counts.astype(float) / np.sum(counts)
        self.pdf = np.array(
            [
                np.concat([subset, np.array([pdf])])
                for subset, pdf in zip(
                    [self._code_to_subset(code) for code in values], counts
                )
            ]
        )

    def calculate_probability(self, probabilities):
        assert probabilities.shape == (self.k, 2)
        posterior = [
            np.prod(
                np.take_along_axis(
                    probabilities, np.expand_dims(1 - pdf[:-1].astype(int), 1), axis=1
                )
            )
            * pdf[-1]
            for pdf in self.pdf
        ]
        return np.sum(posterior)


class BoardEstimator:
    def __init__(
        self,
        board_size,
        colors_default_value=None,
        circle_sampling_k=8,
        circle_sampling_r=0.25,
    ):
        self.board_size = board_size

        if colors_default_value is None:
            colors_default_value = np.array(
                [
                    [10, 10, 10],
                    [90, 110, 130],
                    [110, 130, 150],
                    [130, 150, 170],
                    [150, 170, 190],
                    [170, 190, 210],
                    [245, 245, 245],
                ]
            )

        assert colors_default_value.shape[1] == 3
        assert colors_default_value.shape[0] >= 3
        self.colors_default_value = colors_default_value
        # first row is black stone color
        # second row to penultimate row are board colors affected by temporary shadows
        # last row is white stone color
        self.color_clusters_grouping = {
            Stone.BLACK: [0],
            Stone.WHITE: [len(self.colors_default_value) - 1],
            Stone.EMPTY: np.arange(1, len(colors_default_value) - 1),
        }

        self.circle_sampling = CircleSampling(
            k=circle_sampling_k, r=circle_sampling_r, num_samples=None
        )
        self.color_lighting_estimator = ColorLightingEstimation(
            board_size=board_size, colors_default_value=self.colors_default_value / 255
        )
        self.parameters_list = [
            "priors",
            "lighting",
            "colors",
            "color_covariances",
            "pdf",
            "pdf_sigma",
        ]

    def save_parameters(self, path):
        pdf = self.circle_sampling.pdf

        parameters = {
            "priors": self.color_lighting_estimator.priors,
            "lighting": self.color_lighting_estimator.lighting,
            "colors": self.color_lighting_estimator.colors,
            "color_covariances": self.color_lighting_estimator.color_covariances,
            "pdf": self.circle_sampling.pdf,
            "pdf_sigma": self.circle_sampling.prior_sigma,
        }
        with open(path, "wb") as f:
            for name in self.parameters_list:
                np.save(f, parameters[name], allow_pickle=True)

    def load_parameters(self, path):
        parameters = {}
        with open(path, "rb") as f:
            for name in self.parameters_list:
                parameters[name] = np.load(f, allow_pickle=True)

        self.color_lighting_estimator.priors = parameters["priors"]
        self.color_lighting_estimator.lighting = parameters["lighting"]
        self.color_lighting_estimator.colors = parameters["colors"]
        self.color_lighting_estimator.color_covariances = parameters[
            "color_covariances"
        ]
        self.circle_sampling.pdf = parameters["pdf"]
        assert self.circle_sampling.prior_sigma == parameters["pdf_sigma"]

    def stone_sampling_pts(
        self,
        img,
        lattice,
        camera,
        coord1,
        coord2,
    ):
        lattice_offset_pts = self.circle_sampling.points
        lattice_offset_pts = np.concatenate(
            [lattice_offset_pts, np.zeros_like(lattice_offset_pts[:, 0:1])], axis=1
        )

        lattice_offset_pts *= lattice.lattice_stone_diameter / 2
        assert lattice_offset_pts.shape == (self.circle_sampling.k, 3)
        assert np.isclose(np.mean(lattice_offset_pts), 0)

        lattice_offset_pts_2d = camera.project_points(
            lattice.camera_extrinsics,
            lattice.lattice_3d[coord1, coord2] + lattice_offset_pts,
        )
        lattice_offset_pts_2d = np.round(lattice_offset_pts_2d).astype(int)

        lattice_pt_colors = np.array(
            [img[pt[1], pt[0]] for pt in lattice_offset_pts_2d]
        )
        assert lattice_pt_colors.shape[0] == self.circle_sampling.k
        return lattice_pt_colors, lattice_offset_pts_2d

    @property
    def stone_colors(self):
        return self.color_lighting_estimator.colors * 255

    def estimate_parameters(
        self,
        imgs,
        lattices,
        camera,
        max_imgs=10,
        pdf_monte_carlo_samples=1e6,
        debug=False,
    ):
        # estimate the color of the board and the black and white stones
        # this needs to be empirically estimated because lighting conditions vary
        # imgs is a generator or list of images
        data = np.zeros(
            [self.board_size, self.board_size, max_imgs, self.circle_sampling.k, 3]
        )

        for i, img in enumerate(imgs):
            lattice = lattices[i]
            for coord1 in range(self.board_size):
                for coord2 in range(self.board_size):
                    colors = self.stone_sampling_pts(
                        img=img,
                        lattice=lattice,
                        camera=camera,
                        coord1=coord1,
                        coord2=coord2,
                    )[0]
                    # too slow to use every coord in every image in kmeans
                    # instead sample from each image point randomly
                    for k in range(self.circle_sampling.k):
                        if np.random.uniform() <= 1 / (i // max_imgs + 1):
                            data[coord1, coord2, i % max_imgs, k, :] = colors[k, :]
        if i < max_imgs:
            data = data[:, :, :i, ...]
        data = np.reshape(data, [self.board_size, self.board_size, -1, 3])

        self.color_lighting_estimator.set_train_observations(data / 255)
        self.color_lighting_estimator.estimate_parameters(debug=debug)
        self.color_estimation_noise = np.exp(
            self.color_lighting_estimator.select_log_prob_threshold(percentile=1)
        )

        for i in self.color_clusters_grouping[Stone.EMPTY]:
            assert np.mean(self.stone_colors[0]) <= np.mean(self.stone_colors[i])
            assert np.mean(self.stone_colors[i]) <= np.mean(self.stone_colors[-1])

        self.circle_sampling.approximate_pdf(pdf_monte_carlo_samples)

    def get_instantaneous_board_state_probabilities(
        self,
        lattice,
        camera,
        img,
        obstruction_map,
        obstruction_threshold=0.5,
    ):

        board_lattice_colors = np.array(
            [
                [
                    self.stone_sampling_pts(
                        img=img, lattice=lattice, camera=camera, coord1=i, coord2=j
                    )[0]
                    for j in range(self.board_size)
                ]
                for i in range(self.board_size)
            ]
        )
        assert board_lattice_colors.shape == (
            self.board_size,
            self.board_size,
            self.circle_sampling.k,
            3,
        )

        cluster_grouping = {
            (stone_type.value): indices
            for (stone_type, indices) in self.color_clusters_grouping.items()
        }

        def collapse_probability(x):
            x = np.reshape(x, [self.circle_sampling.k, 3])
            probabilities = np.zeros(3, dtype=float)
            probabilities[Stone.BLACK.value] = (
                self.circle_sampling.calculate_probability(
                    x[:, [Stone.BLACK.value, Stone.EMPTY.value]]
                )
            )
            probabilities[Stone.WHITE.value] = (
                self.circle_sampling.calculate_probability(
                    x[:, [Stone.WHITE.value, Stone.EMPTY.value]]
                )
            )
            probabilities[Stone.EMPTY.value] = np.prod(x[:, Stone.EMPTY.value])
            # return probabilities / (np.sum(probabilities) + math.pow(self.color_estimation_noise, self.circle_sampling.k))
            return probabilities / np.sum(probabilities)

        board_lattice_probs = self.color_lighting_estimator.calculate_test_prob(
            board_lattice_colors / 255
        )
        assert board_lattice_probs.shape == (
            self.board_size,
            self.board_size,
            self.circle_sampling.k,
            self.color_lighting_estimator.num_colors,
        )

        group_prob = np.zeros(
            [
                self.board_size,
                self.board_size,
                self.circle_sampling.k,
                len(cluster_grouping),
            ],
            dtype=float,
        )
        for i, indices in cluster_grouping.items():
            group_prob[:, :, :, i] = np.sum(
                board_lattice_probs[:, :, :, indices], axis=-1
            )
        assert np.all(group_prob >= 0)

        probabilities = np.apply_along_axis(
            collapse_probability,
            2,
            np.reshape(group_prob, [self.board_size, self.board_size, -1]),
        )
        assert probabilities.shape == (self.board_size, self.board_size, 3)

        obstruction_indices = obstruction_map >= obstruction_threshold
        probabilities[obstruction_indices, :] = np.nan

        return probabilities

    @staticmethod
    def instantaneous_probabilities_to_placements(board_probabilities):
        obstructed = np.all(np.isnan(board_probabilities), axis=2)
        board_probabilities[obstructed, :] = np.array([0, 1, 0])
        return np.argmax(board_probabilities, axis=2)

    def draw_board(
        self,
        camera,
        lattice,
        stone_placements,
        k=32,
        difficulty=1.0,
        stone_probabilities=None,
    ):
        # debug function to draw an image from the perspective of a camera and its extrinsics
        # ignores lighting

        thickness = (
            lattice.lattice_line_diameter
            * camera.focal_length
            / lattice.camera_extrinsics.z
            / difficulty
        )
        thickness = int(math.ceil(thickness))
        assert thickness > 0

        assert stone_placements.shape == (self.board_size, self.board_size)
        canvas = np.zeros([camera.height, camera.width, 3], dtype=np.uint8)

        # draw board
        board_color = self.stone_colors[len(self.stone_colors) // 2]
        corners_2d = camera.project_points(
            lattice.camera_extrinsics, lattice.corners_3d
        )
        corners_2d = np.round(corners_2d).astype(int)
        corners_2d = np.reshape(corners_2d, [-1, 2])[[0, 1, 3, 2], :]
        canvas = cv2.fillConvexPoly(canvas, corners_2d, color=board_color)

        lattice_coords_2d = camera.project_points(
            lattice.camera_extrinsics, lattice.lattice_3d
        )

        # draw lattice
        lattice_lines = lattice.generate_lattice_lines(lattice_coords_2d)
        canvas = lattice_lines.overlay_lines(
            canvas, color=(0, 0, 0), thickness=thickness
        )

        circle_angle = np.array([2 * math.pi * i / k for i in range(k)])
        circle_pts = np.array(
            [
                np.cos(circle_angle),
                np.sin(circle_angle),
                np.full_like(circle_angle, fill_value=0.001),
            ]
        )
        circle_pts = np.transpose(circle_pts) * lattice.lattice_stone_diameter / 2
        probabilities_circle_pts = circle_pts * 0.25

        # draw stones
        for i in range(lattice.num_lattice_lines):
            for j in range(lattice.num_lattice_lines):
                stone_pts_3d = lattice.lattice_3d[i, j] + circle_pts
                stone_pts_2d = camera.project_points(
                    lattice.camera_extrinsics, stone_pts_3d
                )
                stone_pts_2d = np.round(stone_pts_2d).astype(int)
                if stone_placements[i, j] == Stone.BLACK.value:
                    color = (0, 0, 0)
                elif stone_placements[i, j] == Stone.WHITE.value:
                    color = (255, 255, 255)
                else:
                    assert stone_placements[i, j] == Stone.EMPTY.value
                if stone_placements[i, j] != Stone.EMPTY.value:
                    canvas = cv2.fillConvexPoly(canvas, stone_pts_2d, color=color)
                if stone_probabilities is not None:
                    if not np.any(np.isnan(stone_probabilities[i, j])):
                        stone_prob_pts_3d = (
                            lattice.lattice_3d[i, j] + probabilities_circle_pts
                        )
                        stone_prob_pts_2d = camera.project_points(
                            lattice.camera_extrinsics, stone_prob_pts_3d
                        )
                        stone_prob_pts_2d = np.round(stone_prob_pts_2d).astype(int)
                        color = (stone_probabilities[i, j, ::-1] * 255).astype(int)
                        canvas = cv2.fillConvexPoly(
                            canvas, stone_prob_pts_2d, color=color.tolist()
                        )
        return canvas


if __name__ == "__main__":
    import argparse
    import image_utils
    import lattice_estimation

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(help="train or inference mode")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--board_estimation_parameters_path", type=str, required=True)
    parser.add_argument("--lattice_estimation_parameters_path", type=str, required=True)
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--train_board_estimation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened()
    test_camera = image_utils.Camera(
        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lattice = lattice_estimation.Lattice2D.from_parameters(
        args.lattice_estimation_parameters_path
    )

    board_estimator = BoardEstimator(
        board_size=args.board_size,
        circle_sampling_k=8,
        circle_sampling_r=0.25,
    )
    if args.train_board_estimation:
        print(
            "estimating parameters and saving to", args.board_estimation_parameters_path
        )
        board_estimator.estimate_parameters(
            imgs=image_utils.video_generator(args.video_path),
            lattices=[lattice] * num_frames,
            camera=test_camera,
            max_imgs=num_frames,
            debug=args.debug,
        )
        board_estimator.save_parameters(args.board_estimation_parameters_path)
    else:
        print("loading parameters from", args.board_estimation_parameters_path)
        board_estimator.load_parameters(args.board_estimation_parameters_path)

    for img in image_utils.video_generator(args.video_path):
        detected_lines = image_utils.find_lines_in_img(
            img, lattice_size=args.board_size, debug=False
        )
        filtered_det_lines_img = (
            lattice.filtered_detected_lines_img_by_camera_extrinsics(
                test_camera, lattice.initial_camera_extrinsics, detected_lines
            )
        )
        obstruction_map = lattice.estimate_obstruction_map(
            test_camera, filtered_det_lines_img
        )
        stone_probabilities = (
            board_estimator.get_instantaneous_board_state_probabilities(
                lattice=lattice,
                camera=test_camera,
                img=img,
                obstruction_map=obstruction_map,
            )
        )
        stone_placements = BoardEstimator.instantaneous_probabilities_to_placements(
            stone_probabilities
        )
        estimated_img = board_estimator.draw_board(
            camera=test_camera,
            lattice=lattice,
            stone_placements=stone_placements,
            stone_probabilities=stone_probabilities,
        )
        cv2.imshow("reconstruction", estimated_img)
        cv2.imshow(
            f"brute force matching results",
            lattice.overlay_grid_with_camera_extrinsics(
                test_camera, lattice.camera_extrinsics, img, difficulty=1
            ),
        )
        cv2.waitKey(0 if args.debug else 30)
