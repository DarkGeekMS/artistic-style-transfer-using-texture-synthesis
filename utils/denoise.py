import numpy as np


def denoise_image(img, sigma_s=60, sigma_r=0.4, iterations=1):
    """
    Performs edge-aware denoising (smoothing) on an image using domain transformation.
    The complexity is linear in the total number of pixels.
    Performs 1D smoothing along all rows then along all columns (a single iteration).

    Args:
        img: Input image.
        sigma_s: The support (SD) of the Gaussian spacial filter.
        sigma_r: The support (SD) of the Gaussian range filter.
        iterations: The number of iterations.

    Returns:
        Smoothed image.
    """
    # If the image range is 0 to 255, change it to 0 to 1.
    if not np.issubclass_(img.dtype.type, np.floating):
        img = img / 255.0
    # Calculate the difference (discrete derivative) in each axis (X & Y)
    # Pad 0 to the beginning of the axis (as the diff size is less by 1 - no diff is calculate for the first element)
    derivative_x = np.abs(np.diff(img, axis=1, prepend=0))
    derivative_y = np.abs(np.diff(img, axis=0, prepend=0))

    # Sum the derivatives absolute values over all color channels
    derivative_x_channels_sum = np.sum(np.abs(derivative_x), axis=2)
    derivative_y_channels_sum = np.sum(np.abs(derivative_y), axis=2)

    # Calculate the domain transform in X & Y (discrete integral)
    transformed_pixels_x = np.cumsum(1 + sigma_s / sigma_r * derivative_x_channels_sum, axis=1)
    transformed_pixels_y = np.cumsum(1 + sigma_s / sigma_r * derivative_y_channels_sum, axis=0)

    smoothed_img = img
    for i in range(iterations):
        # Calculate the box radius for every iteration
        # sigma_h is decreased with each iteration and so is the box_radius, to reduce stripes artifacts
        sigma_h = sigma_s * np.sqrt(3) * np.power(2, (iterations - (i + 1))) / np.sqrt(np.power(4, iterations) - 1)
        box_radius = sigma_h * np.sqrt(3)
        # Apply smoothing along the horizontal axis
        smoothed_img = apply_linear_smoothing_filter_horizontally(smoothed_img, transformed_pixels_x, box_radius)
        # Apply smoothing along the vertical axis by transposing the image first
        smoothed_img = apply_linear_smoothing_filter_horizontally(smoothed_img.transpose(1, 0, 2),
                                                                  transformed_pixels_y.T,
                                                                  box_radius)
        smoothed_img = smoothed_img.transpose(1, 0, 2)

    return (smoothed_img * 255).astype(int)


def apply_linear_smoothing_filter_horizontally(img, transformed_pixels, box_radius):
    """
    Runs smoothing horizontally along all rows of the image.

    Args:
        img: Input image.
        transformed_pixels: The transformation of the pixels along the x axis.
        box_radius: The radius of the filtering box in the transformed domain.

    Returns:
        Horizontally smoothed image.
    """
    # Define the lower and upper bounds in terms of transformed pixels values for each pixel
    lower_bounds, upper_bounds = transformed_pixels - box_radius, transformed_pixels + box_radius
    lower_bound_indices = np.zeros_like(transformed_pixels, dtype='uint16')
    upper_bound_indices = np.zeros_like(transformed_pixels, dtype='uint16')
    sum_table = np.cumsum(img, axis=1)
    height = img.shape[0]

    # For each row, find the indices of the lower and upper bounds for each pixel among transformed pixels.
    for i in range(height):
        lower_bound_indices[i] = np.searchsorted(transformed_pixels[i], lower_bounds[i])
        upper_bound_indices[i] = np.searchsorted(transformed_pixels[i], upper_bounds[i], side='right') - 1

    row_indices = np.indices(upper_bound_indices.shape)[0]  # Dummy matrix to produce row indices
    # For each row,
    # the new image value is the average of the pixels whose transformations are within the upper and lower bounds
    # A sum table is used for efficiency of calculations
    smoothed_img = (sum_table[row_indices, upper_bound_indices] - sum_table[row_indices, lower_bound_indices]
                    + img[row_indices, lower_bound_indices]) / \
                   (upper_bound_indices - lower_bound_indices + 1)[:, :, np.newaxis]
    return smoothed_img
