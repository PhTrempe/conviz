from scipy.misc import imsave


def save_image(img_data, img_path):
    """
    Saves the given image data to the specified image file path.
    :param img_data: The image data as a numpy array.
    :param img_path: The path to which the image is created.
    """
    imsave(img_path, img_data)
