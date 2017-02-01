import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def rotate(img, angle=10, reshape=True):
    if angle == 0:
        return img
    if reshape:
        img = img.reshape(28, 28)
        return ndimage.rotate(img, angle, reshape=False, prefilter=False).flatten()
    else:
        return ndimage.rotate(img, angle, reshape=False, prefilter=False)

# Source: http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor=1.3, reshape=True, **kwargs):
    if zoom_factor == 1:
        return img
    if reshape:
        img = img.reshape(28, 28)
    h, w = img.shape[:2]
    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))
    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, prefilter=False, **kwargs)
    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, prefilter=False, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    assert out.shape == img.shape
    if reshape:
        return out.flatten()
    return out

def noise_filter(img, noise=0.2):
    return (1 - noise) * np.array(img) + noise * 255 * np.random.randn(*img.shape)

def gaussian_filter(img, sigma=1, reshape=True):
    if reshape:
        img = img.reshape(28, 28)
    img = ndimage.gaussian_filter(img, sigma=sigma)
    if reshape:
        return img.flatten()
    return img

def elastic_transform(image, alpha, sigma, random_state=None, reshape=True):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if reshape:
        image = image.reshape(28, 28)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image = ndimage.interpolation.map_coordinates(image, indices, order=1).reshape(shape)

    if reshape:
        return image.flatten()
    return image
