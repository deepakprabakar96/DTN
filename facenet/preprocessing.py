import numpy as np
from skimage.transform import resize


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def align_images(cascade, img, margin=10, image_size=160):
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    (x, y, w, h) = faces[0]
    cropped = img[y-margin//2:y+h+margin//2, x-margin//2:x+w+margin//2, :]
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    return aligned


if __name__ == "__main__":
    # Step 1
    image = "read_image_here"

    # Step 2
    aligned_image = align_images(image)

    # Step 3: Convert image to RGB from BGR if needed before passing it to the model
