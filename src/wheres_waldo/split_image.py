import numpy as np
import matplotlib.pyplot as plt
import matplotlib

BLOCK_SIZE = 256


def split_image(image: np.ndarray):
    image_size = np.shape(image)
    padded_width = image_size[0] // BLOCK_SIZE + 1
    padded_height = image_size[1] // BLOCK_SIZE + 1
    padded_image = np.pad(
        image, (padded_width - image_size[0], padded_height - image_size[1])
    )
    padded_image_sectors = []
    for x in range(padded_width // BLOCK_SIZE):
        padded_image_sectors.append([])
        for y in range(padded_height // BLOCK_SIZE):
            padded_image_sectors[-1].append(
                padded_image[
                    x * BLOCK_SIZE : (x + 1) * BLOCK_SIZE,
                    y * BLOCK_SIZE : (y + 1) * BLOCK_SIZE,
                ]
            )

    return padded_image_sectors, image_size


if __name__ == "__main__":
    sample = []
    for i in range(10):
        sample.append([i] * BLOCK_SIZE)
    img = np.array(sample)
    plt.imshow(img, cmap="gray")
    plt.show()
