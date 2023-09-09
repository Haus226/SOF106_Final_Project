import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def imageReader(filename, model="pretrained/mnist_adam.hdf5"):
    img = Image.open(filename).convert('LA')
    array = np.array(img)[:, :, 0]
    array = 255 - array
    divisor = array.shape[0] // 9
    puzzle = []
    for i in range(9):
        row = []
        for j in range(9):
            # Slice image, reshape it to 28x28 (mnist reader size)
            crop = cv2.resize(array[i * divisor: (i + 1) * divisor, j * divisor: (j + 1) * divisor][3:-3, 3:-3],
                              interpolation=cv2.INTER_CUBIC,
                              dsize=(28, 28))
            crop = crop[3:24, 3:24]     # Prevent to include the border inside the image
            crop = cv2.resize(crop, dsize=(28, 28))
            row.append(crop)
        puzzle.append(row)

    model = load_model(model)
    board = []
    for row in puzzle:
        r = []
        for spot in row:
            if np.mean(spot) > 2:
                # If the mean of the 28 * 28 pixels is > 2 --> There is a number inside the crop
                r.append(np.argmax(model.predict(spot.reshape(1, 28, 28, 1).astype("float32") / 255)))
            else:
                r.append(0)
        board.append(r)
    return board

if __name__ == '__main__':
    imageReader("example/a (1).png")
