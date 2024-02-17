import imageio as iio
import numpy
import matplotlib.pyplot as plt

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def main():
    user_image = input("Enter path to image: ")
    
    image = iio.imread(user_image)
    print(image)
    print("normal")
    gray = rgb2gray(image)
    
    print(gray)
    plt.imshow(gray, cmap='gray')
    plt.show()
main()