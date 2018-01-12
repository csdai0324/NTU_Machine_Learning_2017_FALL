import os
import sys
import numpy as np
import skimage
from skimage import io

def get_images(images_path):
    files = os.path.join(images_path, '*.jpg')
    images_collection = io.ImageCollection(files)
    images_array = images_collection.concatenate().reshape(415, -1) 
    return images_array

def reconstruct_faces(images_path, target_image, top):
    images_array = get_images(images_path)
    target = io.imread(target_image)
    target = target.reshape(1, -1) 
    mean_img = np.mean(images_array, axis=0)
    X = images_array - mean_img
    U, s, V = np.linalg.svd(X.T, full_matrices=False)

    target_a = target - mean_img  # shape = (1, 1080000)

    weights = np.dot(target_a, U)
    recon = mean_img + np.dot(weights[:, :top], U[:, :top].T)
    recon -= np.min(recon)
    recon /= np.max(recon)
    recon = (recon * 255).astype(np.uint8)
    recon = recon.reshape(600, 600, 3)
    io.imsave('./reconstruction.jpg'.format(target_image), recon)


def main():
    images_path = sys.argv[1]
    target_path = sys.argv[2]
    reconstruct_faces(images_path, target_path, 4)

if __name__ == '__main__':
    main()
