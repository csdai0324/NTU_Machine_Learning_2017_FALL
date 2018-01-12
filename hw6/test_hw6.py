import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.cluster import KMeans

def main():
	image_npy_path = sys.argv[1]
	test_case_path = sys.argv[2]
	predict_path = sys.argv[3]
	X = np.load(image_npy_path)
	encoder = load_model('./encoder2.model')
	encoded_imgs = encoder.predict(X)
	encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
	f = pd.read_csv(test_case_path)
	IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
	o = open(predict_path, 'w')
	o.write("ID,Ans\n")
	for idx, i1, i2 in zip(IDs, idx1, idx2):
	    p1 = kmeans.labels_[i1]
	    p2 = kmeans.labels_[i2]
	    if p1 == p2:
	        pred = 1  # two images in same cluster
	    else: 
	        pred = 0  # two images not in same cluster
	    o.write("{},{}\n".format(idx, pred))
	o.close()

if __name__ == '__main__':
	main()