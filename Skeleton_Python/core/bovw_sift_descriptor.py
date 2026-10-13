"""
Efficient Bag of Visual Words (BoVW) + SIFT Descriptor Extraction
- Uses OpenCV SIFT
- Builds visual vocabulary using MiniBatchKMeans for speed
- Computes BoVW histograms for all images
"""
import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

class BoVWSIFTDescriptor:
    def __init__(self, n_words=100, batch_size=1000, sift_nfeatures=200):
        self.n_words = n_words
        self.batch_size = batch_size
        self.sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
        self.kmeans = None
        self.vocab = None

    def extract_sift_features(self, image_paths):
        descriptors_list = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
        return descriptors_list

    def build_vocabulary(self, descriptors_list):
        all_desc = np.vstack(descriptors_list)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_words, batch_size=self.batch_size, random_state=42)
        self.kmeans.fit(all_desc)
        self.vocab = self.kmeans.cluster_centers_

    def compute_bovw_histograms(self, image_paths):
        histograms = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is None:
                hist = np.zeros(self.n_words)
            else:
                words = self.kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(self.n_words+1))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-6)
            histograms.append(hist)
        return np.array(histograms)

    def fit_transform(self, image_paths):
        descriptors_list = self.extract_sift_features(image_paths)
        self.build_vocabulary(descriptors_list)
        return self.compute_bovw_histograms(image_paths)
