import numpy as np
import cv2

class ImageMatcher:
    def __init__(self, descriptor, imagePaths, ratio = 0.7, minMatches = 40, useHamming = True):
        self.descriptor = descriptor
        self.imagePaths = imagePaths
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"

        if useHamming:
            self.distanceMethod += "-Hamming"

    def search(self, queryKps, queryDescs, threshold=.7):
        results = {}

        for imagePath in self.imagePaths:
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)

            if queryDescs == None or descs == None:
                continue

            score, matrix = self.match(queryKps, queryDescs, kps, descs)
            results[imagePath] = score, matrix

        if len(results) > 0:
            results = [(v, k) for (k, v) in results.items() if v[0] > threshold]
            results = sorted(results, reverse=True)

        return results

    def match(self, kpsA, featuresA, kpsB, featuresB):
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        if not len(featuresA) or not len(featuresB):
            return -1.0, []

        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > self.minMatches:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
            (M, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            return float(status.sum()) / status.size, M

        return -1.0, []