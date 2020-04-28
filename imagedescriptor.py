import numpy as np
import cv2

class ImageDescriptor:
    def __init__(self, useSIFT = False):
        self.useSIFT = useSIFT

    def describe(self, image):
        descriptor = cv2.BRISK_create()
        print("CREATED")

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        return (kps, descs)