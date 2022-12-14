import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            #  show an update every 'verbose' image
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print(f"[INFO] processed {i+1} / {len(imagePaths)}")

        # rerurn a tuple of the data and labels
        return (np.array(data), np.array(labels))
