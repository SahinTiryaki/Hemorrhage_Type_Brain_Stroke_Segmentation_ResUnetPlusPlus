from tensorflow.keras.utils import Sequence
import numpy as np
import cv2

class DataGen(Sequence):
    def __init__(self, image_size, images_path, masks_path, batch_size=5):
        self.image_size = image_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.on_epoch_end()

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index*self.batch_size

        images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
        masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        masks_batch = []

        for i in range(len(images_path)):
            ## Read image and mask
            image = cv2.imread(images_path[i],1)
            image = cv2.resize(image,(512,512))
            image[(image==(255,255,255)).all(axis=2)] = (0,0,0)
            image = image/255
            mask = cv2.imread(masks_path[i],0)
            mask = cv2.resize(mask,(512,512))
            mask = np.expand_dims(mask, -1)
            mask = mask/2
            images_batch.append(image)
            mask = np.array(mask,dtype = "float")
            
            masks_batch.append(mask)

        return np.array(images_batch), np.array(masks_batch)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))