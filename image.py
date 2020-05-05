import numpy as np
from PIL import Image

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


#image for mask
image = np.array(Image.open(".png"))


transformed_image_mask = np.ndarray((image.shape[0],image.shape[1]), np.int32)

for i in range(len(brasil)):
    transformed_brasil_mask[i] = list(map(transform_format, image[i]))

np.save('mask.npy',transformed_image_mask)