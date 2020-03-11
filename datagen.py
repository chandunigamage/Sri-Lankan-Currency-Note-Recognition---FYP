from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('path to image to be augmented')  # this is a PIL image
print(img.size)

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='path to save augmented images', save_prefix='curreny_note', save_format='jpg'):
    i += 1
    if i > 30:
        break  # otherwise the generator would loop indefinitely
        

