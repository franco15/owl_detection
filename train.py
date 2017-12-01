import os
import sys
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def train():
	# train and validation directories
	train_dir = 'data/test/resized_imgs/train'
	validation_dir = 'data/test/resized_imgs/validation'

	# total number of samples
	train_samples = 2000
	validation_samples = 800

	epoch = 4

	# model begins
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(MaxPooling2D(2, 2))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(2, 2))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))

	model.add(Dense(2, activation='softmax'))

	# model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# model ends

	train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(150, 150),
		batch_size=32,
		class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(
		validation_dir,
		target_size=(150, 150),
		batch_size=32,
		class_mode='categorical')

	model.fit_generator(
		train_generator,
		samples_per_epoch=train_samples,
		nb_epoch=epoch,
		validation_data=validation_generator,
		nb_val_samples=validation_samples)

	model.save_weights('buhos.h5')


# def resize_images():
# 	if not os.path.exists('data/test/resized_imgs'):
# 		os.makedirs('data/test/resized_imgs')
# 		os.makedirs('data/test/resized_imgs/train/owls')
# 		os.makedirs('data/test/resized_imgs/train/not_owls')
# 		os.makedirs('data/test/resized_imgs/validation/owls')
# 		os.makedirs('data/test/resized_imgs/validation/not_owls')
# 	else:
# 		print ('imgs already resized or check dir')
# 		return
# 	for i in range(0, len(os.listdir('data/test/train/owls'))):
# 		name = os.listdir('data/test/train/owls')[i]
# 		path = 'data/test/train/owls/' + name
# 		img = Image.open(path)
# 		resized = img.resize((150, 150), PIL.Image.BICUBIC)
# 		resized.save('data/test/resized_imgs/train/owls/' + name)
# 	for i in range(0, len(os.listdir('data/test/train/not_owls'))):
# 		name = os.listdir('data/test/train/not_owls')[i]
# 		path = 'data/test/train/not_owls/' + name
# 		img = Image.open(path)
# 		resized = img.resize((150, 150), PIL.Image.BICUBIC)
# 		resized.save('data/test/resized_imgs/train/not_owls/' + name)
# 	for i in range(0, len(os.listdir('data/test/validation/not_owls'))):
# 		name = os.listdir('data/test/validation/not_owls')[i]
# 		path = 'data/test/validation/not_owls/' + name
# 		img = Image.open(path)
# 		resized = img.resize((150, 150), PIL.Image.BICUBIC)
# 		resized.save('data/test/resized_imgs/validation/not_owls/' + name)
# 	for i in range(0, len(os.listdir('data/test/validation/owls'))):
# 		name = os.listdir('data/test/validation/owls')[i]
# 		path = 'data/test/validation/owls/' + name
# 		img = Image.open(path)
# 		resized = img.resize((150, 150), PIL.Image.BICUBIC)
# 		resized.save('data/test/resized_imgs/validation/owls/' + name)
#

if __name__ == "__main__":
	# args = sys.argv
	# if args[1] == 'resize':
	# 	resize_images()
	train()
