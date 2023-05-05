import tensorflow as tf
import numpy as np

model = None
output_class = ["bags", "metal", "plastic", "trash"]
data = {'0':'bags', '1':'metal', '2':'plastic', '3':'trash'}



def load_artifacts():
    global model
    model = tf.keras.models.load_model("C:/Users/shahr/Downloads/modelVGG2.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	load_artifacts()
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	print(predicted_value)
	return predicted_value
	# print(predicted_array.argmax())
	# return predicted_array

#classify_waste('C:/Users/shahr/NEW_TRY/awareness-of-waste-recycling/metal404.jpg')