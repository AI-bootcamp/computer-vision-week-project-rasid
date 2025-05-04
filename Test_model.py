import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model('Model/classfaction_model.h5')

def preprocess_image(image_path, target_size=(224, 224)):
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    
    # Rescale to [0, 1] (same as during training)
    img = img / 255.0  
    
    # Add batch dimension (if model expects it)
    img = np.expand_dims(img, axis=0)
    
    return img

# Preprocess and predict
processed_img = preprocess_image("80_1.jpg")
predictions = model.predict(processed_img)
predicted_class = np.argmax(predictions, axis=1)[0]
print("Predicted Class Index:", predicted_class)
print(predictions)