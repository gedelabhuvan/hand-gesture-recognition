import cv2
import numpy as np
from tensorflow.keras.models import load_model
# Load model
model = load_model("keras_model.h5", compile=False)
# Load labels
class_names = open("labels.txt", "r").readlines()
# Open webcam
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret, image = camera.read()

    # Resize image
    image_resized = cv2.resize(image, (224, 224))

    # Normalize image
    image_np = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1

    # Predict
    prediction = model.predict(image_np)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction
    text = f"{class_name.strip()} ({confidence_score:.2f})"

    cv2.putText(image, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", image)

    # Press q to quit
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()