from tensorflow.keras.models import load_model
import cv2


cracknet_model = load_model('./cracknet_model.h5')
cap = cv2.VideoCapture(0)


def inference(model, image):
    result = model.predict(image, verbose=0)
    return float(result)


while True:
    ret, frame = cap.read()
    if ret:
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (120, 120))
        img_data = img_data.reshape(1, 120, 120, 3).astype('float32')/255.0
        print(inference(model=cracknet_model, image=img_data))
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


