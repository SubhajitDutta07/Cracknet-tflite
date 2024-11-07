import tflite_runtime.interpreter as tflite
import cv2


interpreter = tflite.Interpreter(model_path='./cracknet_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap = cv2.VideoCapture(0)


def inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()[0]


while True:
    ret, frame = cap.read()
    if ret:
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (120, 120))
        img_data = img_data.reshape(1, 120, 120, 3).astype('float32')/255.0
        print(inference(image=img_data))
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


