import tflite_runtime.interpreter as tflite
import glob
import cv2
import tqdm


interpreter = tflite.Interpreter(model_path='./cracknet_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_paths = glob.glob('./dataset/Positive/*')
incorrect_count = 0
correct_count = 0

def inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()[0]


for img_path in tqdm.tqdm(img_paths):
    frame = cv2.imread(img_path)
    try:
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (120, 120))
        img_data = img_data.reshape(1, 120, 120, 3).astype('float32')/255.0
        result = inference(image=img_data)
        if result > 0.5:
            correct_count += 1
        elif result <= 0.5:
            incorrect_count += 1
    except Exception as e:
        pass

print(f'Total: {len(img_paths)}. Correct: {correct_count}, Incorrect: {incorrect_count}.')
