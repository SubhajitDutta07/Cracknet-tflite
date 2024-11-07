import gradio as gr
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np


interpreter = tflite.Interpreter(model_path='./cracknet_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def inference(image):
    img_data = cv2.resize(np.asarray(image), (120, 120))
    img_data = img_data.reshape(1, 120, 120, 3).astype('float32') / 255.0
    interpreter.set_tensor(input_details[0]['index'], img_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return 'Crack detected' if (output_data.flatten()[0]) > 0.5 else 'No crack detected'


demo = gr.Interface(fn=inference,
                    inputs=gr.Image(type="pil"),
                    outputs=gr.Textbox(),
                    )

demo.launch(server_name='0.0.0.0', server_port=7860)
