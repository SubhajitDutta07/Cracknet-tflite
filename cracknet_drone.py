import cv2
import binascii
import socket
from io import BytesIO
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='./cracknet_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# IP and port of the drone, possible to sniff via wireshark of packet sniffer for the phone app
video_address = ("192.168.4.153", 8080)
video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Use your IP here
video.bind(("192.168.4.100", 19797))

# The message 'Bv' will start the video stream from the drone
video_start_message = str.encode('Bv')
video.sendto(video_start_message, video_address)


# The message 'By' will change the video input source
# switch_camera_message = str.encode('By')
# video.sendto(switch_camera_message, video_address)

def inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()[0]


def camera_feed():
    packet_data = bytearray()

    while True:
        data, server = video.recvfrom(2048)

        # The first 8 bytes will crash the video output, since they are not part of the jpeg image.
        # No use for them in this code yet.
        data = data.hex()[16:]
        end_of_pic = data.find("ffffffd9")

        # Same for the last 5 bytes after the end of the jpeg
        if end_of_pic > 0:
            data = data[:end_of_pic + 8]

        data = binascii.a2b_hex(data)
        packet_data += bytearray(data)

        if end_of_pic > 0:

            # If the package is not complete, drop it
            if not packet_data.hex()[0:4] == "ffd8" and not packet_data.hex()[-8:] == "ffffffd9":
                packet_data = bytearray()
            else:
                # Initialize ByteIO buffer class
                buf = BytesIO(packet_data)

                # Try/catch is important here, since the image could still be defective.
                # This leads to a crash in the program.
                try:
                    img = np.array(Image.open(buf))
                    img_data = cv2.resize(img, (120, 120))
                    img_data = img_data.reshape(1, 120, 120, 3).astype('float32') / 255.0
                    print(inference(image=img_data))

                except Exception as e:
                    print(e)
                    pass

                # Close the buffer and empty the package variable
                buf.close()
                packet_data = bytearray()

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                continue


try:
    camera_feed()

finally:
    video.close()
