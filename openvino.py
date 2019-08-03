import cv2 as cv
import picamera
import picamera.array
import time
# Load the model.
net = cv.dnn.readNet('/home/pi/src/face-detection-adas-0001.xml',
                     '/home/pi/src/face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
# Read an image.
camera = picamera.PiCamera()
stream = picamera.array.PiRGBArray(camera)
# camera.resolution = (672, 384)
camera.brightness = 65
# stream.arrayにRGBの順で映像データを格納
camera.capture(stream, 'bgr', use_video_port=True)

# グレースケールに変換
# frame = cv.cvtColor(stream.array, cv.COLOR_BGR2RGB)
frame = stream.array

stream.seek(0)
stream.truncate()
# frame = cv.imread('/home/pi/src/test.jpg')
# Prepare input blob and perform an inference.
start = time.time()
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()
print("Exec Time:{:.1f}".format(time.time() - start) + "[sec]")
# Draw detected faces on the frame.
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])
    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
# Save the frame to an image file.
cv.imwrite('/home/pi/src/out.png', frame)
