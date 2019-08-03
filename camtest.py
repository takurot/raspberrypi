# coding: utf-8
import picamera
import picamera.array
import cv2

# cascade_file = 'haarcascade_frontalface_default.xml'
cascade_file = 'haarcascade_frontalcatface_extended.xml'
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (320, 240)
        while True:
            # stream.arrayにRGBの順で映像データを格納
            camera.capture(stream, 'bgr', use_video_port=True)

            # グレースケールに変換
            gray = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

            # カスケードファイルを利用して顔の位置を見つける
            cascade = cv2.CascadeClassifier(cascade_file)
            # cascade = cv2.CascadeClassifier('/home/pi/src/haarcascade_frontalface_default.xml')
            face_list = cascade.detectMultiScale(gray, minSize=(100, 100))
            # face_list = cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face_list:
                print("face_position:",x, y, w, h)
                color = (0, 0, 255)
                pen_w = 5
                cv2.rectangle(stream.array, (x, y), (x+w, y+h), color, thickness = pen_w)
            # system.arrayをウィンドウに表示
            cv2.imshow('frame', stream.array)

            # "q"でウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # streamをリセット
            stream.seek(0)
            stream.truncate()
        cv2.destroyAllWindows()
