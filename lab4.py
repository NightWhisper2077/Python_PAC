import queue
import time
import threading
import cv2
import keyboard


class Sensor:
    def get(self):
        raise NotImplementedError('Subclasses must implement method get()')


class SensorCam(Sensor):
    def __init__(self, name, resol):
        self._name = name
        self._resol = resol
        self._cap = cv2.VideoCapture(0)

    def get(self):
        res, frame = self._cap.read()

        frame = cv2.resize(frame, self._resol)

        if res is False:
            return

        return frame

    def __del__(self):
        self._cap.release()


class WindowImage:
    def __init__(self, fps):
        self._tbf = int(1000 / fps)
        self._name = 'image'

    def show(self, img):
        cv2.waitKey(1)

        cv2.imshow(self._name, img)

    def __del__(self):
        cv2.destroyAllWindows()


class SensorX(Sensor):
    def __init__(self, delay):
        self._delay = delay
        self._data = 0

    def get(self):
        time.sleep(self._delay)
        self._data += 1

        return self._data


def loop(sensor, queue, event_stop):
    while not event_stop.is_set():
        a = sensor.get()

        if not queue.empty():
            queue.get()

        queue.put(a)


def main():
    fps = 60

    cam = SensorCam(0, [1280, 720])
    window = WindowImage(fps)

    rate0 = 0
    rate1 = 0
    rate2 = 0
    rate3 = None

    sensor0 = SensorX(1)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(0.01)

    queue0 = queue.Queue()
    queue1 = queue.Queue()
    queue2 = queue.Queue()
    queue3 = queue.Queue()

    event_stop = threading.Event()

    sens2 = threading.Thread(target=loop, args=(sensor2, queue2, event_stop))
    sens1 = threading.Thread(target=loop, args=(sensor1, queue1, event_stop))
    sens0 = threading.Thread(target=loop, args=(sensor0, queue0, event_stop))
    sens_cam = threading.Thread(target=loop, args=(cam, queue3, event_stop))

    sens0.start()
    sens1.start()
    sens2.start()
    sens_cam.start()

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = (1100, 650)

    fontScale = 0.5

    color = (0, 0, 0)

    thickness = 1

    while True:
        while not queue3.empty():
            rate3 = queue3.get()
            queue3.task_done()

        while not queue0.empty():
            rate0 = queue0.get()
            queue0.task_done()

        while not queue1.empty():
            rate1 = queue1.get()
            queue1.task_done()

        while not queue2.empty():
            rate2 = queue2.get()
            queue2.task_done()

        if rate3 is None:
            continue

        cv2.rectangle(rate3, (1080, 620), (1280, 720), (255, 255, 255), thickness=-1)

        frame = cv2.putText(rate3, f'Sensor0: {rate0}', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        frame = cv2.putText(frame, f'Sensor1: {rate1}', (org[0], org[1] + 25), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        frame = cv2.putText(frame, f'Sensor2: {rate2}', (org[0], org[1] + 50), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        window.show(frame)

        if keyboard.is_pressed('q'):
            event_stop.set()
            cam.__del__()
            window.__del__()

            sens0.join()
            sens1.join()
            sens2.join()
            break


if __name__ == '__main__':
    main()
