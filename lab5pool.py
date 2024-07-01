import cv2
import argparse
import os
import time
from ultralytics import YOLO
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('-p', '--path')
parser.add_argument('-s', '--single', type=bool)
parser.add_argument('-n', '--name')
parser.add_argument('-c', '--count', type=int)
args = parser.parse_args()

cap = cv2.VideoCapture(args.path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

time1 = time.time()


def worker(input_queue, output_queue):
    model = YOLO('yolov8n-pose.pt')

    while True:
        input = input_queue.get()

        # print(f'b: {input[1]} - {time.time()}')

        while not input_queue.empty():
            input = input_queue.get()

        result = model.predict(input[0], verbose=False)

        frame = result[0].plot()
        output_queue.put([frame, input[1]])


if __name__ == '__main__':
    if args.single:
        cap = cv2.VideoCapture(0)

        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        num_processes = args.count

        last_ind = -1

        a = {}
        b = {}

        with Pool(num_processes, worker, (input_queue, output_queue)) as pool:
            count = 0

            while True:
                res, frame = cap.read()

                a[count] = time.time() - time1

                if not res:
                    break

                input_queue.put([frame, count])

                # print(f'a: {count} - {time.time()}')

                count += 1

                if not output_queue.empty():
                    result = output_queue.get()

                    while not output_queue.empty():
                        result = output_queue.get()

                    if result[1] > last_ind:
                        last_ind = result[1]
                        cv2.imshow('bebra', result[0])

                        b[result[1]] = time.time() - time1

                cv2.waitKey(100)

        print('=====')
        print(a)
        print('-----')
        print(b)
    else:
        cap = cv2.VideoCapture(args.path)

        input_queue = multiprocessing.Queue(maxsize=10)
        output_queue = multiprocessing.Queue(maxsize=10)

        num_processes = args.count

        with Pool(num_processes, worker, (input_queue, output_queue)) as pool:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                input_queue.put(frame)

                if not output_queue.empty():
                    result = output_queue.get()

                    cv2.imshow('bebra', result)

                cv2.waitKey(100)
