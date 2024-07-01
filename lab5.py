import cv2
import argparse
import queue
import time
from threading import Thread
from ultralytics import YOLO

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


def thread_safe_predict(image_path, queue):
    batch_size = 1
    local_model = YOLO('yolov8n-pose.pt')

    for i in range(len(image_path) // batch_size):
        local_model.predict(image_path[i * batch_size:(i + 1) * batch_size], verbose=False, device='cpu')

    if len(image_path) % batch_size != 0:
        local_model.predict(image_path[(len(image_path) // batch_size) * batch_size:len(image_path)], verbose=False, device='cpu')


if __name__ == '__main__':
    if args.single:
        cur_time = time.time()
        model = YOLO('yolov8n-pose.pt')
        model.predict(args.path, save=True, device='cpu')
        last_time = time.time()

        print(last_time - cur_time)
    else:
        cap = cv2.VideoCapture(args.path)

        output_video = cv2.VideoWriter(args.name, fourcc, fps, (frame_width, frame_height))

        giga_out = []
        queues = []

        for i in range(args.count):
            items_per_thread = count_frames // args.count
            items_per_thread_mod = count_frames % args.count

            if items_per_thread_mod > i:
                lb = i * items_per_thread + i
                ub = (i + 1) * items_per_thread + i
            else:
                lb = i * items_per_thread + items_per_thread_mod
                ub = (i + 1) * items_per_thread + items_per_thread_mod - 1

            out = []

            for _ in range(ub - lb + 1):
                res, frame = cap.read()
                out.append(frame)

            giga_out.append(out)
            queues.append(queue.Queue())

        thread = []

        for i in range(args.count):
            thread.append(Thread(target=thread_safe_predict, args=(giga_out[i], queues[i],)))

        cur_time = time.time()
        for i in range(args.count):
            thread[i].start()

        for i in range(args.count):
            thread[i].join()

        last_time = time.time()

        print(last_time - cur_time)

        for i in range(args.count):
            while not queues[i].empty():
                output_video.write(queues[i].get())

        output_video.release()
