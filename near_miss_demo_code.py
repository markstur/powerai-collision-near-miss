import cv2
import cv2.cv as cv
import subprocess as sp
from label import LabelObject
import requests
import json
import os
import fire
import time
import numpy
import matplotlib.path as mplPath
import uuid
import random
import threading
from threading import Lock

class near_miss():

    # initialize class parameters
    # cls.step : perform a detect every cls.step frames
    # cls.display_idx: currently processing frame
    @classmethod
    def init_class_parameter(cls):
        cls.display_lock = Lock()
        cls.step = 2
        cls.display_idx = cls.step

    # initialize APIs from aivision
    # multiple APIs could be registered and executed in parallel, this would increase the inference speed,
    # while consume more resources
    # NOTE: for video inference, the API response speed depends on the network
    def __init__(self):
        DLAAS_DETECT_API = os.environ.get("DLAAS_DETECT_API", "http://172.17.0.1:9080/powerai-vision/api/dlapis/%(api_id)s")
        self.api_id = []

        # a group of API which would be invoked in round robin
        self.api_id.append(DLAAS_DETECT_API % {"api_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"})
        self.api_id.append(DLAAS_DETECT_API % {"api_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"})
        self.api_id.append(DLAAS_DETECT_API % {"api_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"})
        self.api_id.append(DLAAS_DETECT_API % {"api_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"})

    @classmethod
    def get_current_display_idx(cls):
        return cls.display_idx

    @classmethod
    def increase_display_idx(cls):
        cls.display_lock.acquire()
        try:
            cls.display_idx += cls.step
        finally:
            cls.display_lock.release()

    # detect whether an object is in crossway
    # @param crossway: vertex list of the cross way
    # @param xmin, ymin, xmax, ymax: the bounding box of the detected object

    def obj_in_crossway(self, crossway, xmin, ymin, xmax, ymax):
        bbPath = mplPath.Path(numpy.array(crossway))
        center_point = ((xmin+xmax)/2, (ymin+ymax)/2)
        point_list = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax), center_point]
        for point_item in point_list:
            contain_flag = bbPath.contains_point(point_item)
            if contain_flag:
                return True
        return False

    # crossway is defined in the crossway json file, points in clockwise or anti-clockwise direction
    # example defination as following, each array represents one cross way with 4 vertex
    # [
    #   [[147, 653], [281, 719], [382, 681], [242, 622]],
    #   [[620, 699], [636, 720], [1280, 720], [1280, 665]]
    # ]
    # @param image: matrix of the image extracted from opencv
    # @param vertex_array: vertex array of the cross way
    # @param is_nearmiss: flag showing if there is a near miss event on the crossway
    def draw_crossway(self, image, vertex_array, is_nearmiss):
        overlay = image.copy()

        color = cv.Scalar(255, 0, 0)
        if is_nearmiss:
            color = cv.Scalar(0,0,255)
        pts = numpy.array(vertex_array, numpy.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(overlay, pts, color)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                        0, image)

    # process the person an vehicles that are identified from the video frame
    # detect if there is a near miss event on the crossway
    # detect persons and vehicles that are involved in the near miss event
    # @param item_list: list of persons and vehicles detected
    # @param crossway_array: list of crossways in the frame
    # @return crossway_result: whether there is a near miss scene on the crossway
    # @return items_result: list of items involved in the nearmiss event

    def process_display_candidates(self, items_list, crossway_array):
        crossway_result = []
        items_result = []

        for crossway in crossway_array:
            tmp_crossway_contains = []
            for item in items_list:
                if self.obj_in_crossway(crossway, item[1], item[2], item[3], item[4]):
                    tmp_crossway_contains.append(item[0])
            crossway_contains_set = set(tmp_crossway_contains)
            near_miss_flag = False
            if len(crossway_contains_set) >=2:
                near_miss_flag = True
            crossway_result.append((crossway, near_miss_flag))

        for item in items_list:
            near_miss_flag = False
            for crossway_item in crossway_result:
                (crossway, is_nearmiss) = crossway_item
                if is_nearmiss:
                    if self.obj_in_crossway(crossway, item[1], item[2], item[3], item[4]):
                        near_miss_flag = True
            items_result.append((item, near_miss_flag))

        return crossway_result, items_result

    # invoke api on a frame, detect near miss event
    # @param frame: frame extracted from opencv
    # @param proc: stream process accepting the input frame
    # @param cross_array: position of cross_way (might be multiple crossways)
    # @param frame_id: current processing id of the frame
    def perform_detect(self, frame, proc, crossway_array, frame_id = 0):

            jpg = cv2.imencode(".jpg", frame)[1].tostring()
            tmp_id_key = random.randint(0, 3)
            tmp_api = self.api_id[tmp_id_key]

            headers = {}
            ret = requests.post(tmp_api,
                                headers=headers,
                                files={'imagefile': jpg}
                                )

            result = ret.text
            if 'fail' in result:
                result = []
            else:
                result = json.loads(result)

            classified = result
            items_list = []
            label_object = LabelObject()
            if classified and len(classified):
                detect_objects = classified
                for obj in detect_objects:
                    label = obj.get("label")
                    if label == 'crossway':
                        continue
                    xmin = int(obj.get("xmin"))
                    ymin = int(obj.get("ymin"))
                    xmax = int(obj.get("xmax"))
                    ymax = int(obj.get("ymax"))
                    confidence = round(obj.get("confidence", 0) * 100)
                    items_list.append([label, xmin, ymin, xmax, ymax])

            crossway_result, items_result = self.process_display_candidates(items_list, crossway_array)

            for items_item in items_result:
                (pocar, is_nearmiss) = items_item
                tmp_label = pocar[0]
                color = cv.Scalar(255, 0, 0)
                if tmp_label == "people":
                    color = cv.Scalar(0, 255, 0)
                label_object.add_object(pocar[0], 1, pocar[1], pocar[2], pocar[3], pocar[4], color)

            label_object.draw_label(cv.fromarray(frame))
            for crossway_item in crossway_result:
                (crossway, is_nearmiss) = crossway_item
                self.draw_crossway(frame, crossway, is_nearmiss)

                if is_nearmiss:
                    base_filename = str(uuid.uuid4()) + ".jpg"
                    filename = "/var/www/html/" + base_filename
                    cv2.imwrite(filename=filename, img=frame)

            while True:
                if frame_id - near_miss.display_idx >=near_miss.step*3:
                    near_miss.increase_display_idx()
                if frame_id//near_miss.step == near_miss.display_idx//near_miss.step:
                    near_miss.display_lock.acquire()
                    try:
                        proc.stdin.write(frame.tostring())
                    finally:
                        near_miss.display_lock.release()
                    near_miss.increase_display_idx()
                    break
                if frame_id < near_miss.display_idx:
                    break
                else:
                    time.sleep(0.05)



    # main method for processing a stream, taking a stream as input and process
    # @param video_src: input video stream, rtsp or rtmp protocol
    # @param crossway_file: file containing the crossway vertexes
    # @param run_detect: run detect or directly output original frames
    def process_stream(self, video_src, crossway_file, run_detect = True):
        print "capture video src %s" % video_src
        cap = cv2.VideoCapture(video_src)
        ret, frame = cap.read()
        height, width, ch = frame.shape

        ffmpeg = 'ffmpeg'
        dimension = '{}x{}'.format(width, height)
        fps = str(cap.get(cv2.cv.CV_CAP_PROP_FPS))

        command = [ffmpeg,
                   '-f', 'rawvideo',
                   #'-vcodec','rawvideo',
                   '-s', dimension,
                   '-pix_fmt', 'bgr24',
                   '-i', '-',
                   #'-an',
                   '-b:v', '1000k',
                   '-codec:v', 'mpeg1video',
                   '-f', 'mpegts',
                   '-b:a', '128k',
                   '-bf', '0',
                   '-muxdelay', '0.01',
                   'http://127.0.0.1:9999/video']

        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        headers = {}
        latest_detection_time = int(round(time.time() * 1000))
        crossway_array = []
        with open(crossway_file) as cfp:
            crossway_array = json.load(cfp)

        count = 1
        while True:
            count +=1

            ret, frame = cap.read()
            if not (count % near_miss.step) == 0:
                continue
            for crossway in crossway_array:
                self.draw_crossway(frame, crossway, False)
            if not ret:
                if cap.isOpened():
                    print ("Connection lost, waiting for a second")
                    time.sleep(1)
                    break
                else:
                    cap = cv2.VideoCapture(video_src)
                    time.sleep(5)

            if run_detect:
                detection_thread = threading.Thread(target=self.perform_detect, args=(frame, proc, crossway_array, count, 'no'))
                detection_thread.start()


        cap.release()
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()

def main():
    near_miss.init_class_parameter()
    fire.Fire(near_miss)

if __name__ == "__main__":
    main()

