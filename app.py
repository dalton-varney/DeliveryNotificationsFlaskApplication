import time
import edgeiq
from helpers import *
from sample_writer import *
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from websocket import create_connection
import itertools
import json
from collections import deque
from Autoannotate import *
from copy import deepcopy
from collections import defaultdict

app = Flask(__name__, template_folder='./templates/')
socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(app, logger=socketio_logger, engineio_logger=socketio_logger)
SAMPLE_RATE = 25
SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.FileVideoStream("deliveryonsite.m4v", play_realtime=True)
video_stream1 = edgeiq.FileVideoStream("constructionppe.m4v", play_realtime=True)
video_stream2 = edgeiq.FileVideoStream("heroconstructionbb.avi", play_realtime=True)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@socketio.on('write_data')
def write_data():
    controller.start_writer()
    socketio.sleep(0.05)
    controller.update_text('Data Collection Started')

    print('Data Collection Started')

@socketio.on('stop_writing')
def stop_writing():
    print('Stopped Data Collection')
    controller.stop_writer()
    controller.complete_annotations()
    socketio.sleep(0.01)


@socketio.on('close_app')
def close_app():
    print('Closing App...')
    controller.close_writer()
    controller.close()


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    file = os.path.join(".", get_file(filename))
    return send_file(file, as_attachment=True)

@app.route('/videos', methods=['GET'])
def videos():
    videos = {}
    files = get_all_files()
    if files:
        for f in files:
            videos[f] = (os.path.join(os.path.sep, get_file(f)))
    return render_template('videos.html', videos=videos)

@app.route('/view_video/<filename>', methods=['GET'])
def view_video(filename):
    file = os.path.join(os.path.sep, get_file(filename))
    if '.jpeg' in file:
        return render_template('view_video.html', image=file, filename=filename)
    else:
        return render_template('view_video.html', video=file, filename=filename)

@app.route('/delete/<filename>', methods=['GET'])
def delete(filename):
    file = os.path.join(".", get_file(filename))
    if file is not None:
        delete_file(file)
    return redirect(url_for('videos'))

ZONES = [
            'incoming_traffic_cumulative',
            'outgoing_traffic_cumulative',
        ]

class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler, Lila Mullany, and Dalton Varney
        Copyright alwaysAI, Inc. 2022

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self.Ended = False
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.writer = SampleWriter()
        self.all_frames = deque()
        self.video_frames = deque()
        self.auto_annotator = AutoAnnotator(confidence_level=0.5, overlap_threshold=0.3, labels=['Adult', 'Child'], markup_image=False)
        self.alert_text = " "
        self.alert_img = []
        self.alert = "False"
        self.alert_timer = time.time()
        self.alert_flag = False
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        print("Starting Up")

        obj_detect = edgeiq.ObjectDetection(
                "dvarney/DeliveryNotificationsYolo")
        obj_detect.load(engine=edgeiq.Engine.DNN_CUDA)
        zones = edgeiq.ZoneList("zone_config.json")
        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))
        print("Model:\n{}\n".format(obj_detect.model_id))
        print("Labels:\n{}\n".format(obj_detect.labels))
        workers_in_zone_a = []
        workers_in_zone_b = []
        cars_in_zone_a = []
        cars_in_zone_b = []
        Worker_count = 0
        Car_count = 0
        fps = edgeiq.FPS()
        video_stream.start()
        video_stream1.start()
        video_stream2.start()
        self.fps.start()


        def object_enters(person_id, prediction):
            """
            Detects when a new person enters.
            Referenced from https://github.com/alwaysai/snapshot-security-camera
            """
            global new_detection
            new_detection = True

        def object_exits(person_id, prediction):
            """
            Detects when a person exits.
            Referenced from https://github.com/alwaysai/snapshot-security-camera
            """
            global new_detection
            new_detection = False

        #We enable two trackers, one for Cars and one for trucks
        car_tracker = edgeiq.CentroidTracker(
            deregister_frames=30, enter_cb=object_enters, exit_cb=object_exits)
        truck_tracker = edgeiq.KalmanTracker(
            deregister_frames=30,
        max_distance=200, min_inertia = 1, enter_cb=object_enters, exit_cb=object_exits)

        while True:
            #A flag to prevent us from sending constant alerts
            self.alert = "False"
            if (time.time() - self.alert_timer > 50):
                self.alert_flag = False

            #Navigate the 3 video streams we are passing to the application. A bit of a shortcut to prevent errors
            try:
                frame = video_stream.read()
            except:
                frame = video_stream.read()
            try:
                video1 = video_stream1.read()
            except:
                video_stream1.start()
            try:
                video2 = video_stream2.read()
            except:
                video_stream2.start()

            ogframe = deepcopy(frame)
            self.alert_img = deepcopy(frame)

            #The object detection part
            results = obj_detect.detect_objects(frame, confidence_level=.7, overlap_threshold=0.1)
            people = edgeiq.filter_predictions_by_label(results.predictions, ['person'])
            cars = edgeiq.filter_predictions_by_label(results.predictions, ['Truck', 'Car', 'Machinery'])
            trucks = edgeiq.filter_predictions_by_label(results.predictions, ['Truck'])
            frame = edgeiq.blur_objects(frame, people)

            #Tracker for cars or trucks or workers
            tracked_cars = car_tracker.update(cars)
            tracked_trucks = truck_tracker.update(trucks)
            Zone_Entry = zones.get_zone("EntryZone")
            Zone_Exit = zones.get_zone("ExitZone")
            if len(tracked_cars) > 0:
                for key, value in tracked_cars.items():
                    if Zone_Entry.check_object_detection_prediction_within_zone(value.prediction):
                        if key not in workers_in_zone_a:
                            if key in workers_in_zone_b:
                                Worker_count -= 1
                            workers_in_zone_a.append(key)
                    if Zone_Exit.check_object_detection_prediction_within_zone(value.prediction):
                        if key not in workers_in_zone_b:
                            if key in workers_in_zone_a:
                                Worker_count += 1
                            workers_in_zone_b.append(key)
            if len(tracked_trucks) > 0:
                print(len(tracked_trucks))
                for key, value in tracked_trucks.items():
                    if Zone_Entry.check_object_detection_prediction_within_zone(value.prediction):
                        print("object in entry")
                        if key not in cars_in_zone_a:
                            if key in cars_in_zone_b:
                                print("Inside first if")
                                Car_count -= 1
                            cars_in_zone_a.append(key)
                    if Zone_Exit.check_object_detection_prediction_within_zone(value.prediction):
                        print("object in exit")
                        if key not in cars_in_zone_b:
                            if key in cars_in_zone_a:
                                print("Inside second if")
                                Car_count += 1
                            cars_in_zone_b.append(key)
            #Activate alert if there is a truck or car in the delivery zones, we pass the alert through self.trigger_alert
            if not self.alert_flag and Car_count >= 1:
                self.trigger_alert("Delivery Arrived")

            #Markup image and zones
            frame = edgeiq.markup_image(frame, cars, colors=obj_detect.colors, line_thickness=2, font_size=0.7, show_confidences=False)
            frame = edgeiq.markup_image(frame, people, colors=obj_detect.colors, line_thickness=2, font_size=0.7, show_confidences=False)
            frame = zones.markup_image_with_zones(frame, ['EntryZone'], show_labels=False, fill_zones=True, alpha=0.3)
            frame = zones.markup_image_with_zones(frame, ['ExitZone'], show_labels=False, color = (150,150,0), fill_zones=True, alpha=0.3)
            #frame = edgeiq.markup_image(frame, violatedPersons, colors=[(0,0,255)], line_thickness=2, font_size=0.7, show_confidences=False)

            #I moved the resize frames here
            frame = edgeiq.resize(
                    frame, width=960, height=480, keep_scale=False)
            video1 = edgeiq.resize(
                    video1, width=960, height=480, keep_scale=False)
            video2 = edgeiq.resize(
                    video2, width=960, height=480, keep_scale=False)
            self.alert_img = edgeiq.resize(
                    self.alert_img, width=100, height=150, keep_scale=True)
            text = ["Additional Video Feeds"]

            self.all_frames.append(frame)

            #This is for writing the data collection
            if self.writer.write == True:
                self.video_frames.append(ogframe)
                frame2 = deepcopy(frame)
                (annotation_xml, frame2, image_name, annotationText) = self.auto_annotator.annotate(frame2, results.predictions)
                self.auto_annotator.write_image(annotation_xml, frame2, image_name)
                start = time.time()
                self.auto_annotator.image_index += 1
            zone_dictionary = {}
            zone_dictionary['incoming_traffic_cumulative'] = Worker_count
            zone_dictionary['outgoing_traffic_cumulative'] = Car_count
            zone_dictionary['incoming_traffic'] = len(results.predictions)
            zone_dictionary['outgoing_traffic'] = len(Zone_Entry.get_results_for_zone(results).predictions)
            edgeiq.publish_analytics(zone_dictionary)

            #We send the data to the flask application
            self.send_data(frame, video1, video2, text, str(len(trucks)), str(len(cars)))

            self.fps.update()


            if self.Ended:
                break

    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def trigger_alert(self, alert_message):
        """Activates an alert to the webapp

        Args:
            alert message (string): The message to be included with alert

        """
        self.alert_flag = True
        self.alert = "True"
        self.alert_text = "Delivery Arrived " + time.strftime("%H:%M", time.localtime(time.time()))
        self.alert_timer = time.time()

    def send_data(self, frame, video1, video2, text, adult_occupancy,child_occupancy):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'video1': self._convert_image_to_jpeg(video1),
                        'video2': self._convert_image_to_jpeg(video2),
                        'text': '<br />'.join(text),
                        'adult_occupancy': adult_occupancy,
                        'child_occupancy': child_occupancy,
                        'alert': self.alert,
                        'alert_img': self._convert_image_to_jpeg(self.alert_img),
                        'alert_text': self.alert_text
                        #'data': get_all_files()
                    })
            socketio.sleep(0.0001)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.writer.close

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()

class Controller(object):
    def __init__(self):
        self.write = False
        self.currentDatasetName = " "
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('alwaysAI Dashboard on http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.cvclient.Ended = True
        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()
        self.fps.stop()

    def complete_annotations(self):
        self.cvclient.auto_annotator.zip_annotations(self.currentDatasetName)
        self.cvclient.auto_annotator.write_default_file()
        print("Zipped Dataset")

    def close_writer(self):
        self.cvclient.writer.write = False
        self.cvclient.writer.close = True

    def start_writer(self):
        self.currentDatasetName = "DataCollection" + " " + time.ctime(time.time()).replace(":", "-")
        self.cvclient.auto_annotator.make_directory_structure(self.currentDatasetName)
        self.cvclient.writer.write = True

    def stop_writer(self):
        self.cvclient.writer.write = False
        self.cvclient.writer.close = True

    def is_writing(self):
        return self.cvclient.writer.write

    def update_text(self, text):
        self.cvclient.writer.text = text

controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        print("Program Complete - Thanks for using alwaysAI")
        controller.close()
