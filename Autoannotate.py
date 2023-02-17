import os
import shutil
from typing import List
import zipfile

import cv2
import edgeiq
from lxml import etree
from lxml.builder import E
import numpy


class AutoAnnotator(object):
    def __init__(self, confidence_level: float,
                 overlap_threshold: float, labels: List[str], markup_image: bool) -> None:

        self.image_index = 0
        self.confidence_level = confidence_level
        self.overlap_threshold = overlap_threshold
        self.labels = labels
        self.images_path = None
        self.annotations_path = None
        self.image_sets_path = None
        self.markup_image = markup_image

    def annotate(self, inputFrame, results):
        frame = inputFrame
        image_name = f'{self.image_index:0>7d}.png'  # Doesn't work on videos which product more than 10M images
        predictions = edgeiq.filter_predictions_by_label(results, self.labels)
        annotation_xml = self._annotate_frame(frame, image_name, predictions)
        text = []
        text.append(f"Image: {image_name}")
        text.append("Annotated Objects: ")
        for prediction in predictions:
            text.append(f"{prediction.label}")
        return (annotation_xml, frame, image_name, text)

    @staticmethod
    def _annotate_frame(frame: numpy.ndarray, image_name: str,
                        predictions: List[edgeiq.ObjectDetectionPrediction]) -> etree._ElementTree:
        annotation_element = (
            E.annotation(
                E.filename(image_name),
                E.size(
                    E.width(str(frame.shape[1])),
                    E.height(str(frame.shape[0])),
                    E.depth(str(3)),
                )
            )
        )
        for prediction in predictions:
            object_xml = (
                E.object(
                    E.name(prediction.label),
                    E.bndbox(
                        E.xmin(str(prediction.box.start_x)),
                        E.ymin(str(prediction.box.start_y)),
                        E.xmax(str(prediction.box.end_x)),
                        E.ymax(str(prediction.box.end_y)),
                    )
                )
            )
            annotation_element.append(object_xml)
        return etree.ElementTree(annotation_element)

    def make_directory_structure(self, dataset_name: str) -> None:
        if os.path.exists(dataset_name):
            shutil.rmtree(dataset_name)
        os.mkdir(dataset_name)
        annotations_path = os.path.join(dataset_name, 'Annotations')
        os.mkdir(annotations_path)
        images_path = os.path.join(dataset_name, 'JPEGImages')
        os.mkdir(images_path)
        image_sets_path = os.path.join(dataset_name, 'ImageSets', 'Main')
        os.makedirs(image_sets_path)
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.image_sets_path = image_sets_path

    def write_image(self, annotation_xml: etree._ElementTree, frame: numpy.ndarray, image_name: str):
        if self.images_path is None:
            raise FileNotFoundError('Images directory does not exist')
        if self.annotations_path is None:
            raise FileNotFoundError('Annotations directory does not exist')
        cv2.imwrite(os.path.join(self.images_path, image_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        annotation_xml.write(os.path.join(self.annotations_path, f'{self.image_index:0>7d}.xml'), pretty_print=True)

    def write_default_file(self):
        if self.image_sets_path is None:
            raise FileNotFoundError('ImageSets directory does not exist')
        try:
            with open(os.path.join(self.image_sets_path, 'default.txt'), 'w') as default_file:
                default_file.write('\n'.join([f'{i:0>7d}' for i in range(0, self.image_index)]))
        except FileNotFoundError:
            print("Default file not found, moving on anyway")

    def zip_annotations(self, dataset_name):
        if self.image_sets_path is None:
            raise FileNotFoundError('ImageSets directory does not exist')
        if self.annotations_path is None:
            raise FileNotFoundError('Annotations directory does not exist')
        zip_file = zipfile.ZipFile(f'{dataset_name}.zip', mode='w')

        for root, dirs, files in os.walk(self.annotations_path):
            for file in files:
                zip_file.write(os.path.join(root, file))
        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                zip_file.write(os.path.join(root, file))
        for root, dirs, files in os.walk(self.image_sets_path):
            for file in files:
                zip_file.write(os.path.join(root, file))
        zip_file.close()
