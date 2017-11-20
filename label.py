import cv2
import cv2.cv as cv
import logging


LOG = logging.getLogger(__name__)


class LabelObject(object):
    def _random_color(self, name):
        """
        Return a random color
        """
        icolor = abs(hash(name)) % 0xFFFFFF
        return cv.Scalar(icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff)

    def __init__(self):
        self.objects = []

    def add_object(self, name, confidence, xmin, ymin, xmax, ymax, color=None):
        if color == None:
            color = self._random_color(name)
        filtered_object = filter(lambda tmp_obj: tmp_obj["name"] == name, self.objects)
        if not len(filtered_object):
            self.objects.append({
                "name": name,
                "color": color,
                "positions": [{
                    "confidence": confidence,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }]
            })
        else:
            filtered_object = filtered_object[0]
            index = self.objects.index(filtered_object)
            positions = filtered_object.get("positions", "")
            positions.append({
                "confidence": confidence,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })
            filtered_object["positions"] = positions
            self.objects[index] = filtered_object




    def draw_label(self, image):
        for detect_object in self.objects:
            name = detect_object.get("name", "")
            color = detect_object.get("color", "")
            font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0.0, 2, cv.CV_AA)
            text_size, ymin = cv.GetTextSize(name, font)
            for position in detect_object.get("positions"):
                # confidence = position.get("confidence", 0)
                # name = "%s: %d" % (name, confidence)
                # name += "%"
                xmin = position.get("xmin", "")
                xmax = position.get("xmax", "")
                ymin = position.get("ymin", "")
                ymax = position.get("ymax", "")
                cv2.cv.Rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
                cv2.cv.Rectangle(image, (xmin, ymin - text_size[1] - 10),
                             (xmin + text_size[0], ymin), color, cv.CV_FILLED)
                cv2.cv.PutText(image, name, (xmin, ymin-10), font, (0, 0, 0))
