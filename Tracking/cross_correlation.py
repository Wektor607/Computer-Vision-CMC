import numpy as np
import os

from moviepy.editor import VideoFileClip
from skimage.feature import match_template
from skimage.color import rgb2gray

from detection import extract_detections, draw_detections, detection_cast
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-(X - x) ** 2 / dx ** 2 - (Y - y) ** 2 / dy ** 2)

import numpy as np
class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""
    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        frame_gray = rgb2gray(frame)
        prev_frame_gray = rgb2gray(self.prev_frame)
        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame
            prev_bbox = prev_frame_gray[ymin:ymax, xmin:xmax]
            # Step 1: Extract new_bbox from current frame with the same coordinates
            new_bbox = frame_gray[ymin:ymax, xmin:xmax]
            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            matching = match_template(new_bbox, prev_bbox, pad_input = True, mode='edge')
            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            result = matching * gaussian((ymax - ymin, xmax - xmin), (xmax-xmin)//2, (ymax-ymin)//2, ymax-ymin, xmax-xmin)
            y, x = np.unravel_index(np.argmax(result), result.shape)
            # Step 4: Append to detection list
            h, w = ymax - ymin, xmax - xmin
            x1 = min(max(xmin + x - w//2, 0), frame_gray.shape[1] - 1)
            y1 = min(max(ymin + y - h//2, 0), frame_gray.shape[0] - 1)
            y2 = min(max(ymax + y - h // 2, 0), frame.shape[0] - 1)
            x2 = min(max(xmax + x - w // 2, 0), frame.shape[1] - 1)
            detections.append([label, x1, y1, x2, y2])

        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()

