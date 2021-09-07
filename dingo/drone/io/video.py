import threading
import cv2

class VideoRecorder(threading.Thread):
    """Records Video"""
    def __init__(self, driver, **config):
        self.config = config
        self.reader = None
        self.driver = driver

    @property
    def frame(self):
        return self.reader.frame

    def run(self, show=True):
        self.driver.streamon()
        self.reader = self.driver.get_frame_read()
        img = cv2.resize(self.frame, (self.config.width, self.config.height))
        if show:
            cv2.imshow("DingoFrame", img)
        return img

    def close(self):
        self.driver.streamoff()

    def __del__(self):
        self.close()