from djitellopy import Tello
from drone.io.video import VideoRecorder

if __name__ == "__main__"
    driver = Tello()
    driver.connect()
    recorder = VideoRecorder(driver, height=300, width=300)
    recorder.start()