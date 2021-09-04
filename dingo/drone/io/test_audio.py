import time
from drone.io.audio import AudioRecorder

if __name__ == "__main__":   
    recorder = AudioRecorder()
    recorder.start()
    time.sleep(10)
    recorder.stop()
    recorder.playall()
    #recorder.plot()