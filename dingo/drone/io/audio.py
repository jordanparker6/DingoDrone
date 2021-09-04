from typing import Optional
import logging
from datetime import datetime
from box import Box
import numpy as np
import sounddevice as sd

from utils import IterableQueue

log = logging.getLogger(__name__)

class AudioRecorder:
    """Records audio in realtime and pushes to queue for stream processing"""
    def __init__(self,
            samplerate: Optional[int] = 16000,
            blocksize: int = 2048, 
            threshold: float = 0.005
        ):
        self.config = Box({
            "input": sd.query_devices(sd.default.device, 'input'),
            "ouput": sd.query_devices(sd.default.device, 'output'),
            "silence": {
                "threshold": threshold,
                "min_window": 1
            },
            "blocksize": sd.default.blocksize,
        })
        if samplerate is None:
            samplerate = self.config.input.default_samplerate
        self.config.samplerate = samplerate
        self.config.channels = self.config.input.max_input_channels

        self.audio = IterableQueue()
        self.stream = sd.InputStream(
            samplerate=self.config.samplerate,
            blocksize=blocksize, 
            callback=self.callback
        )
        self.time = Box({
            "start": None,
            "end": None
        })
        self._inbuffer = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        log.info("AudioRecorder starting...")
        self.stream.start()
        self.time.start = datetime.now()
        self.time.end = None
    
    def stop(self):
        self.stream.stop()
        self.time.end = datetime.now()
        timedelta = (self.time.end - self.time.start).total_seconds()
        log.info(f"AudioRecorder ending: {timedelta} seconds recorded")

    def close(self):
        self.stop()
        self.stream.close()

    def callback(self, indata, frames, time, status):
        """Calback to detect audio above a silence threshold"""
        if status:
            print(status, file=sys.stderr)
        now = datetime.now()
        if self._inbuffer is None:
            if not self.is_silent(indata):
                initial_silence_padding = np.zeros((int(0.5 * self.config.samplerate), self.config.channels))
                self._inbuffer = { 
                    "speech": np.concatenate((initial_silence_padding, indata.copy()), axis=0),
                    "starttime": now,
                    "lasttime": now,
                    "endtime": None
                }
        else:
            timedelta = (now - self._inbuffer["lasttime"]).total_seconds()
            if self.is_silent(indata):
                if timedelta <= self.config.silence.min_window:
                    self._inbuffer["speech"] = np.concatenate((self._inbuffer["speech"], indata), axis=0) 
                else:
                    self._inbuffer["endtime"] = now
                    del self._inbuffer["lasttime"]
                    self.audio.put_nowait(self._inbuffer)
                    self._inbuffer = None
            else:
                self._inbuffer["speech"] = np.concatenate((self._inbuffer["speech"], indata), axis=0)  
                self._inbuffer["lasttime"] = now 

    def is_silent(self, data):
        return np.abs(data).mean() <= self.config.silence.threshold


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~ DEV & TEST Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plot(self, frames: int = 512, interval: int = 30):
        """A realtime plot of the incoming input signals for dev/test"""
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plotdata = np.zeros((frames, self.config.channels))
        lines = ax.plot(plotdata)
        if self.config.channels > 1:
            ax.legend(['channel {}'.format(c) for c in self.config.channels], loc='lower left', ncol=self.config.channels)
        ax.axis((0, len(plotdata), -0.1, 0.1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        fig.tight_layout(pad=0)

        def update_plot(frame, plotdata):
            for data in self.audio:
                data = data["speech"]
                shift = len(data)
                plotdata = np.roll(plotdata, -shift, axis=0)
                plotdata[-shift:, :] = data
            for column, line in enumerate(lines):
                line.set_ydata(plotdata[:, column])
            return lines

        ani = FuncAnimation(fig, lambda frame: update_plot(frame, plotdata), interval=interval, blit=True)

        with self:
            plt.show()

    def play(self, data):
        try:
            sd.play(data, self.config.samplerate)
            status = sd.wait()
        except KeyboardInterrupt:
            parser.exit('\nInterrupted by user')

    def playall(self):
        log.info("AudioRecorder Playback")
        for data in self.audio:
            print(data["speech"].shape)
            self.play(data["speech"])