import sys
from typing import Optional
import logging
from datetime import datetime
from box import Box
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal
import os

from utils import IterableQueue

log = logging.getLogger(__name__)

SAMPLERATE = 16000
BLOCKSIZE = 512
SAMPLE_NOISE_PROFILE = "./dingo/assets/sample_background_noise.noise-profile"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def butter_filter(data, samplerate, pass_freq, stop_freq, btype="band"):
    """Removes the background noise using a butterworth filter"""
    wp = 2 * np.pi * pass_freq
    ws = 2 * np.pi * stop_freq
    Rp = 3
    Rs = 80
    [order, cutoff] = signal.buttord(wp, ws, Rp, Rs, fs=samplerate)
    sos = signal.butter(order, cutoff, btype=btype, output="sos", fs=samplerate)
    return signal.sosfiltfilt(sos, data.flatten()).reshape(-1, 1)

def wiener_filter(data, samplerate, mysize = 64, noise = None):
    """Removes the background noise uisng a wiener filter"""
    return signal.wiener(data.flatten(), mysize=mysize, noise=noise).reshape(-1, 1)

def fft(data, samplerate):
    """Fast Fourier Transform to PSD"""
    data = data.flatten()
    n = len(data)
    dt = 1 / samplerate
    fhat = np.fft.fft(data, n)
    psd = fhat * np.conj(fhat) / n             # magnitude of each fourier transformed
    freq = (1/(dt * n)) * np.arange(n)          # create x axis of increasing frequencies
    return freq, psd.real, fhat

def fft_filter(
        data, 
        samplerate, 
        low_pass: float = None, 
        high_pass: float = None, 
        psd_threshold: float = None, 
        amount: float = 1
    ):
    """Filter signals by psd or freq after fft
    
    Args:
        data: np.ndarray                       The incoming signal
        samplerate: int                        The sample rate of the incoming signal
        low_pass:  Optional[float]             Exclude all freqencies above
        high_pass:  Optional[float]            Exclude all freqencies below
        psd_threshold: Optional[float]         The minimum psd threshold
        ammount: float                         The attenuation percentage
    
    Returns: np.ndarray
    """
    nyq = int(np.floor(data.shape[0] / 2))
    freq, psd, fhat = fft(data, samplerate)
    freq = freq[:nyq]
    if low_pass is not None:
        i = freq < low_pass
        i = np.concatenate((i, np.flip(i))) 
        fhat = fhat * (i + (1 - amount) * np.invert(i))
    if high_pass is not None:
        i = freq > high_pass
        i = np.concatenate((i, np.flip(i)))  
        fhat = fhat * (i + (1 - amount) * np.invert(i))
    if psd_threshold is not None:
        i = psd > psd_threshold
        fhat = fhat * (i + (1 - amount) * np.invert(i))

    return np.fft.ifft(fhat).real.reshape(-1, 1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~ Audio Recorder Class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AudioRecorder:
    """Records audio in realtime and pushes to queue for stream processing"""
    def __init__(self,
            samplerate: Optional[int] = SAMPLERATE,
            blocksize: int = BLOCKSIZE, 
            threshold: float = 0.02
        ):
        self.config = Box({
            "input": sd.query_devices(sd.default.device, 'input'),
            "ouput": sd.query_devices(sd.default.device, 'output'),
            "silence": {
                "threshold": threshold,
                "min_window": 1.5
            }
        })
        if samplerate is None:
            samplerate = self.config.input.default_samplerate
        if blocksize is None:
            blocksize = sd.default.blocksize
        self.config.samplerate = samplerate
        self.config.blocksize = blocksize
        self.config.channels = self.config.input.max_input_channels

        self.audio = IterableQueue()
        self.stream = sd.InputStream(
            samplerate=self.config.samplerate,
            blocksize=self.config.blocksize, 
            callback=self.callback
        )
        self.time = Box({
            "start": None,
            "end": None
        })
        self._inbuffer = None

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
        indata = indata.copy()
        if self._inbuffer is None:
            if not self.is_silent(indata):
                initial_silence_padding = np.zeros((int(0.5 * self.config.samplerate), self.config.channels))
                self._inbuffer = { 
                    "speech": np.concatenate((initial_silence_padding, indata), axis=0),
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
                    self._inbuffer["speech"] = self.remove_background_noise(
                            self._inbuffer["speech"], 
                            self.config.samplerate
                        )
                    self._inbuffer["endtime"] = now
                    del self._inbuffer["lasttime"]
                    self.audio.put_nowait(self._inbuffer)
                    self._inbuffer = None
            else:
                self._inbuffer["speech"] = np.concatenate((self._inbuffer["speech"], indata), axis=0)  
                self._inbuffer["lasttime"] = now 

    def is_silent(self, data):
        data = self.remove_background_noise(data, self.config.samplerate)
        silent = np.abs(data).mean() <= self.config.silence.threshold
        if not silent:
            log.info(f"The silence thrshold has been passed")
        return silent

    def remove_background_noise(self, data, samplerate):
        assert data.shape[1] == 1
        data = fft_filter(data, samplerate, low_pass=2500, high_pass=250, psd_threshold=0.002, amount=0.8)
        return data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~ DEV & TEST Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plot(self, interval: int = 30):
        """A realtime plot of the incoming input signals for dev/test"""
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        frames = self.config.blocksize
        threshold = self.config.silence.threshold
        channels = self.config.channels
        samplerate = self.config.samplerate
        audio = IterableQueue()

        # Create a new stream
        def callback(indata, frames, time, status):
            audio.put(indata.copy())

        stream = sd.InputStream(
            samplerate=samplerate,
            callback=callback,
            blocksize=frames
        )

        # Draw the plot
        fig, ax = plt.subplots(2)
        fig.tight_layout(pad=0)

        # Draw Time Domain Plot
        timeplot = np.zeros((frames, channels * 2))
        timelines = ax[0].plot(timeplot)
        timelines[0].set_label("raw_audio")
        timelines[1].set_label("filtered_audio")
        if channels > 1:
            ax[0].legend(['channel {}'.format(c) for c in channels], loc='lower left', ncol=channels)
        ax[0].axis((0, len(timeplot), -0.1, 0.1))
        ax[0].set_yticks([0])
        ax[0].yaxis.grid(True)
        ax[0].plot([0, frames], [threshold] * 2, color="red", label="silence_threshold")
        ax[0].plot([0, frames], [-threshold] * 2, color="red")
        ax[0].legend()

        # Draw Frequency Domain Plot
        freqplot = np.zeros((self.config.blocksize, 2))
        freqlines = ax[1].plot(freqplot)
        freqlines[0].set_label("raw_psd")
        freqlines[1].set_label("filtered_psd")
        ax[1].axis((0, self.config.samplerate / 2, 0, 0.1))
        ax[1].legend()

        # Update plot annimations
        def update_plot(frame, timeplot, freqplot):
            for data in audio:
                self.is_silent(data)
                freq, raw_psd, raw_fhat = fft(data, self.config.samplerate)

                cleaned = self.remove_background_noise(data, self.config.samplerate)
                freq, cleaned_psd, cleaned_fhat = fft(cleaned, self.config.samplerate)

                shift = len(data)
                timeplot = np.roll(timeplot, -shift, axis=0)
                timeplot[-shift:, :channels] = data
                timeplot[-shift:, channels:] = cleaned
                freqplot = np.roll(freqplot, -shift, axis=0)
                freqplot[-shift:, :1] = raw_psd.reshape(-1, 1)
                freqplot[-shift:, 1:] = cleaned_psd.reshape(-1, 1)
                for column, line in enumerate(timelines):
                    line.set_ydata(timeplot[:, column])
                for column, line in enumerate(freqlines):
                    line.set_ydata(freqplot[:, column])
                    line.set_xdata(freq.reshape(-1, 1))
            return [*timelines, *freqlines]

        ani = FuncAnimation(fig, lambda frame: update_plot(frame, timeplot, freqplot), interval=interval, blit=True)

        log.info("AudioRecorder starting...")
        stream.start()
        plt.show()

    def play(self, data):
        try:
            sd.play(data, self.config.samplerate)
            status = sd.wait()
        except KeyboardInterrupt:
            self.close()

    def playall(self):
        log.info("AudioRecorder Playback")
        for data in self.audio:
            print(data["speech"].shape)
            self.play(data["speech"])

    def save_clip(self, data):
        sf.write("./voice_sample.wav", data, self.config.samplerate)