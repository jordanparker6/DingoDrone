import logging
import time
import threading
import multiprocessing
from box import Box
import pyphonetics
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

from utils import IterableQueue, timeit
from drone.io.audio import AudioRecorder
from drone.control.commands import COMMANDS

log = logging.getLogger(__name__)

TRIGGER = "dingo"

class Speech2Command(threading.Thread):
    """Command Controll operated by speech commands"""
    def __init__(self):
        super().__init__(name="Speech2Command")
        log.info("Loading command control")
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.recorder = AudioRecorder()
        self.commands = IterableQueue()
        self.soundex = pyphonetics.FuzzySoundex()
    
    def run(self):
        self.recorder.start()
        while True:
            for audio in self.recorder.audio:
                if audio:
                    audio = audio["speech"].flatten()
                    text = timeit(lambda: self.transcribe(audio)[0], "transcribe")
                    command = timeit(lambda: self.text2command(text), "text2command")
                    if command:
                        self.commands.put(command)

    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
        generated_ids = self.model.generate(input_ids=inputs["input_features"])
        transcription = self.processor.batch_decode(generated_ids)
        return transcription

    def text2command(self, text):
        text = text.replace("</s>", "").strip()
        text = text.split(" ")
        trigger_phonetic_distance = self.soundex.distance(TRIGGER, text[0])
        log.info(f"Processseing text: {text} | Trigger similairty {trigger_phonetic_distance}")
        if trigger_phonetic_distance <= 1:
            text = " ".join(text[1:])
            for name, command in COMMANDS.items():
                for trigger in command.keywords:
                    if text.find(trigger) != -1:
                        return command
        log.debug("Text processed but key word 'dingo' was not present.")
        return None
    
    def _test(self):
        from datasets import load_dataset
        import soundfile as sf
        from random import randint
        
        def map_to_array(batch):
            speech, _ = sf.read(batch["file"])
            batch["speech"] = speech
            return batch

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.map(map_to_array)
        audio = ds["speech"][randint(0, len(ds))]
        text = self.transcribe(audio)
        print(text)
        self.recorder.play(audio)