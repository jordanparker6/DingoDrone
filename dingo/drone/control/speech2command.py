import torch
import logging
import threading
import pyphonetics
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from utils import IterableQueue, timeit
from drone.io.audio import AudioRecorder
from drone.control.commands import COMMANDS

log = logging.getLogger(__name__)

TRIGGER = "hey dingo"
MODEL = "./dingo/models/s2t-small-librispeech-asr"
#MODEL = "./dingo/models/wav2vec2-large-xlsr-53-english"
#MODEL = "./dingo/models/wav2vec2_tiny_random_robust"

def download_model():
    """Downloads the model config and binary from Hugginface"""
    log.info("Downloading remote model...")
    model = Wav2Vec2ForCTC.from_pretrained(f"patrickvonplaten/wav2vec2_tiny_random_robust")
    processor = Wav2Vec2Processor.from_pretrained(f"patrickvonplaten/wav2vec2_tiny_random_robust")
    model.save_pretrained(MODEL)
    processor.save_pretrained(MODEL)
    return

class Speech2Command(threading.Thread):
    """Command Controll operated by speech commands"""
    def __init__(self, model_path:str = MODEL):
        super().__init__(name="Speech2Command")
        log.info("Loading command control")
        #self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        #self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(model_path)
        self.processor = Speech2TextProcessor.from_pretrained(model_path)
        self.recorder = AudioRecorder()
        self.commands = IterableQueue()
        self.soundex = pyphonetics.FuzzySoundex()
    
    def run(self):
        self.recorder.start()
        while True:
            for audio in self.recorder.audio:
                if audio:
                    self.recorder.save_clip(audio["speech"])
                    audio = audio["speech"].flatten()
                    text = timeit(lambda: self.transcribe(audio)[0], "transcribe")
                    command = timeit(lambda: self.text2command(text), "text2command")
                    if command:
                        self.commands.put(command)

    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
        if isinstance(self.model, Speech2TextForConditionalGeneration):
            generated_ids = self.model.generate(input_ids=inputs["input_features"])
            transcription = self.processor.batch_decode(generated_ids)
        else:
            logits = self.model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
        return transcription

    def text2command(self, text):
        text = text.replace("</s>", "").strip().lower()
        if len(text) < 10:
            return None
        text = text.split(" ")
        trigger = TRIGGER.split(" ")
        trig_phn_dist = 0
        for i, word in enumerate(trigger):
            trig_phn_dist += self.soundex.distance(word, text[i])
        log.info(f"Processseing text: {text} | Trigger similairty {trig_phn_dist}")
        if trig_phn_dist <= len(trigger):
            text = " ".join(text[len(trigger):])
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