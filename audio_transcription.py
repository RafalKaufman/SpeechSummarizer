import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

class DialogueTranscription():

    def __init__(self, path):
        self.path = path
        self.recognizer = sr.Recognizer()
        self.transcription = ""

    def transcribe(self,filename):
        with sr.AudioFile(filename) as source:
            audio_listened = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_listened)
        return text

    def splitting_on_silence(self):
        sound = AudioSegment.from_file(self.path)
        chunks = split_on_silence(
            sound, min_silence_len = 500,
            silence_thresh = sound.dBFS-14, keep_silence=500)
        return chunks

    def chunk_processing(self,chunks):
        folder_name = f"{self.path}-chunks"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""
        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            try:
                text = self.transcribe(chunk_filename)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
        return whole_text
    def get_dialogue_transcription(self):
        self.transcription = self.chunk_processing(self.splitting_on_silence())
    def __str__(self):
        return self.transcription
    def __len__(self):
        return len(self.transcription)
dial1 = DialogueTranscription("dailylife022.wav")
dial1.get_dialogue_transcription()
print(dial1)
print(len(dial1))
