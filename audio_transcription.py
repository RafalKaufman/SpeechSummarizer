import os

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence


class DialogueTranscription:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.text_recognizer = sr.Recognizer()
        self.dialogue_transcription = ""

    def transcribe(self, audio_file_name: str) -> str:
        with sr.AudioFile(audio_file_name) as audio_source:
            audio_listened = self.text_recognizer.record(audio_source)
            transcription = self.text_recognizer.recognize_google(audio_listened)
        return transcription

    def splitting_audio_on_silence(self) -> list:
        entire_recording = AudioSegment.from_file(self.audio_path)
        # min_silence_len and keep_silence values are represented in miliseconds
        audio_chunks = split_on_silence(
            entire_recording,
            min_silence_len=500,
            silence_thresh=entire_recording.dBFS - 14,
            keep_silence=500,
        )
        return audio_chunks

    def audio_chunk_processing(self, audio_chunks: list) -> str:
        chunks_folder_name = f"{self.audio_path}-chunks"
        if not os.path.isdir(chunks_folder_name):
            os.mkdir(chunks_folder_name)
        whole_transcription = ""
        for i, audio_chunk in enumerate(audio_chunks, start=1):
            chunk_filename = os.path.join(chunks_folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            try:
                chunk_transcription = self.transcribe(chunk_filename)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                chunk_transcription = f"{chunk_transcription.capitalize()}. "
                whole_transcription += chunk_transcription
        return whole_transcription

    def get_dialogue_transcription(self) -> str:
        if self.dialogue_transcription == "":
            self.dialogue_transcription = self.audio_chunk_processing(
                self.splitting_audio_on_silence()
            )
        return self.dialogue_transcription

    def __len__(self) -> int:
        return len(self.dialogue_transcription)


dial1 = DialogueTranscription("dailylife022.wav")
print(dial1.get_dialogue_transcription())
