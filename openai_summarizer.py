from openai import OpenAI
from audio_transcription import DialogueTranscription

def generate_summary(input_text, max_tokens=200):

    openai_prompt = f"Summarize the following conversation: '{input_text}'"

    openai_response = client.completions.create(
        model="text-davinci-002",
        prompt=openai_prompt,
        max_tokens=max_tokens
    )
    summary = openai_response.choices[0].text
    return summary

client = OpenAI()

transcr = DialogueTranscription("dailylife022.wav")
transcr.get_dialogue_transcription()
transcr_text = transcr.transcription

transc_summary = generate_summary(transcr_text)
print("Summary:", transc_summary)
