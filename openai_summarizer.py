from openai import OpenAI

from audio_transcription import DialogueTranscription


def generate_openai_summary(
    dialogue_to_summarize: str, max_token_number: int = 100
) -> str:
    openai_prompt = f"Summarize the following conversation: '{dialogue_to_summarize}'"

    openai_response = openai_client.completions.create(
        model="text-davinci-002", prompt=openai_prompt, max_tokens=max_token_number
    )
    openai_summary = openai_response.choices[0].text
    return openai_summary


openai_client = OpenAI()

dialogue = DialogueTranscription("dailylife022.wav")
dialogue_transcription = dialogue.get_dialogue_transcription()

dialogue_openai_summary = generate_openai_summary(dialogue_transcription)
print("OpenAI summary:", dialogue_openai_summary)
