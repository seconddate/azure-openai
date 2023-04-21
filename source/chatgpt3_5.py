"""
@ created_by: ayaan
@ created_at: 2023. 04. 19
"""
import os
import openai
from dotenv import load_dotenv


def main():
    load_dotenv()

    sample_message = [{
        "role": "user",
        "content": "메가존 클라우드에 간략히 설명해줘."
    }]

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_ENDPOINT")
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    response = openai.ChatCompletion.create(
        engine=os.getenv('OPENAI_MODEL_NAME'),
        messages=sample_message,
        temperature=0.5,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    print(response['choices'][0]['message']['content'])
    if response and response['choices'] and len(response['choices']) > 0:
        return response['choices'][0]


if __name__ == '__main__':
    main()
