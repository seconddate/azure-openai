"""
@ created_by: ayaan
@ created_at: 2023. 04. 19
"""
import os
import openai
import time
from dotenv import load_dotenv


def main():
    """Main"""
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_ENDPOINT")
    openai.api_type = "azure"
    openai.api_version = "2022-12-01"

    # This will correspond to the custom name you chose for your deployment when you deployed a model.
    deployment_name = os.getenv("OPENAI_MODEL_NAME")

    # Send a completion call to generate an answer
    print('Sending a test completion job')
    start_phrase = '메가존클라우드에 대해 설명해줘.'
    response = openai.Completion.create(
        engine=deployment_name, prompt=start_phrase, max_tokens=10)
    text = response['choices'][0]['text'].replace(
        '\n', '').replace(' .', '.').strip()


if __name__ == '__main__':
    main()
