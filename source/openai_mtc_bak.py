"""
@ created_by: ayaan
@ created_at: 2023. 04. 19
"""
import os
import openai
import requests
from dotenv import load_dotenv


class OpenaiMTC:
    """MTC OpenAI Class"""

    def __init__(self, model, api_version="2022-12-01"):
        """생성자

        Args:
            api_version (str, optional): openai api version. Defaults to '2022-12-01'.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.api_version = api_version
        self.api_type = "azure"
        self.api_model = model

    def execute_chatgpt(self):
        """ChatGPT 실행"""

        sample_message = [{"role": "user", "content": "메가존 클라우드에 간략히 설명해줘."}]

        openai.api_key = self.api_key
        openai.api_base = self.api_endpoint
        openai.api_type = self.api_type
        openai.api_version = self.api_version

        response = openai.ChatCompletion.create(
            engine=self.api_model,
            messages=sample_message,
            temperature=0.5,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )

        print(response["choices"][0]["message"]["content"])
        if response and response["choices"] and len(response["choices"]) > 0:
            return response["choices"][0]

    def upload_openai_file(self):
        """Upload OpenAI File"""
        api_url = f"{self.api_endpoint}/openai/files"

        file_path = "../files/test.jsonl"
        file = open(file_path, "rb")
        binary = file.read()
        file.close()

        params = {"api-version": self.api_version}
        data = {"purpose": "fine-tune", "file": binary}
        headers = {"api-key": self.api_key, "content-type": "application/json"}

        response = requests.post(url=api_url, headers=headers, timeout=5000, params=params, data=data)
        print(response.__dict__)

    def create_fine_tuning(self, model):
        """Create Fine Tuning Job"""
        api_url = f"{self.api_endpoint}/openai/fine-tunes"

        response = requests.post(
            url=api_url,
            timeout=5000,
            params={"api-version": self.api_version},
            data={"model": model, "training_file": ""},
        )


def chatgpt_test():
    """ChatGPT 3.5 Test"""
    openai_mtc = OpenaiMTC("ayaan-gpt35", "2023-03-15-preview")
    openai_mtc.execute_chatgpt()


def create_fine_tuning():
    """Create Fine Tuning"""
    openai_mtc = OpenaiMTC("curie")
    openai_mtc.upload_openai_file()


if __name__ == "__main__":
    create_fine_tuning()
