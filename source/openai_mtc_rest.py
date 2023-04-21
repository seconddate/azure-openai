"""
@ created_by: ayaan
@ created_at: 2023. 04. 19
"""
import os
import json
import shutil
import traceback
import time
from datetime import datetime
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
        self.today = datetime.now().strftime("%y%m%d")
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
        """Step.1 - 학습 데이터 업로드"""

        training_file_name = f"../files/training_{self.today}.jsonl"
        validation_file_name = f"../files/validation_{self.today}.jsonl"

        sample_data = [
            {"prompt": "When I go to the store, I want an", "completion": "apple"},
            {"prompt": "When I go to work, I want a", "completion": "coffee"},
            {"prompt": "When I go home, I want a", "completion": "soda"},
        ]

        # Generate the training dataset file.
        print(f"Generating the training file: {training_file_name}")
        with open(training_file_name, "w", encoding="utf-8") as training_file:
            for entry in sample_data:
                json.dump(entry, training_file)
                training_file.write("\n")

        # Copy the validation dataset file from the training dataset file.
        # Typically, your training data and validation data should be mutually exclusive.
        # For the purposes of this example, we're using the same data.
        print("Copying the training file to the validation file")
        shutil.copy(training_file_name, validation_file_name)

        # Upload the training & validation dataset files to Azure OpenAI with the REST API.
        upload_params = {"api-version": self.api_version}
        upload_headers = {"api-key": self.api_key}
        upload_data = {"purpose": "fine-tune"}

        # Upload the training file
        response = requests.post(
            self.api_endpoint + "openai/files",
            params=upload_params,
            headers=upload_headers,
            data=upload_data,
            files={"file": (training_file_name, open(training_file_name, "rb"), "application/json")},
        )
        training_id = (response.json())["id"]
        print(f"Training ID : {training_id}")

        # Upload the validation file
        response = requests.post(
            self.api_endpoint + "openai/files",
            params=upload_params,
            headers=upload_headers,
            data=upload_data,
            files={"file": (validation_file_name, open(validation_file_name, "rb"), "application/json")},
        )
        validation_id = (response.json())["id"]
        print(f"Validation ID : {validation_id}")

        return training_id, validation_id

    def create_openai_custom_model(self, training_id, validation_id):
        """Step2 - 맞춤형 모델 만들기
        # This example defines a fine-tune job that creates a customized model based on curie,
        # with just a single pass through the training data. The job also provides classification-
        # specific metrics, using our validation data, at the end of that epoch.

        Args:
            training_id (str): training_id
            validation_id (str): validation_id

        Returns:
            job_id: job_id
        """

        fine_tune_params = {"api-version": self.api_version}
        fine_tune_headers = {"api-key": self.api_key}
        fine_tune_data = {
            "model": "curie",
            "training_file": training_id,
            "validation_file": validation_id,
            "hyperparams": {"batch_size": 1, "learning_rate_multiplier": 0.1, "n_epochs": 4},
        }

        # Start the fine-tune job using the REST API
        response = requests.post(
            self.api_endpoint + "openai/fine-tunes", params=fine_tune_params, headers=fine_tune_headers, data=json.dumps(fine_tune_data)
        )

        # Retrieve the job ID and job status from the response
        job_id = (response.json())["id"]
        status = (response.json())["status"]

        print(f"Fine-tuning model with job ID: {job_id}. #### Status : {status}")
        return job_id

    def status_check_custom_model(self, job_id):
        """맞춤형 모델 상태 확인

        Args:
            job_id (str): job_id
        """
        # Get the status of our fine-tune job.
        fine_tune_params = {"api-version": self.api_version}
        fine_tune_headers = {"api-key": self.api_key}
        response = requests.get(self.api_endpoint + "openai/fine-tunes/" + job_id, params=fine_tune_params, headers=fine_tune_headers)

        # If the job isn't yet done, poll it every 2 seconds.
        status = (response.json())["status"]
        if status not in ["succeeded", "failed"]:
            print(f"Job not in terminal status: {status}. Waiting.")
            while status not in ["succeeded", "failed"]:
                time.sleep(2)
                response = requests.get(self.api_endpoint + "openai/fine-tunes/" + job_id, params=fine_tune_params, headers=fine_tune_headers)
                status = (response.json())["status"]
                print(f"Status: {status}")
        else:
            print(f"Fine-tune job {job_id} finished with status: {status}")

        # List all fine-tune jobs available in the subscription
        print("Checking other fine-tune jobs in the subscription.")
        response = requests.get(self.api_endpoint + "openai/fine-tunes", params=fine_tune_params, headers=fine_tune_headers)
        print(f'Found {len((response.json())["data"])} fine-tune jobs.')


def chatgpt_test():
    """ChatGPT 3.5 Test"""
    openai_mtc = OpenaiMTC("ayaan-gpt35", "2023-03-15-preview")
    openai_mtc.execute_chatgpt()


def create_fine_tuning():
    """Create Fine Tuning"""
    openai_mtc = OpenaiMTC("curie")
    # Step.1 - 학습 데이터 업로드
    training_id, validation_id = openai_mtc.upload_openai_file()
    # Step2 - 맞춤형 모델 만들기
    job_id = openai_mtc.create_openai_custom_model(training_id, validation_id)
    # Step3 - 맞춤형 모델의 상태 확인
    openai_mtc.status_check_custom_model(job_id)
    # Step4 - 맞춤형 모델 배포


if __name__ == "__main__":
    # 학습데이터 업로드&사용자 모델 생성
    create_fine_tuning()
