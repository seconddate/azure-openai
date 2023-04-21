"""
@ created_by: ayaan
@ created_at: 2023. 04. 19
"""
import os
import json
import shutil
import time
from datetime import datetime
import openai
from openai import cli
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

        openai.api_key = self.api_key
        openai.api_base = self.api_endpoint
        openai.api_type = self.api_type
        openai.api_version = self.api_version

    def execute_chatgpt(self):
        """ChatGPT 실행"""

        sample_message = [{"role": "user", "content": "메가존 클라우드에 간략히 설명해줘."}]

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

        def check_status(training_id, validation_id):
            train_status = openai.File.retrieve(training_id)["status"]
            valid_status = openai.File.retrieve(validation_id)["status"]
            print(f"Status (training_file | validation_file): {train_status} | {valid_status}")
            return (train_status, valid_status)

        # Upload the training and validation dataset files to Azure OpenAI.
        training_id = cli.FineTune._get_or_upload(training_file_name, True)
        validation_id = cli.FineTune._get_or_upload(validation_file_name, True)

        # Check on the upload status of the training and validation dataset files.
        (train_status, valid_status) = check_status(training_id, validation_id)

        # Poll and display the upload status once a second until both files have either
        # succeeded or failed to upload.
        while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
            time.sleep(1)
            (train_status, valid_status) = check_status(training_id, validation_id)

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

        create_args = {
            "training_file": training_id,
            "validation_file": validation_id,
            "model": self.api_model,
            # "hyperparams": {"n_epochs": 1},
            "n_epochs": 1,
            "compute_classification_metrics": True,
            "classification_n_classes": 3,
        }
        # Create the fine-tune job and retrieve the job ID
        # and status from the response.
        resp = openai.FineTune.create(**create_args)
        job_id = resp["id"]
        status = resp["status"]

        # You can use the job ID to monitor the status of the fine-tune job.
        # The fine-tune job may take some time to start and complete.
        print(f"Fine-tuning model with job ID: {job_id}. \n Status : {status}")
        return job_id

    def status_check_custom_model(self, job_id):
        """맞춤형 모델 상태 확인

        Args:
            job_id (str): job_id
        """
        # Get the status of our fine-tune job.
        status = openai.FineTune.retrieve(id=job_id)["status"]

        # If the job isn't yet done, poll it every 2 seconds.
        if status not in ["succeeded", "failed"]:
            print(f"Job not in terminal status: {status}. Waiting.")
            while status not in ["succeeded", "failed"]:
                time.sleep(2)
                status = openai.FineTune.retrieve(id=job_id)["status"]
                print(f"Status: {status}")
        else:
            print(f"Fine-tune job {job_id} finished with status: {status}")

        # Check if there are other fine-tune jobs in the subscription.
        # Your fine-tune job may be queued, so this is helpful information to have
        # if your fine-tune job hasn't yet started.
        print("Checking other fine-tune jobs in the subscription.")
        result = openai.FineTune.list()
        print(f"Found {len(result)} fine-tune jobs.")

    def deploy_custom_model(self, job_id):
        """Azure OpenAI로 모델 배포

        Args:
            job_id (str): job_id

        Returns:
            deployment_id(str): deployment_id
        """

        # Retrieve the name of the customized model from the fine-tune job.
        result = openai.FineTune.retrieve(id=job_id)
        if result["status"] == "succeeded":
            model = result["fine_tuned_model"]

        # Create the deployment for the customized model, using the standard scale type without specifying a scale
        # capacity.
        print(f"Creating a new deployment with model: {model}")
        result = openai.Deployment.create(model=model, scale_settings={"scale_type": "standard", "capacity": None})
        # Retrieve the deployment job ID from the results.
        deployment_id = result["id"]

        return deployment_id

    def download_analyze_custom_model(self, job_id):
        """맞춤형 모델 결과 분석

        Args:
            job_id (str): job_id
        """

        # Retrieve the file ID of the first result file from the fine-tune job for
        # the customized model.
        result = openai.FineTune.retrieve(id=job_id)
        if result["status"] == "succeeded":
            result_file_id = result.result_files[0].id
            result_file_name = result.result_files[0].filename

        # Download the result file.
        print(f"Downloading result file: {result_file_id}")
        # Write the byte array returned by the File.download() method to
        # a local file in the working directory.
        with open(f"../files/{self.today}_{result_file_name}", "wb") as file:
            result = openai.File.download(id=result_file_id)
            file.write(result)

        return result_file_id

    def delete_deploy_custom_model(self, deployment_id):
        """배포 모델 삭제

        Args:
            deployment_id (str): deployment_id
        """
        # Delete the deployment for the customized model
        print(f"Deleting deployment ID: {deployment_id}")
        result = openai.Deployment.delete(sid=deployment_id)
        print(result)

    def delete_cusom_model(self, job_id):
        """사용자 정의 모델 삭제

        Args:
            job_id (str): job_id
        """
        # Delete the customized model
        print(f"Deleting customized model ID: {job_id}")
        result = openai.FineTune.delete(sid=job_id)
        print(result)

    def delete_training_file(self, file_ids):
        """학습 파일 삭제

        Args:
            file_ids (list): file_ids
        """
        for file_id in file_ids:
            openai.File.delete(sid=file_id)

    def completion_custom_model(self, deployment_id):
        """질의 하기

        Args:
            joib_id (str): _description_
        """
        print("Sending a test completion job")
        start_phrase = "When I go to the store, I want a"
        response = openai.Completion.create(engine=deployment_id, prompt=start_phrase, max_tokens=4)
        text = response["choices"][0]["text"].replace("\n", "").replace(" .", ".").strip()
        print(f'"{start_phrase} {text}"')


def chatgpt_test():
    """ChatGPT 3.5 Test"""
    openai_mtc = OpenaiMTC("ayaan-gpt35", "2023-03-15-preview")
    openai_mtc.execute_chatgpt()


def create_fine_tuning():
    """Create Fine Tuning"""
    openai_mtc = OpenaiMTC("curie")
    # Step.1 - 학습 데이터 업로드
    training_id, validation_id = openai_mtc.upload_openai_file()
    # Step2 - 맞춤형 모델 만들기(학습 시작)
    job_id = openai_mtc.create_openai_custom_model(training_id, validation_id)
    # Step3 - 맞춤형 모델의 상태 확인
    openai_mtc.status_check_custom_model(job_id)
    # Step4 - 맞춤형 모델 배포
    deployment_id = openai_mtc.deploy_custom_model(job_id)
    # Step5 - 맞춤형 모델 결과 다운로드
    result_file_id = openai_mtc.download_analyze_custom_model(job_id)
    # Step6 - 질의하기
    openai_mtc.completion_custom_model(deployment_id)

    # 배포 모델 삭제
    # openai_mtc.delete_deploy_custom_model(deployment_id)
    # 사용자 정의 모델 삭제
    # openai_mtc.delete_custom_model(job_id)
    # 학습 파일 삭제
    # openai_mtc.delete_training_file([training_id, validation_id, result_file_id])


if __name__ == "__main__":
    # 학습데이터 업로드&사용자 모델 생성
    create_fine_tuning()
