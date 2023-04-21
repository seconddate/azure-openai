## Azure OpenAI ChatGPT & Fine-tuning For Python SDK

### 참조

> https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/fine-tuning?pivots=programming-language-python

```sh
# venv 세팅
python -m venv .openaienv
source .openaienv/Source/activate

# .env 파일 생성 후 env값 작성
cp .env.sampel .env

# lib download
pip install -r requirements.txt
# 실행
cd source
python openai_mtc.py
```
