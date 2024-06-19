EMBEDDING_MODEL = "vonjack/bge-m3-gguf" #  https://huggingface.co/vonjack/bge-m3-gguf
MODEL = "bartowski/UNA-ThePitbull-21.4B-v2-GGUF" # https://huggingface.co/bartowski/UNA-ThePitbull-21.4B-v2-GGUF
BASE_URL = "http://localhost:8085/v1"
API_KEY = "lm-studio"

QUESTION_PATTERN = r'\[Question Start\](.*?)\[Question End\]'

EVAL_BASE_URL = "http://localhost:1234/v1"
EVAL_MODEL = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
EVAL_API_KEY = "lm-studio"
EVAL_TEMPERATURE = 0.8
EVAL_RETRIES = 3
QUESTION_EVAL_LENGTH = 4
OVERALL_EVAL_LENGTH = 4