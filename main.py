from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from dataset import Lecture
from openai import OpenAI
from langchain import hub
from tqdm import tqdm
from config import *
import pickle
import logging
import re

logging.getLogger().setLevel(logging.ERROR)

class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def embed(self, text: str):
        text = text.replace("\n", " ")
        text = re.sub(r'[^\x00-\x7F]+', '', text) # remove all non-ascii characters
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        
    def embed_documents(self, texts):
        embeddings = []
        for text in tqdm(texts):
            text = text.replace("\n", " ")
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            temp = self.embed(text)
            if temp is None:
                continue
            embeddings.append(temp)
        return embeddings

    def embed_query(self, text):
        return self.embed(text)

    async def aembed_documents(self, texts):
        raise NotImplementedError
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text):
        raise NotImplementedError
        return await run_in_executor(None, self.embed_query, text)

def generate_questions():
    lectures = pickle.load(open("dataset.pkl", "rb")) #type: list[Lecture]
    lectures = list(lectures.values())

    PROMPT = """
    Create exam questions based for the topic : {topic}

    Lecture content:
    {content}

    Generate 5 questions based on the content above, make sure they are relevant and challenging.

    MAKE SURE TO INCLUDE at max 1 multiple choice question
    DO NOT INTRODUCE ANY QUESTIONS THAT ARE NOT BASED ON THE CONTENT ABOVE
    THE GENERTED QUESTIONS SHOULD NOT ASKED ABOUT RESEARCH PAPERS OR AUTHORS
    """

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    for i, lecture in enumerate(lectures):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes exam questions for a NLP course. You are given lecture slides and asked to generate questions based on the content."},
                {"role": "user", "content": PROMPT.format(topic=lecture.topic, content=lecture.content)}
            ],
            temperature=0.7,
        )

        with open(f"questions/{lecture.topic}.txt", "w", encoding="utf-8") as f:
            f.write(completion.choices[0].message.content)

        print(f"Questions for lecture {i+1} - {lecture.topic}")
        print("=====================================")
        print(completion.choices[0].message.content)
        print("\n\n")


generate_questions()