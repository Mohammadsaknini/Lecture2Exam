from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain import hub
from tqdm import tqdm
from config import *
import logging

logging.getLogger().setLevel(logging.ERROR)
class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def embed(self, text):
        try:
          return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        except:
          return None
        
    def embed_documents(self, texts):
        embeddings = []
        for text in tqdm(texts):
            text = text.replace("\n", " ")
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

def test():
    # read and split text with overlapping chunks
    text = open("wiki_nlp.txt", encoding="utf-8").read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Create the first 5 documents based on the chunks created
    docs = text_splitter.split_text(text)[:5]

    # Create the embeddings and the retriever
    embeddings = BGEEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.from_texts(docs, embeddings)
    retriever = db.as_retriever()

    # incase we want to save the db
    # db.save_local("faiss_db")
    # retriever = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    # retriever = retriever.as_retriever()

    # Query the retriever
    query = "Rule-based vs. statistical NLP"
    docs = retriever.invoke(query)
    print(docs)

    tool = create_retriever_tool(
        retriever=retriever,
        name="history_of_nlp",
        description="Searches are turns information about the history of NLP.",
    )

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, verbose=True)
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools=[tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

    print(agent_executor.invoke({"input": "What is the history of NLP?"}))