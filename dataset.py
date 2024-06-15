from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from config import *
import pymupdf
import warnings
import pickle
import torch
import os
import io
import re

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class Questions():
    def __init__(self, lecture: str) -> None:
        self.questions = []
        self.lecture = lecture

    def add_question(self, text: str):
        self.questions.append(text)

    def add_free_text(self, text: str):
        self.free_text = re.findall(r'\d+\.\s(.+)', text)
        self.questions.extend(self.free_text)

    def __str__(self) -> str:
        return "\n\n".join([f"[Question Start]{q}[Question End]" for q in self.questions])









class Lecture():

    def __init__(self, topic: str, content:str, num_slides: int, lecture_num: int, dependencies: list["Lecture"] = None) -> None:
        self.content = content
        self.topic = topic
        self.dependencies = dependencies
        self.num_slides = num_slides
        self.lecture_num = lecture_num
    
class Dataset():

    def __init__(self):
        self._img_model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        self._img_model = self._img_model.to("cuda")
        self._img_model.eval()
        self.lectures = {} #type: dict[int, Lecture]

    def create_dataset(self):
        
        for file in os.listdir("data"):
            metadata = file.split("-")
            lecture_num = int(metadata[0])
            topic = metadata[-1].split(".pdf")[0]
            
            pdf = pymupdf.open("data/" + file)
            page: pymupdf.Page = None  # for type hinting
            slides = []
            for page in tqdm(pdf.pages(start=1, stop=pdf.page_count-1), desc=f"Processing {file}", total=pdf.page_count, unit="slides"):
                clip = page.rect 
                clip.y1 = 505 # remove footer
                text = page.get_textpage(clip=clip).extractText()
                table: pymupdf.table.Table = None

                tables = page.find_tables()
                tables = tables.tables[1:]
                for table in tables:
                    text += table.to_pandas().to_html()

                # generate a description for the image if any exists
                description = None
                if page.get_images():
                    img = page.get_pixmap(dpi=300, clip=page.bound()).tobytes()
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                    description = self.describe_image(topic, img)

                if description:
                    text += "\n" + description + "\n"
                    slides.append(text)
                else:
                    slides.append(text)
                

            text = "\n\n".join(slides)
            # text = self.cleanup_text(text)

            lecture = Lecture(topic=topic, content=text,
                            lecture_num=lecture_num, num_slides=len(slides))

            self.lectures[lecture_num] = lecture

        return self.lectures

    def cleanup_text(self, text: str):
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """
            You review are good at reviewing lecture slides and merging their content into a single long text wihout changing any of the information or wording"""},
            {"role": "user", "content": f"Merge the following slides: {text}."}],
        temperature=0.2)

        return completion.choices[0].message.content

    def describe_image(self, topic: str, img: Image):
        """
        Given a context and an image, generate a description for the image

        Args:
            topic (str): The topic of the lectur
            img (Image): The image to describe
        """
        prompt = "describe what you see knowing we are in a NLP lecture"
        messages = [
            {"role":"user","content":f"The lecture is about the topic: {topic}"},
            {"role":"user","content":prompt}
        ]
        res = self._img_model.chat(
            image=img,
            msgs=messages,
            tokenizer=self._tokenizer,
            temperature=0.7,
            max_new_tokens=500,
            system_prompt='You are a helpful assistant that describes an image and relate it to the given topic',
        )

        return f"[IDS] {res} [IDE]"

    def add_dependencies(self):
        lectures = pickle.load(open("dataset.pkl", "rb"))
        dependencies = {
            "1_BytePairEncoding.py": [1], # excersise 1 depends on lecture 1
            "2_N-Grams.ip.py": [2], # excersise 2 depends on lecture 2
            "3_SimpleEmbeddings.py": [3,4], # excersise 3 depends on lecture 3 and 4
            "4_VectorSimilarity.py": [5,6], # excersise 4 depends on lecture 5 and 6
            "5_Neural_Language_Model.py": [7], # excersise 5 depends on lecture 7
            "6_Keywords.py":[]# 6 is a standalone excersise
        }
        for lecture in lectures.values():
            lecture.dependencies = None

        for k,v in dependencies.items():
            # add standalone excersises to the first lecture
            assignment = open(f"data/assignments/{k}", "r").read()
            assignment = "[Code Start]\n\n" + assignment + "\n\n[Code End]"
            if len(v) == 0:
                if lectures[0].dependencies is None:
                    lectures[0].dependencies = [assignment]
                else:
                    lectures[0].dependencies.append(assignment)

            for i in v:
                if lectures[i].dependencies is None:
                    lectures[i].dependencies = [assignment]
                else:
                    lectures[i].dependencies.append(assignment)

        with open("dataset.pkl", "wb") as f:
            pickle.dump(lectures, f)


if __name__ == "__main__":
    # dataset = Dataset()
    # dataset.create_dataset()
    # with open("dataset.pkl", "wb") as f:
    #     pickle.dump(dataset.lectures, f)
    # dataset.add_dependencies()
    lectures = pickle.load(open("dataset.pkl", "rb"))
    with open("data.txt", "w", encoding="utf-8") as f:
        for lecture in lectures.values():
            f.write("[Lecture Start]\n\n")
            f.write(f"------------{lecture.topic}------------\n")
            f.write(f"{lecture.content}")
            if lecture.dependencies:
                for dep in lecture.dependencies:
                    f.write("\n\n")
                    f.write(f"{dep}")
            f.write("[Lecture End]\n\n")


