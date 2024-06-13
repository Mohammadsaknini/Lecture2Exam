from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from config import *
import pymupdf
import warnings
import re
import os
import io
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class Lecture():

    def __init__(self, topic: str, content:str, num_slides: int, lecture_num: int, dependencies: list["Lecture"] = None) -> None:
        self.content = content
        self.topic = topic
        self.dependencies = dependencies
        self.num_slides = num_slides
        self.lecture_num = lecture_num
    
class Dataset():

    def __init__(self):
        self._img_model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, device_map="cuda")
        self._tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, device_map="cuda")
        self._img_model.eval()
        self.lectures = {} #type: dict[int, Lecture]
        self.create_dataset()

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

                # generate a description for the image if any exists
                description = None
                if page.get_images():
                    img = page.get_pixmap(dpi=300, clip=page.bound()).tobytes()
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                    context = "\n\n".join(slides)
                    pattern = r'\[IDS\].*?\[IDE\]'
                    context = re.sub(pattern, '', context) # becasue the model will just copy and paste the prompt
                    description = self.describe_image(topic, context, img)

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

    def describe_image(self, topic: str, context: str, img: Image):
        """
        Given a context and an image, generate a description for the image

        Args:
            context (str): The context of the lecture from the first slide to the current slide
            img (Image): The image to describe
        """
        prompt = "Explain the image within the slide with the help of the context of the lecture."
        messages = [
            {"role":"user","content":f"The lecture is about the topic: {topic}. \n context: \n{context}"},
            {"role":"user","content":prompt}
        ]
        res = self._img_model.chat(
            image=img,
            msgs=messages,
            # sampeling=True,
            tokenizer=self._tokenizer,
            temperature=0.2,
            system_prompt='You are an expert of linking lecture context to a given image and describing it in a maximum of 50 words.',
        )

        return f"[IDS] {res} [IDE]"
    

import pickle

if __name__ == "__main__":
    # dataset = Dataset()
    # with open("dataset.pkl", "wb") as f:
    #     pickle.dump(dataset.lectures, f)
    lecutres = pickle.load(open("dataset.pkl", "rb"))

