from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from prompts import *
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

    def __init__(self, topic: str, content: str, num_slides: int, lecture_num: int,
                 dependencies: list["Lecture"] = None) -> None:
        self.content = content
        self.topic = topic
        self.dependencies = dependencies
        self.num_slides = num_slides
        self.lecture_num = lecture_num


class Dataset():

    def __init__(self):
        self._img_model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True,
                                                    torch_dtype=torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        self._img_model = self._img_model.to("cuda")
        self._img_model.eval()
        self.lectures = {}  # type: dict[int, Lecture]

    def create_dataset(self):
        print("Loading files in data/lectures...")
        dir = os.listdir("data/lectures")
        page_counter = 0
        for file in dir:
            if (file[0].isdigit() is False) or ("-" not in file):
                raise ValueError("Error: expected file name to be: number-topic")
            metadata = file.split("-")
            lecture_num = int(metadata[0])
            topic = metadata[-1].split(".pdf")[0]

            pdf = pymupdf.open("data/" + file)
            page: pymupdf.Page = None  # for type hinting
            page_counter += pdf.page_count

            slides = []
            for page in tqdm(pdf.pages(start=1, stop=pdf.page_count - 1), desc=f"Processing {file}",
                             total=pdf.page_count, unit="slides"):
                clip = page.rect
                clip.y1 = 505  # remove footer
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

        with open("data/store/dataset.pkl", "wb") as f:
            pickle.dump(self.lectures, f)

        self.add_dependencies()
        print("Successfully loaded {} pdfs with a total of {} pages".format(len(dir), page_counter))
        return self.lectures

    def cleanup_text(self, text: str):
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": """You review text related to a NLP lecture and clean it up for better readability.
             You never change the meaning of the text, only the structure and style.
             You never reword any technical terms or concepts. You only reformat the text to make it more readable.
             Incase you encounter non sense letters you can remove it."""},
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
            {"role": "user", "content": f"The lecture is about the topic: {topic}"},
            {"role": "user", "content": prompt}
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

    def add_dependencies(self) -> dict[int, Lecture]:
        """
        Add dependencies to the lectures based on the excersises in the assignments folder

        Returns:
            dict[int, Lecture]: The lectures with dependencies added
        """
        lectures = pickle.load(open("data/store/dataset.pkl", "rb"))
        dependencies = {
            "1_BytePairEncoding.py": [1],  # excersise 1 depends on lecture 1
            "2_N-Grams.ip.py": [2],  # excersise 2 depends on lecture 2
            "3_SimpleEmbeddings.py": [3, 4],  # excersise 3 depends on lecture 3 and 4
            "4_VectorSimilarity.py": [5, 6],  # excersise 4 depends on lecture 5 and 6
            "5_Neural_Language_Model.py": [7],  # excersise 5 depends on lecture 7
            "6_Keywords.py": []  # 6 is a standalone excersise
        }
        for lecture in lectures.values():
            lecture.dependencies = None

        for k, v in dependencies.items():
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

        with open("data/store/dataset.pkl", "wb") as f:
            pickle.dump(lectures, f)

        return lectures


class QuestionsGenerator():

    def __init__(self, base_url=None, api_key=None, model=None):
        """
        Args:
            base_url (str): The base url for the openai api
            api_key (str): The api key for the openai api
            model (str): The model to use for generating questions
        """
        if base_url is None:
            base_url = BASE_URL
        if api_key is None:
            api_key = API_KEY
        if model is None:
            model = MODEL
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that writes exam questions for a NLP course. You are given "
                            "lecture slides and asked to generate questions based on the content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

    def generate_questions(self, num_questions=5, mc_questions=1, code_questions=1, verbose=False) -> list[Questions]:
        """
        Generate questions for each lecture in the dataset

        Args:
            num_questions (int): Total number of questions to generate for each lecture
            mc_questions (int): Number of multiple choice questions to generate
            code_questions (int): Number of coding questions to generate
            verbose (bool): Print questions to console
        """
        lectures = pickle.load(open("data/store/dataset.pkl", "rb"))  # type: list[Lecture]
        lectures = list(lectures.values())
        module_questions = []  # type: list[Questions]

        for i, lecture in enumerate(lectures):
            lecture_questions = Questions(lecture)
            if lecture.dependencies is not None:
                if len(lecture.dependencies) == 0 and code_questions > 1:
                    print(f"No coding question for lecture {i + 1} - {lecture.topic}")

            ft_questions = num_questions - mc_questions - code_questions
            if lecture.dependencies is None:
                ft_questions += code_questions

            # ft questions
            questions = self.generate(
                FREE_TEXT_QUESTIONS.format(topic=lecture.topic, content=lecture.content, num_questions=ft_questions))
            lecture_questions.add_free_text(questions.choices[0].message.content)

            # mc questions
            for _ in range(mc_questions):
                questions = self.generate(MC_QUESTIONS.format(topic=lecture.topic, content=lecture.content))
                lecture_questions.add_question(questions.choices[0].message.content)

            # code questions
            if lecture.dependencies is not None:
                for _ in range(code_questions):
                    code_content = [i for i in lecture.dependencies]
                    questions = self.generate(CODE_QUESTIONS.format(topic=lecture.topic, content=code_content))
                    lecture_questions.add_question(questions.choices[0].message.content)

            module_questions.append(lecture_questions)

            with open(f"questions/{lecture.topic}.txt", "w", encoding="utf-8") as f:
                f.write(str(lecture_questions))

            if verbose:
                print(f"Questions for lecture {i + 1} - {lecture.topic}")
                print(lecture_questions)
                print("\n\n")

        pickle.dump(module_questions, open("data/store/questions.pkl", "wb"))
        return module_questions


if __name__ == "__main__":
    lectures = pickle.load(open("data/store/dataset.pkl", "rb"))
    with open("data/store/data.txt", "w", encoding="utf-8") as f:
        for lecture in lectures.values():
            f.write("[Lecture Start]\n\n")
            f.write(f"------------{lecture.topic}------------\n")
            f.write(f"{lecture.content}")
            if lecture.dependencies:
                for dep in lecture.dependencies:
                    f.write("\n\n")
                    f.write(f"{dep}")
            f.write("[Lecture End]\n\n")
