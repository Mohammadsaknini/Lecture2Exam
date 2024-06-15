from dataset import Lecture, Questions
from openai import OpenAI
from prompts import *
from config import *
import pickle

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
generate = lambda prompt: client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes exam questions for a NLP course. You are given lecture slides and asked to generate questions based on the content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )


def generate_questions(num_questions=5, mc_questions=1, code_questions=1):
    lectures = pickle.load(open("dataset.pkl", "rb")) #type: list[Lecture]
    lectures = list(lectures.values())
    module_questions = []  #type: list[Questions]

    for i, lecture in enumerate(lectures):
        lecture_questions = Questions(lecture)
        if lecture.dependencies is not None:
            if len(lecture.dependencies) == 0 and code_questions > 1:
                print(f"No coding question for lecture {i+1} - {lecture.topic}")


        ft_questions = num_questions - mc_questions - code_questions
        if lecture.dependencies is None:
            ft_questions += code_questions

        # ft questions
        questions = generate(FREE_TEXT_QUETIONS.format(topic=lecture.topic, content=lecture.content, num_questions=ft_questions))        
        lecture_questions.add_free_text(questions.choices[0].message.content)

        # mc questions
        for _ in range(mc_questions):
            questions = generate(MC_QUESTION.format(topic=lecture.topic, content=lecture.content))
            lecture_questions.add_question(questions.choices[0].message.content)

        # code questions
        if lecture.dependencies is not None:
            for _ in range(code_questions):
                code_content = [i for i in lecture.dependencies]
                questions = generate(CODE_QUESTION.format(topic=lecture.topic, content=code_content))
                lecture_questions.add_question(questions.choices[0].message.content)

        module_questions.append(lecture_questions)

        with open(f"questions/{lecture.topic}.txt", "w", encoding="utf-8") as f:
            f.write(str(lecture_questions))


    pickle.dump(module_questions, open("questions.pkl", "wb"))
    return module_questions

generate_questions(5, 1, 1)

