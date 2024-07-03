Creating exam questions is tedious for professors and often time-consuming, distracting them from their primary responsibilities of teaching and research. Developing a large language model (LLM) pipeline to generate exam questions from lecture notes aims to streamline this process and enhance efficiency and accuracy. This automated solution ensures comprehensive coverage of course material and produces high-quality, consistent questions. It allows customization to suit different question formats and difficulty levels. Ultimately, this pipeline should enable educators to focus more on student engagement and the quality of instruction rather than stressing about exam questions.

To evaluate the performance of our pipeline, we employed two different evaluation processes that used three different LLMs, namely GPT3.5-Turbo, GPT4o and [ThePitbull-21B-v2](https://huggingface.co/fblgit/UNA-ThePitbull-21.4B-v2)

First, we asked the models to evaluate themselves based mainly on the difficulty and relevance of the generated questions according to the lecture context. Notably, the models did not know they generated the questions themselves.

![Evaluation Plot](https://github.com/MohammadSakhnini/nlp_project/blob/main/poster/figures/eval_plot.png?raw=true)

The takeaway from these results is that all models considered the generated questions relevant to the given context, but they varied in their difficulty. The local model showed much higher variance than GPT-3.5 Turbo and GPT-4o. On the other hand, GPT-4o tended to generate only difficult questions. Additionally, we observed that GPT-3.5 Turbo and the local model had some issues, as indicated by the (0,0) position, meaning their answers could not be validated, whereas GPT-4o had no issues providing the correct format.

Additionally, we conducted a live survey in which we asked 29 participants to identify the human-generated questions from four different groups of questions. Each group contained three questions from an anonymous model (those mentioned above) and an additional group containing human-written questions. Notably, the participants were all knowledgeable about LLMs, and the majority had taken part in the course from which the questions were generated from.

The groups in the survey were as follows:
![Alt text](https://github.com/MohammadSakhnini/Lecture2Exam/blob/main/poster/figures/questions.png?raw=true)

### Before checking the results, try to identify the human-generated questions yourself!

<details>
  <summary>Results</summary>

<br>

| Group 1 (Local) 	| Group 2 (GPT4o) 	| Group 3 (GPT3.5T) 	| Group 4 (Human) 	|
|:---------------:	|:---------------:	|:-----------------:	|:---------------:	|
|        4        	|        9        	|         4         	|        12       	|

Based on the results we see that the LLMs have recived 58% of the votes, while the human questions have recived 42% of the votes. This shows that the LLMs are able to generate questions that are similar to human generated questions.

 The human group received the most votes, followed by the GPT-4o group. The local model and GPT-3.5 Turbo received the least votes. After the participants completed the survey, we asked them to reason why they chose the group they did. The most common reason for choosing group 4 was that the questions were more "human-like" and that it contained a typo. 

</details>
<br>

# Usage
To use the pipeline, you need to follow these steps:

1. Make sure the lectures are named as follows {Number of lecture}-{Lecture Title}.pdf
2. Place the lectures in the data/lectures folder
3. In order to be able to generate questions that are based on matrials other than the lecture slides such as assignments, you need to place the files in the data/assignments folder.