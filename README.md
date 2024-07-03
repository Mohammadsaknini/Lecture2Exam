Creating exam questions is tedious for professors and often time-consuming, distracting them from their primary responsibilities of teaching and research. Developing a large language model (LLM) pipeline to generate exam questions from lecture notes aims to streamline this process and enhance efficiency and accuracy. This automated solution ensures comprehensive coverage of course material and produces high-quality, consistent questions. It allows customization to suit different question formats and difficulty levels. Ultimately, this pipeline should enable educators to focus more on student engagement and the quality of instruction rather than stressing about exam questions.

To evaluate the performance of our pipeline, we employed two different evaluation processes that used three different LLMs, namely GPT3.5-Turbo, GPT4o and [ThePitbull-21B-v2](https://huggingface.co/fblgit/UNA-ThePitbull-21.4B-v2)

First, we asked the models to evaluate themselves based mainly on the difficulty and relevance of the generated questions according to the lecture context. Notably, the models did not know they generated the questions themselves.

![Evaluation Plot](https://github.com/MohammadSakhnini/nlp_project/blob/main/poster/figures/eval_plot.png?raw=true)

The takeaway from these results is that all models considered the generated questions relevant to the given context, but they varied in their difficulty. The local model showed much higher variance than GPT-3.5 Turbo and GPT-4o. On the other hand, GPT-4o tended to generate only difficult questions. Additionally, we observed that GPT-3.5 Turbo and the local model had some issues, as indicated by the (0,0) position, meaning their answers could not be validated, whereas GPT-4o had no issues providing the correct format.

Additionally, we conducted a live survey in which we asked 26 participants to identify the human-generated questions from four different groups of questions. Each group contained three questions from an anonymous model (those mentioned above) and an additional group containing human-written questions. Notably, the participants were all knowledgeable about LLMs, and the majority had taken part in the course from which the questions were generated from.

The questions can be found under the survey section in ![Alt text]()

The results were

| Group1  | Group 2 | Group 3 | Group 4 |
|---------|---------|---------|---------|
| 3       | 8       | 4       | 11      |


