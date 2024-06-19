
FREE_TEXT_QUESTIONS = """
Create exam questions based on the topic : {topic}

Lecture content:
{content}

Generate {num_questions} questions based on the content above, make sure they are relevant and challenging.

DO NOT INTRODUCE ANY QUESTIONS THAT ARE NOT BASED ON THE CONTENT ABOVE
THE GENERATED QUESTIONS SHOULD NOT ASK ABOUT RESEARCH PAPERS OR AUTHORS
NEVER PROVIDE AN ANSWER
"""


CODE_QUESTIONS = """
Create one exam question based on the topic : {topic}

Lecture content:
{content}

Generate a single coding question based on the content above, make sure it is relevant and challenging.

YOU MUST ALWAYS PROVIDE A CODE SNIPPET IN THE QUESTION AND ASK A QUESTION BASED ON IT 
THE QUESTIONS SHOULD NOT REQUIRE ANY ACTUAL CODING, JUST UNDERSTANDING OF THE CODE
NEVER PROVIDE AN ANSWER
"""

MC_QUESTIONS = """
Create one exam question based on the topic : {topic}

Lecture content:
{content}

Generate a single multiple choice question based on the content above, make sure it is relevant and challenging.

THE QUESTION SHOULD HAVE 4 POSSIBLE ANSWERS, DO NOT PROVIDE MORE OR LESS, NO ANSWER SHOULD BE PROVIDED
NEVER PROVIDE IRRELEVANT OPTIONS

OUTPUT FORMAT:
### QUESTION

- Answer 1
- Answer 2
- Answer 3
- Answer 4
"""