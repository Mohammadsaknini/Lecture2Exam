
FREE_TEXT_QUETIONS = """
Create exam questions based for the topic : {topic}

Lecture content:
{content}

Generate {num_questions} questions based on the content above, make sure they are relevant and challenging.

DO NOT INTRODUCE ANY QUESTIONS THAT ARE NOT BASED ON THE CONTENT ABOVE
THE GENERTED QUESTIONS SHOULD NOT ASKED ABOUT RESEARCH PAPERS OR AUTHORS
NEVER PROVIDE AN ANSWER
"""


CODE_QUESTION = """
Create exam questions based for the topic : {topic}

Lecture content:
{content}
Generate a single coding question based on the content above, make sure it is relevant and challenging.
YOU MUST ALWAYS PROVIDE A CODE SNIPPET IN THE QUESTION AND ASK A QUESTION BASED ON IT 
THE QUESTIONS SHOULD NOT REQUIERE ANY ACTUAL CODING, JUST UNDERSTANDING OF THE CODE
NEVER PROVIDE AN ANSWER
"""

MC_QUESTION = """
Create exam questions based for the topic : {topic}

Lecture content:
{content}

Generate a single multiple choice question based on the content above, make sure it is relevant and challenging.
THE QUESTION SHOULD HAVE 4 POSSIBLE ANSWERS, DO NOT PROVIDE MORE OR LESS, NO ANSWER SHOULD BE PRPOVIDED
NEVER PROVIDE IRRELEVANT OPTIONS

OUTPUT FORMAT:
A) Answer 1
B) Answer 2
C) Answer 3
D) Answer 4
"""