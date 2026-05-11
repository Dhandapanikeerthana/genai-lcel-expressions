Name: Keerthana D
Reg No: 212224040155

## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
LangChain Expression Language (LCEL) simplifies interactions with large language models (LLMs) by creating reusable and structured expressions. This task involves:

Designing an LCEL expression with dynamic prompt parameters (e.g., topic and length).
Using three essential components: Prompt- A structured input with placeholders for parameters, Model- An LLM used to process the prompt and Output Parser- A parser to interpret the model's output.
Demonstrating the LCEL expression's functionality in generating structured, relevant outputs.

### DESIGN STEPS:

#### STEP 1:
Identify the parameters (topic and length) to allow dynamic customization of prompts
#### STEP 2:
Create a structured prompt template with placeholders for parameters.
#### STEP 3:
Use an LLM, such as OpenAI's GPT, to process the prompt.
#### STEP 4:
Design an output parser to format and structure the model's output.
#### STEP 5:
Combine the prompt template, model, and output parser into a LangChain pipeline.
#### STEP 6:
Test the LCEL expression using multiple input values for topic and length.


### PROGRAM:
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "tell me about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "rain"})
```

```

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch   

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Generative AI is a type of artificial intelligence that creates new content, such as text, images, or code, based on learned patterns.","(LCEL)LangChain Expression Language (LCEL) is a declarative syntax for composing and chaining LangChain components efficiently."],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("what is generative ai?")
retriever.get_relevant_documents("what is the full form of LCEL")
```
```
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "what is the full form of LCEL?"})
```
```
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "what is the full form of LCEL?"})
```

### OUTPUT:
<img width="1227" height="232" alt="Screenshot 2026-05-11 162137" src="https://github.com/user-attachments/assets/5a2dca91-b408-4985-ae96-caa68d4b6f21" />



<img width="1230" height="107" alt="Screenshot 2026-05-11 162153" src="https://github.com/user-attachments/assets/02e7d118-e6d4-49e7-adf5-4ba6d41711bc" />


<img width="357" height="47" alt="Screenshot 2026-05-11 162207" src="https://github.com/user-attachments/assets/9f70a65f-7909-4501-8053-5e35c6df701f" />


<img width="1317" height="122" alt="Screenshot 2026-05-11 162223" src="https://github.com/user-attachments/assets/c2e60bdf-e4eb-4e0e-8f33-ee119e1bdd36" />



### RESULT:
