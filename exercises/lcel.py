import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def exercise():
    load_dotenv()
    prompt = """Tell a joke about {topic}"""
    template = PromptTemplate(input_variables=["topic"],template=prompt)
    llm = ChatGroq(temperature=1, model=os.getenv("MODEL_GROQ"))
    """
    DOES NOT WORK
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("MODEL_HUGGINGFACE"),
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    chat_model = ChatHuggingFace(llm=llm)
    """
    parser = StrOutputParser()
    chain = template | llm | parser

    topic = input("Choose topic: ")
    print(chain.invoke({"topic": topic}))

    exit(0)


if __name__ == "__main__":
    exercise()
