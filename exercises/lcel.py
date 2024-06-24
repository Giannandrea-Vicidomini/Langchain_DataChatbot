import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser


def exercise():
    load_dotenv()
    template = PromptTemplate.from_template("""Tell a joke about {topic}""")
    llm = ChatGroq(temperature=1, model=os.getenv("MODEL_GROQ"))
    parser = StrOutputParser()

    chain = template | llm | parser
    for t in chain.stream({"topic": "frogs"}):
        print(t)

    exit(0)


if __name__ == "__main__":
    exercise()
