import os
from utils.vector_store import load_vector_db_from_file
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_file_to_load():
    file_list = os.listdir("files")
    file = list(filter(lambda x: ".docx" in x or ".doc" in x, file_list))
    if len(file) != 1:
        raise Exception(
            "The word file in the files folder must be only 1 (not counting instructions.txt)"
        )

    file = [os.path.join("files", path) for path in file]

    return file[0]


def main():
    done = False
    # si carica nell environment la chiave api di openai
    load_dotenv()
    file = get_file_to_load()

    llm = ChatOpenAI(
        temperature=0.4, model="gpt-3.5-turbo", max_tokens=300
    )  # si instanzia l' LLM
    vector_db = load_vector_db_from_file(
        db_path=os.getenv("DBPATH"), file_path=file
    )  # si carica il vector DB o lo si crea se non c'è
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vector_db.as_retriever()
    )  # si crea la catena che si occupa di prendere i chunk rilevanti dalla domanda e fa rispondere l'LLM

    print("REPL initialized (write quit to exit)")
    while not done:
        question = input("What is your question?: ")
        if question == "quit":
            done = True
        else:
            result = qa_chain.invoke(
                {"query": question}
            )  # si invoca la catena passandogli la domanda che sara fatta all'LLM
            lines = result["result"].split(".")
            # lines= [ token for token in vector_db.similarity_search(question)]
            for line in lines:
                print(line)
    """
    res = vector_db.max_marginal_relevance_search(question,fetch_k=5,k=3)
    print(vector_db._collection.count())
    print(res)
    print(len(res))
"""


if __name__ == "__main__":
    main()
