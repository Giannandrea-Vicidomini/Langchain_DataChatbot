import os
from utils.vector_store import load_vector_db_from_file
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def get_file_to_load():
    file_list = os.listdir("files")
    file = list(filter(lambda x: ".docx" in x or ".doc" in x,file_list))
    if len(file)!=1:
        raise Exception("The word file in the files folder must be only 1 (not counting instructions.txt)")

    file = [os.path.join("files",path) for path in file]

    return file[0]

def main():
    #si carica nell environment la chiave api di openai

    load_dotenv()
    file = get_file_to_load()
    db_path = "docs\\chroma\\"
    llm = ChatOpenAI(temperature=0.4,model="gpt-3.5-turbo")
    vector_db = load_vector_db_from_file(db_path=db_path,
                                         file_path=file)

    question = input("What is your question?: ")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever = vector_db.as_retriever()
    )
    result = qa_chain.invoke({"query":question})
    print(result["result"])
    """
    res = vector_db.max_marginal_relevance_search(question,fetch_k=5,k=3)
    print(vector_db._collection.count())
    print(res)
    print(len(res))
"""

if __name__ == "__main__":
    main()
