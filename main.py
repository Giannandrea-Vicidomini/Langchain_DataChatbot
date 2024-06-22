from utils.vector_store import load_vector_db_from_file
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def main():
    #si carica nell environment la chiave api di openai
    load_dotenv()
    llm = ChatOpenAI(temperature=0.4,model="gpt-3.5-turbo")
    vector_db = load_vector_db_from_file(db_path="docs\\chroma\\",
                                         file_path=".\\files\\My_Own_Meditation.docx")

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
