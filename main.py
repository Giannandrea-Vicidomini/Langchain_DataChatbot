import os
from utils.misc import get_file_to_load
from utils.vector_store import load_vector_db_from_file
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


def main():
    done = False
    # si carica nell environment la chiave api di openai
    load_dotenv()

    # si prepara il prompt
    prompt = """Usa i seguenti pezzi di contesto per ottenere un risultato. Se non sai la risposta, dici semplicemente che non la sai, non provare ad inventare risposte. Mantieni la risposta il più concisa possibile. 
{context}
Domanda: {question}
Risposta:"""

    # si prepara il buffer di memoria che permette al chatbot di ricordare le precedenti domande e risposte
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    p_template = PromptTemplate(input_variables=["context,question"], template=prompt)
    file = get_file_to_load()
    llm = ChatOpenAI(
        temperature=0.4, model="gpt-3.5-turbo", max_tokens=300
    )  # si instanzia l' LLM
    vector_db = load_vector_db_from_file(
        db_path=os.getenv("DBPATH"), file_path=file
    )  # si carica il vector DB o lo si crea se non c'è
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vector_db.as_retriever(),chain_type_kwargs={"prompt": p_template}
    )  # si crea la catena che si occupa di prendere i chunk rilevanti dalla domanda e fa rispondere l'LLM
    """

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=vector_db.as_retriever(), memory=memory
    )

    print("REPL initialized (write quit to exit)")
    while not done:
        question = input("What is your question?: ")
        if question == "quit":
            done = True
        else:

            """result = qa_chain.invoke(
                {"query": question}
            )  # si invoca la catena passandogli la domanda che sara fatta all'LLM
            lines = result["result"].split(".")"""
            result = qa_chain.invoke(
                question
            )  # si invoca la catena passandogli la domanda che sara fatta all'LLM
            lines = result["answer"].split(".")
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
