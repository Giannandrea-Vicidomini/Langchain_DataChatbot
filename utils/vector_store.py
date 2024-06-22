import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_vector_db_from_file(db_path: str, file_path: str) -> Chroma:

    embedder = OpenAIEmbeddings()

    if os.path.exists(db_path) and os.path.isdir(db_path):
        print("Vector Database already exists. loading...")
        vectordb = Chroma(embedding_function=embedder, persist_directory=db_path)
    else:
        print("Creating vector db from document.")
        _, extension = os.path.splitext(file_path)
        if extension.lower() not in [".docx", ".doc"]:
            raise Exception("You have to provide a word file")

        loader = Docx2txtLoader(file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, separators=["\n", " "]
        )
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            documents=chunks, embedding=embedder, persist_directory=db_path
        )

    return vectordb


"""
 # prompt = input("write the prompt to ask:")
    #si crea il document loader per word che parsa il documento word
    # e crea un documento usabile da langchain
    word_loader = Docx2txtLoader(".\\files\\My_Own_Meditation.docx")

    #si carica il contenuto del file word in una variabile di tipo lista di Document
    docs = word_loader.load()
    print(len(docs))
    print(docs)

    #si istanzia un text splitter che serve a splittare il documento caricato in dei chunk,
    # di una dimensione e overlap determinato dai parametri
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n", " "]
    )

    #i chunk vengono creati
    chunks = splitter.split_documents(docs)
    print(chunks)

    #viene creato l embedder che crea una rappresentazione vettoriale dei chunk,
    # che potranno essere capiti dal modello
    embedder = OpenAIEmbeddings()


    #embeddings = [embedder.embed_query(chunk) for chunk in chunks]
    #print(embeddings[0])
    # print(np.dot(embeddings[0],embeddings[1]))

    #si crea la cartella che conterra tutti gli embeddings all interno di un database
    store_directory = "docs\\chroma\\"
    if os.path.exists(store_directory) and os.path.isdir(store_directory):
        print(f"Folder '{store_directory}' exists. Deleting it...")
        shutil.rmtree(store_directory)
        print(f"Folder '{store_directory}' has been deleted.")
    else:
        print(f"Folder '{store_directory}' does not exist.")

    #si crea il db che conterra tutti gli embeddings
    #si da in input i chunks creati dal text splitter,
    #l embedder instanziato
    #e la cartella in cui creare il database di vettori
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embedder, persist_directory=store_directory
    )
    print(vectordb._collection.count())

    #si fa una domanda
    question="chi è mike of the desert"

    #si puo gia fare query al database per similitudine delle frasi,
    #i vettori che sono pui simili saranno scelti, fino ad un numero indicato da k
    answer = vectordb.similarity_search(question,k=3)
    print(len(answer))
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
        res = llm.invoke(prompt)
        print("the model response is:")
        print(res.content)
        
    exit(0)

"""
