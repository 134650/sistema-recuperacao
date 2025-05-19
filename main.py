import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def carregar_textos(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return f.read()

def criar_banco_vetorial(texto, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(texto)

    embeddings = OpenAIEmbeddings()
    vetor_db = FAISS.from_texts(chunks, embeddings)

    return vetor_db

def criar_retrieval_qa(vetor_db):
    llm = OpenAI(temperature=0)
    retriever = vetor_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

def main():
    # Carregar texto
    texto = carregar_textos('../data/textos.txt')

    # Criar banco vetorial
    vetor_db = criar_banco_vetorial(texto)

    # Criar pipeline RetrievalQA
    qa = criar_retrieval_qa(vetor_db)

    print("Sistema de Recuperação de Informações")
    print("Digite 'sair' para encerrar.")
    while True:
        pergunta = input("\nFaça sua pergunta: ")
        if pergunta.lower() == 'sair':
            break
        resposta = qa.run(pergunta)
        print("\nResposta:", resposta)

if __name__ == "__main__":
    main()
