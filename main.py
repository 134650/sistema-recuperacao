from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA


def carregar_e_processar_pdf(caminho_pdf: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Carrega um PDF, divide em trechos e gera uma base vetorial usando FAISS e embeddings locais via Ollama.
    """
    print(f"Carregando PDF: {caminho_pdf}")
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load_and_split()

    print("Dividindo o conteúdo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documentos)

    print("Gerando embeddings e criando base vetorial...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def criar_qa_chain(vectorstore):
    """
    Cria a cadeia de Pergunta e Resposta baseada em Recuperação com o modelo LLM via Ollama.
    """
    llm = OllamaLLM(model="llama3")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain


def executar_chat_qa(caminho_pdf: str):
    print("Sistema de Perguntas e Respostas baseado em PDF + Ollama iniciado!\n")

    vectorstore = carregar_e_processar_pdf(caminho_pdf)
    qa_chain = criar_qa_chain(vectorstore)

    print("Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        pergunta = input("\nPergunta: ")
        if pergunta.strip().lower() == "sair":
            print("Encerrando o sistema.")
            break

        resposta = qa_chain.invoke(pergunta)
        print("\nResposta:", resposta["result"])


if __name__ == "__main__":
    executar_chat_qa("PPC_TSI_EaD.pdf")
