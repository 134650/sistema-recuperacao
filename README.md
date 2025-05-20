# Sistema de Perguntas e Respostas baseado em PDF com LangChain, FAISS e Ollama

Este projeto permite a criação de um sistema de **Perguntas e Respostas** (QA) sobre o conteúdo de um arquivo PDF. Ele utiliza as ferramentas do ecossistema LangChain, como FAISS para base vetorial e Ollama para embeddings e modelo LLM local.

## 📂 Estrutura do Projeto

├── index.py # Código principal do sistema
├── requirements.txt # Dependências do projeto
└── PPC_TSI_EaD.pdf # Documento PDF usado como base de conhecimento


## ⚙️ Requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado e rodando localmente
- Modelos suportados pelo Ollama (ex: `llama3`, `nomic-embed-text`)

## 🧠 Funcionalidades

- Carregamento e divisão do conteúdo de um PDF em trechos (chunks)
- Geração de embeddings com `nomic-embed-text` via Ollama
- Armazenamento vetorial com FAISS
- Recuperação de informações via LangChain + LLM local (`llama3`)
- Interface de linha de comando para realizar perguntas

## 🚀 Como executar

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

Crie um ambiente virtual e ative-o (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Instale as dependências:
bash
pip install -r requirements.txt

Certifique-se de que o Ollama esteja rodando com os modelos necessários:
bash
ollama run llama3
ollama run nomic-embed-text
Execute o sistema:

bash
python index.py
Digite suas perguntas com base no conteúdo do PDF. Para sair, digite sair.

📝 Exemplo
bash
Sistema de Perguntas e Respostas baseado em PDF + Ollama iniciado!

Digite sua pergunta ou 'sair' para encerrar.

Pergunta: Quais são os objetivos do curso?
Resposta: [Texto extraído do PDF com base na pergunta]

📌 Observações
O arquivo PPC_TSI_EaD.pdf pode ser substituído por qualquer outro PDF de interesse.

O desempenho e a precisão das respostas dependem da qualidade do OCR e da segmentação do texto.

🧩 Tecnologias Utilizadas
LangChain

FAISS (Facebook AI Similarity Search)

Ollama

Python 3.10+

📃 Licença
Este projeto está licenciado sob a MIT License.

yaml

---

Se quiser, posso também gerar o conteúdo do `requirements.txt` com os pacotes corretos. Deseja isso agora?


