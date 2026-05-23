import os

from dotenv import load_dotenv

# DOCUMENTS
from langchain_core.documents import Document

# SPLITTER
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# VECTORSTORE
from langchain_community.vectorstores import FAISS

# EMBEDDINGS
from langchain_openai import OpenAIEmbeddings


load_dotenv()


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

VECTORSTORE_PATH = os.path.join(
    BASE_DIR,
    "vectorstore"
)


# LOAD DOCUMENTS
def load_documents():

    documents = []

    print("\n Iniciando pipeline RAG...\n")

    if not os.path.exists(DATA_FOLDER):

        print(
            f" La carpeta no existe: {DATA_FOLDER}"
        )

        return []

    for file in os.listdir(DATA_FOLDER):

        # SOLO TXT
        if file.endswith(".txt"):

            path = os.path.join(
                DATA_FOLDER,
                file
            )

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    content = f.read()

               
                # NORMALIZAR NOMBRE MODELO
                modelo = (
                    file
                    .replace(".txt", "")
                    .replace("_", " ")
                    .upper()
                )

            
                # INYECTAR MODELO EN CONTENIDO
                content = f"""
MODELO: {modelo}

{content}
"""

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file,
                        "modelo": modelo
                    }
                )

                documents.append(doc)

                print(
                    f"Documento cargado: {file}"
                )

            except Exception as e:

                print(
                    f"Error cargando {file}: {e}"
                )

    print(
        f"\n📄 Documentos cargados: "
        f"{len(documents)}"
    )

    return documents


# SPLIT DOCUMENTS
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"✂️ Chunks creados: {len(chunks)}"
    )

    return chunks


# EMBEDDINGS
def create_embeddings():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url=os.getenv("GITHUB_BASE_URL")
    )

    return embeddings



# CREATE VECTORSTORE
def create_vectorstore(
    chunks,
    embeddings
):

    print(
        "\n Generando embeddings por lotes..."
    )

    BATCH_SIZE = 50

    vectorstore = None

    for i in range(
        0,
        len(chunks),
        BATCH_SIZE
    ):

        batch = chunks[
            i:i + BATCH_SIZE
        ]

        print(
            f" Procesando lote "
            f"{i} - {i + len(batch)}"
        )

        if vectorstore is None:

            vectorstore = FAISS.from_documents(
                batch,
                embeddings
            )

        else:

            vectorstore.add_documents(
                batch
            )

    
    # SAVE
    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print(
        "\n Vectorstore creado correctamente"
    )



# MAIN
def main():

    docs = load_documents()

    if not docs:

        print(
            "\n No se encontraron documentos"
        )

        return

    chunks = split_documents(docs)

    embeddings = create_embeddings()

    create_vectorstore(
        chunks,
        embeddings
    )

    print(
        "\n🎉 Pipeline finalizado"
    )


if __name__ == "__main__":
    main()