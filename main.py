import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from engine.agent_logic import AgentOrchestrator

# LOAD ENV VARIABLES
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL")


# LOAD EMBEDDINGS
def load_embeddings() -> OpenAIEmbeddings:

    try:

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE_URL,
        )

        return embeddings

    except Exception as e:

        print(
            f"\n Error cargando embeddings:\n{e}"
        )

        raise SystemExit(1)


# LOAD VECTORSTORE

def load_vectorstore(
    embeddings: OpenAIEmbeddings
) -> FAISS:

    try:

        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        return vectorstore

    except Exception as e:

        print(
            f"\n Error cargando vectorstore:\n{e}"
        )

        print(
            "\n Ejecuta primero:"
        )

        print(
            "python vectorstore.py"
        )

        raise SystemExit(1)



# MAIN
def main() -> None:

    print("=" * 60)
    print("Samsung AI Agent - Asistente Virtual para Soporte Técnico")
    print("=" * 60)
   
    # INIT
    

    embeddings = load_embeddings()

    vectorstore = load_vectorstore(
        embeddings
    )

    agent = AgentOrchestrator(
        vectorstore
    )

    print(
        "Agente inicializado correctamente\n"
    )
    print("Hola! Soy tu asistente virtual,\n ¿En qué puedo ayudarte hoy?")

    # CHAT LOOP
    while True:

        try:

            query = input(
                "Tú: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\n\n Cerrando agente..."
            )

            break

       # EMPTY INPUT
        if not query:
            continue

        # EXIT CONDITIONS

        if query.lower() in (
            "salir",
            "exit",
            "quit"
        ):

            print(
                "\n Hasta pronto!"
            )

            break

        
        # AGENT EXECUTION
        try:

            print(
                "\nAgente:\n"
            )

            response = agent.handle_query(
                query
            )

            print(
                "\n" + "=" * 60 + "\n"
            )

        except Exception as e:

            print(
                f"\nError procesando consulta:\n{e}"
            )

            print(
                "\n" + "=" * 60 + "\n"
            )



# ENTRYPOINT
if __name__ == "__main__":
    main()