
def retrieve_context(
    query,
    vectorstore,
    modelo=None,
    k=3
):

    try:

        
        # FILTRO POR MODELO
        if modelo:

            filtro = {
                "modelo": modelo
            }

            docs = vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filtro
            )

        else:

            docs = vectorstore.similarity_search(
                query=query,
                k=k
            )

        
        # DEBUG
        print(
            f"\n[DEBUG] Chunks recuperados: "
            f"{len(docs)}"
        )

        if not docs:

            return (
                "No se encontró información relevante."
            )

        for i, doc in enumerate(docs):

            print(f"\n--- Chunk {i+1} ---")

            print(
                doc.page_content[:400]
            )


        # BUILD CONTEXT
        context = "\n\n".join([
            doc.page_content
            for doc in docs
        ])

        return context

    except Exception as e:

        print(
            f"\n Error en retrieve_context: {e}"
        )

        return (
            "Error recuperando contexto."
        )