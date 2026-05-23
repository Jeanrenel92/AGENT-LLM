import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from engine.rag_pipeline import retrieve_context
from engine.utils import (
    extract_model_from_query,
    is_samsung_related
)



# SYSTEM PROMPT
SYSTEM_PROMPT = """
Eres un asistente técnico especializado en dispositivos Samsung Galaxy.

PRIORIDAD DE INFORMACIÓN
Usa principalmente la información disponible en el CONTEXTO recuperado desde la base de conocimiento.
Si el contexto incluye información sobre un modelo o procedimiento, priorízala en tu respuesta.
Evita contradecir la información del contexto.

REGLAS GENERALES
Responde de forma clara, útil y directa.
No inventes información técnica.
Si falta información en el contexto, puedes apoyarte en conocimiento general o fuentes externas confiables para consultas relacionadas con dispositivos Samsung Galaxy.

Cuando uses información externa, indícalo diciendo:
"Esta información no se encuentra en mi base de conocimiento interna, pero fue obtenida desde fuentes externas confiables."

Si no encuentras información suficiente, responde:
"No he podido encontrar la información disponible para responder completamente la consulta."

Si no es posible identificar una solución clara al problema, responde:
"Tu consulta será transferida a servicio técnico."

ESTILO DE RESPUESTA
- Responde siempre en español.
- Sé profesional y técnico.
- Mantén respuestas simples y enfocadas.
- Evita agregar información innecesaria.
- No menciones otros dispositivos salvo que sea relevante.
"""



# AGENT ORCHESTRATOR
class AgentOrchestrator:

    def __init__(self, vectorstore):

        self.vectorstore = vectorstore

        # LLM PRINCIPAL
        self.llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0.2,
            streaming=True,
        )

        # LLM CLASIFICADOR
        self.classifier_llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0,
            streaming=False,
        )

        # MEMORIA
        self.memory = ChatMessageHistory()


    # MODEL DETECTION
    def detect_model(self, query: str):

        return extract_model_from_query(query)

   
    # CLASSIFY INTENT
    

    def classify_intent(self, query: str):

        query_lower = query.lower()

        modelo_detectado = self.detect_model(query)

        
        # VALIDAR DOMINIO
        if not is_samsung_related(query):

            small_talk = [

                "hola",
                "gracias",
                "ok",
                "entiendo",
                "estas aqui",
                "estás aquí",
                "que paso",
                "qué pasó",
                "perfecto",
                "bien",
                "vale"
            ]

            if query_lower.strip() in small_talk:

                return {
                    "intent": "general",
                    "modelo_detectado": modelo_detectado,
                    "confianza": "alta"
                }

            return {
                "intent": "fuera_contexto",
                "modelo_detectado": None,
                "confianza": "alta"
            }

        
        # SOPORTE TECNICO
        soporte_words = [

            "problema",
            "falla",
            "error",
            "no funciona",
            "no carga",
            "se reinicia",
            "lento",
            "calienta",
            "pantalla negra",
            "wifi falla",
            "bluetooth falla",
            "bateria",
            "batería",
            "microfono",
            "micrófono",
            "altavoz",
            "camara",
            "cámara"
        ]

        if any(
            w in query_lower
            for w in soporte_words
        ):

            return {
                "intent": "soporte",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta"
            }

       
        # CONFIGURACION
        config_words = [

            "activar",
            "configurar",
            "desactivar",
            "ajustar",
            "habilitar",
            "cambiar",
            "modo oscuro",
            "huella",
            "nfc"
        ]

        if any(
            w in query_lower
            for w in config_words
        ):

            return {
                "intent": "configuracion",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta"
            }

        
        # COMPARACION
        compare_words = [

            "vs",
            "comparar",
            "mejor",
            "diferencia"
        ]

        if any(
            w in query_lower
            for w in compare_words
        ):

            return {
                "intent": "comparacion",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta"
            }

        
        # GENERAL
        general_words = [

            "hola",
            "gracias",
            "como estas",
            "cómo estás",
            "buenas"
        ]

        if any(
            w in query_lower
            for w in general_words
        ):

            return {
                "intent": "general",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta"
            }

       
        # CONSULTA TECNICA
        if modelo_detectado:

            return {
                "intent": "consulta_tecnica",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta"
            }

        
        # FALLBACK
        return {
            "intent": "general",
            "modelo_detectado": modelo_detectado,
            "confianza": "media"
        }

    
    # PLAN TASKS
    def plan_tasks(self, intent, modelo):

        return [

            {
                "paso": 1,
                "accion": "recuperar_contexto"
            },

            {
                "paso": 2,
                "accion": "generar_respuesta"
            }
        ]

   
    # EXECUTE PLAN
    def execute_plan(
        self,
        plan,
        query,
        intent,
        modelo=None
    ):

        print(
            f"\n[Plan generado: "
            f"{len(plan)} pasos]"
        )

        for step in plan:

            print(
                f"→ Paso {step['paso']}: "
                f"{step['accion']}"
            )

        
        # GENERAL
        if intent == "general":

            messages = [

                SystemMessage(
                    content=SYSTEM_PROMPT
                ),

                HumanMessage(
                    content=query
                )
            ]

            return self._stream_response(
                messages
            )

        # FUERA CONTEXTO
        if intent == "fuera_contexto":

            return (
                "Solo puedo ayudarte con dispositivos Samsung Galaxy."
            )

        # RECUPERAR CONTEXTO
        context = retrieve_context(
            query=query,
            vectorstore=self.vectorstore,
            modelo=modelo
        )

  
        # PROMPT FINAL
        user_message = f"""
CONTEXTO:
{context}

CONSULTA:
{query}

REGLAS:
- Usa primero la información del CONTEXTO.
- Responde solo lo necesario.
- No agregues información de otros dispositivos.
- Si no existe información suficiente, dilo claramente.
- Mantén respuestas técnicas pero simples.
"""

        history = self.memory.messages

        messages = (

            [SystemMessage(content=SYSTEM_PROMPT)]
            + history
            + [HumanMessage(content=user_message)]
        )

        response = self._stream_response(
            messages
        )

        return response

  
    # STREAM RESPONSE
    def _stream_response(self, messages):

        response_text = ""

        for chunk in self.llm.stream(messages):

            token = chunk.content

            if token:

                print(
                    token,
                    end="",
                    flush=True
                )

                response_text += token

        print()

        return response_text

    # HANDLE QUERY
    def handle_query(self, query):

        
        # RECUPERAR ULTIMO MODELO DESDE MEMORIA
        

        ultimo_modelo = None

        for msg in reversed(self.memory.messages):

            content = msg.content.lower()

            detected = self.detect_model(content)

            if detected:

                ultimo_modelo = detected
                break

        # CONSULTAS CORTAS CONTEXTUALES
    

        query_for_classification = query

        if ultimo_modelo:

            short_queries = [

                "camara",
                "cámara",
                "la camara",
                "la cámara",
                "bateria",
                "batería",
                "la bateria",
                "la batería",
                "pantalla",
                "procesador",
                "ram",
                "almacenamiento",
                "y la camara",
                "y la cámara",
                "y la bateria",
                "y la batería"
            ]

            if query.lower().strip() in short_queries:

                query_for_classification = (
                    f"{ultimo_modelo} {query}"
                )

        # ================================================
        # CLASIFICACION
        # ================================================

        classification = self.classify_intent(
            query_for_classification
        )

        intent = classification.get(
            "intent"
        )

        modelo = classification.get(
            "modelo_detectado"
        )

        confianza = classification.get(
            "confianza"
        )

        print(
            f"\n[Intención: {intent}] "
            f"[Modelo: {modelo}] "
            f"[Confianza: {confianza}]"
        )

        # ================================================
        # PLAN
        # ================================================

        plan = self.plan_tasks(
            intent,
            modelo
        )

        # ================================================
        # EJECUCION
        # ================================================

        response = self.execute_plan(
            plan,
            query,
            intent,
            modelo
        )

        # ================================================
        # MEMORIA
        # ================================================

        self.memory.add_user_message(
            query
        )

        self.memory.add_ai_message(
            response
        )

        return response