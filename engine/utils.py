import re


# =========================================================
# MODELOS SOPORTADOS
# =========================================================

MODELOS_SAMSUNG: dict[str, str] = {

    # S25
    "s25 ultra": "GALAXY S25 ULTRA",
    "galaxy s25 ultra": "GALAXY S25 ULTRA",

    "s25": "GALAXY S25",
    "galaxy s25": "GALAXY S25",

    # A55
    "a55": "GALAXY A55",
    "galaxy a55": "GALAXY A55",

    # TAB S10 ULTRA
    "tab s10 ultra": "GALAXY TAB S10 ULTRA",
    "tablet s10 ultra": "GALAXY TAB S10 ULTRA",
    "tablet s 10 ultra": "GALAXY TAB S10 ULTRA",
    "tab s10": "GALAXY TAB S10 ULTRA",

    # WATCH
    "watch7": "GALAXY WATCH7",
    "galaxy watch7": "GALAXY WATCH7",

    # FOLD
    "z fold6": "GALAXY Z FOLD6",
    "fold6": "GALAXY Z FOLD6",
    "galaxy z fold6": "GALAXY Z FOLD6",
}


# =========================================================
# KEYWORDS DEL DOMINIO
# =========================================================

KEYWORDS_DOMINIO: list[str] = [

    "samsung",
    "galaxy",
    "tab",
    "tablet",

    "s25",
    "a55",
    "watch7",
    "fold6",

    "batería",
    "pantalla",
    "wifi",
    "bluetooth",
    "cámara",
    "android",
    "one ui",
    "actualización",
    "configurar",
    "reiniciar",
    "memoria",
    "almacenamiento",
    "carga",
    "aplicación",
    "notificación",
    "modo oscuro",
]


# =========================================================
# EXTRAER MODELO
# =========================================================

def extract_model_from_query(
    query: str
) -> str | None:

    query_lower = query.lower()

    for keyword, nombre in MODELOS_SAMSUNG.items():

        if keyword in query_lower:

            return nombre

    return None


# =========================================================
# VALIDAR SI ES SAMSUNG
# =========================================================

def is_samsung_related(
    query: str
) -> bool:

    query_lower = query.lower()

    return any(
        kw in query_lower
        for kw in KEYWORDS_DOMINIO
    )


# =========================================================
# FORMATEAR PLAN
# =========================================================

def format_plan_summary(
    plan: list[dict]
) -> str:

    n = len(plan)

    lines = [
        f"Plan de ejecución ({n} pasos):"
    ]

    for step in plan:

        lines.append(
            f"  {step['paso']}. "
            f"{step['accion']}"
        )

    return "\n".join(lines)


# =========================================================
# LIMPIAR RESPUESTA LLM
# =========================================================

def clean_llm_response(
    text: str
) -> str:

    text = text.strip()

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r" {2,}",
        " ",
        text
    )

    return text