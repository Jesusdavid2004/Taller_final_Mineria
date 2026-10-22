"""Perfiles de negocio derivados de los centroides y resúmenes observados.

Advertencia honesta: el dataset NO contiene rentabilidad, frecuencia de compra
ni lifetime value; las recomendaciones son hipótesis de marketing basadas en
ingreso y gasto observados, no hechos medidos.
"""

CLUSTER_PROFILES = {
    0: {
        "nombre": "Clientes promedio",
        "descripcion": "Ingresos medios (~55 k$) y gasto medio (~50). Es el grupo más grande (81 clientes, 40.5 %).",
        "estrategia": "Programas de fidelización y venta cruzada para elevar gradualmente el ticket sin riesgo alto.",
        "oportunidad": "Volumen: pequeñas mejoras de gasto escalan por su tamaño.",
        "riesgo": "Grupo heterogéneo; mensajes genéricos pueden diluirse.",
    },
    1: {
        "nombre": "Clientes de alto valor",
        "descripcion": "Ingresos altos (~87 k$) y gasto alto (~82). 39 clientes (19.5 %), edad media ~33 años.",
        "estrategia": "Trato premium, lanzamientos exclusivos y recompensas por lealtad; proteger contra la competencia.",
        "oportunidad": "Mayor gasto actual; ideal para productos de alto margen y referidos.",
        "riesgo": "Expectativas altas; sensibles a caídas en calidad o servicio.",
    },
    2: {
        "nombre": "Clientes con potencial (jóvenes entusiastas)",
        "descripcion": "Ingresos bajos (~26 k$) pero gasto alto (~79). 22 clientes (11 %), edad media ~25 años.",
        "estrategia": "Facilidades de pago, gamificación y marketing digital; convertir entusiasmo en hábito de largo plazo.",
        "oportunidad": "Alta propensión a gastar; si su ingreso crece, su valor futuro puede ser grande.",
        "riesgo": "Capacidad de pago limitada; riesgo de sobreendeudamiento y abandono si suben precios.",
    },
    3: {
        "nombre": "Clientes cautelosos premium",
        "descripcion": "Ingresos altos (~88 k$) pero gasto bajo (~17). 35 clientes (17.5 %), edad media ~41 años.",
        "estrategia": "Demostrar valor y calidad (garantías, pruebas, asesoría personalizada) para desbloquear su poder adquisitivo.",
        "oportunidad": "MAYOR OPORTUNIDAD DE CRECIMIENTO: tienen el ingreso más alto y el gasto más bajo; cada punto de conversión vale mucho.",
        "riesgo": "Exigentes y comparan antes de comprar; el marketing agresivo los aleja.",
    },
    4: {
        "nombre": "Clientes de bajo valor / cautelosos",
        "descripcion": "Ingresos bajos (~26 k$) y gasto bajo (~21). 23 clientes (11.5 %), edad media ~45 años.",
        "estrategia": "Ofertas de valor, paquetes básicos y comunicación de precio justo; bajo costo de atención.",
        "oportunidad": "Sensibles a promociones puntuales que liquiden inventario.",
        "riesgo": "Bajo retorno por esfuerzo comercial; evitar inversión costosa en este segmento.",
    },
}

GROWTH_SEGMENT_ID = 3
GROWTH_JUSTIFICATION = (
    "El segmento 3 (cautelosos premium) concentra el ingreso medio más alto (~88 k$) "
    "con el gasto medio más bajo (~17). Esa brecha entre capacidad de pago y gasto actual "
    "es la mayor palanca de crecimiento: convertir aunque sea una fracción de ese poder "
    "adquisitivo genera más ingreso incremental que exprimir segmentos que ya gastan mucho "
    "o que tienen poco ingreso disponible."
)


def get_profile(cluster_id: int) -> dict:
    if cluster_id not in CLUSTER_PROFILES:
        raise ValueError(f"Cluster inválido: {cluster_id}")
    return CLUSTER_PROFILES[cluster_id]
