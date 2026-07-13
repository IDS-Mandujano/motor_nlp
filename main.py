"""
SafeRoute - Motor NLP v2.0
Búsqueda con TF-IDF + BM25 + expansión de consulta
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import logging
import os
import uvicorn
from datetime import datetime
import requests
from dotenv import load_dotenv
import math
from collections import Counter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SafeRoute - Motor NLP",
    description="Búsqueda de reportes con TF-IDF + BM25",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "")

# ============================================================
# MODELOS
# ============================================================

class BusquedaRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    metodo: str = Field(default="bm25", description="bm25 o tfidf")

class ReporteResultado(BaseModel):
    id: str
    texto: str
    tipo: str
    ruta_id: str
    score: float
    timestamp: str

class BusquedaResponse(BaseModel):
    resultados: List[ReporteResultado]
    total: int
    consulta: str
    consulta_expandida: str
    tiempo_ms: float

class TopicosResponse(BaseModel):
    topicos: List[dict]
    total_reportes: int
    topico_dominante: dict
    fecha_analisis: str

class ReporteIngestRequest(BaseModel):
    id: str
    texto: str = ""
    tipo: str = "otro"
    ruta_id: str = ""
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    timestamp: Optional[str] = None
    vigente: bool = True
    confirmaciones: int = 0
    evento: Optional[str] = None

class ReporteEstadoRequest(BaseModel):
    id: str
    vigente: bool
    evento: Optional[str] = None
    timestamp: Optional[str] = None

class ClasificarRequest(BaseModel):
    texto: str = Field(..., min_length=5, max_length=1000)
    tipo_actual: Optional[str] = None

class ClasificarResponse(BaseModel):
    texto: str
    tipo_predicho: str
    confianza: float
    modelo: str
    similitudes: Optional[dict] = None
    tipo_actual: Optional[str] = None
    coincide: Optional[bool] = None

class NERRequest(BaseModel):
    texto: str = Field(..., min_length=5, max_length=1000)

class NERResponse(BaseModel):
    texto: str
    tipo_incidente: str
    severidad: str
    entidades: dict
    resumen: str

# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "a", "al", "por", "con", "sin", "para",
    "que", "es", "fue", "son", "esta", "estan", "hay", "muy",
    "mas", "ya", "desde", "hasta", "entre", "sobre", "todo",
    "pero", "o", "y", "ni", "se", "lo", "le", "su", "me", "mi",
    "como", "cuando", "donde", "tambien", "porque", "cada",
    "ese", "esa", "eso", "este", "esto", "aquel",
    "solo", "hace", "hacia", "ahi", "alli",
    "tiene", "tenia", "han", "habia", "era", "eran", "ser",
    "pero", "aun", "vez", "mismo",
}

# ============================================================
# MAPA SEMÁNTICO MEJORADO
# ============================================================

MAPA_SEMANTICO = {
    # Agua/Inundación
    "inundaciones": "inundacion", "inundada": "inundacion", "inundado": "inundacion",
    "anegada": "inundacion", "anegado": "inundacion", "encharcamiento": "inundacion",
    "desbordamiento": "inundacion", "desbordado": "inundacion", "desbordo": "inundacion",
    "hidrico": "agua", "pluvial": "lluvia", "precipitacion": "lluvia",
    "torrencial": "lluvia", "aguacero": "lluvia", "llovizna": "lluvia",
    "creciente": "rio", "cauce": "rio",
    # Accidentes
    "accidentes": "accidente", "colision": "choque", "colisiono": "choque",
    "impacto": "choque", "impactaron": "choque",
    "volcadura": "volcado", "volco": "volcado", "volcaron": "volcado",
    "siniestro": "accidente", "percance": "accidente",
    "heridos": "herido", "lesionados": "herido",
    # Baches
    "baches": "bache", "socavon": "bache", "hoyo": "bache", "hoyos": "bache",
    "ponchado": "bache", "ponchados": "bache", "ponchadura": "bache",
    "bacheo": "bache",
    # Derrumbes
    "derrumbes": "derrumbe", "deslave": "derrumbe", "deslaves": "derrumbe",
    "desprendimiento": "derrumbe", "alud": "derrumbe",
    "bloqueada": "bloqueo", "bloqueado": "bloqueo", "obstruido": "bloqueo",
    "rocas": "piedra", "piedras": "piedra",
    # Infraestructura
    "asfaltico": "pavimento", "deterioro": "danado", "deteriorado": "danado",
    "destruido": "danado", "roto": "danado", "rota": "danado",
    "iluminacion": "sin_luz", "alumbrado": "sin_luz", "apagon": "sin_luz",
    "oscuro": "sin_luz", "oscura": "sin_luz",
    # Visibilidad
    "neblina": "niebla", "bruma": "niebla",
    "visibilidad": "niebla",
    # Vehículos
    "vehiculos": "vehiculo", "automovil": "vehiculo", "automoviles": "vehiculo",
    "camion": "camion", "camiones": "camion", "trailer": "camion",
    "motocicleta": "moto", "motociclistas": "moto",
    "taxi": "taxi", "taxis": "taxi",
    # Lugares
    "carretera": "carretera", "autopista": "carretera", "federal": "carretera",
    "tramo": "tramo", "tramos": "tramo",
    "puente": "puente", "puentes": "puente",
    "curva": "curva", "curvas": "curva",
    "cruce": "crucero", "cruzando": "crucero", "interseccion": "crucero",
    "entrada": "entrada", "salida": "salida",
    "km": "km", "kilometro": "km",
    "libramiento": "libramiento",
    "zona": "zona", "area": "zona",
    # Severidad
    "grave": "grave", "graves": "grave",
    "leve": "leve", "leves": "leve",
    "total": "total", "parcial": "parcial",
    "cerrado": "cerrado", "cerrada": "cerrado",
    "intransitable": "bloqueo", "imposible": "bloqueo",
}

# ============================================================
# EXPANSIÓN DE CONSULTA
# ============================================================

EXPANSION_CONSULTA = {
    "inundacion": ["agua", "lluvia", "desbordado", "anegada", "inundada"],
    "accidente": ["choque", "volcado", "herido", "colision", "siniestro"],
    "bache": ["hoyo", "socavon", "ponchado", "roto", "danado"],
    "derrumbe": ["deslave", "piedra", "roca", "bloqueo", "cerrado"],
    "sin_luz": ["oscuro", "apagon", "iluminacion", "noche"],
    "niebla": ["neblina", "visibilidad", "cerrada"],
}

# ============================================================
# CORPUS AMPLIADO
# ============================================================

CORPUS_BASE = [
    # INUNDACIONES
    {"id": "rep-001", "texto": "inundación severa en Suchiapa km 12 agua cubre ambos carriles", "tipo": "inundacion", "ruta_id": "tuxtla-suchiapa", "timestamp": "2026-06-10T08:15:00Z"},
    {"id": "rep-002", "texto": "inundación repentina en la zona baja de Berriozábal el agua subió medio metro", "tipo": "inundacion", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-10T09:30:00Z"},
    {"id": "rep-003", "texto": "inundación en campos agrícolas cerca de Teopisca pérdida de cosecha", "tipo": "inundacion", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-10T07:45:00Z"},
    {"id": "rep-004", "texto": "inundación en la carretera a Comitán después de la lluvia torrencial", "tipo": "inundacion", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-11T14:00:00Z"},
    {"id": "rep-005", "texto": "inundación en el puente de Suchiapa no se puede pasar el agua tapa todo", "tipo": "inundacion", "ruta_id": "tuxtla-suchiapa", "timestamp": "2026-06-11T08:00:00Z"},
    {"id": "rep-021", "texto": "el río se desbordó cerca de la carretera a Suchiapa inundando los campos", "tipo": "inundacion", "ruta_id": "tuxtla-suchiapa", "timestamp": "2026-06-12T06:00:00Z"},
    {"id": "rep-022", "texto": "lluvia intensa en Berriozábal calles anegadas y carretera inundada en zona baja", "tipo": "inundacion", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-12T14:00:00Z"},
    {"id": "rep-023", "texto": "agua estancada en la entrada a Suchiapa desde ayer no drena el alcantarillado", "tipo": "inundacion", "ruta_id": "tuxtla-suchiapa", "timestamp": "2026-06-13T08:00:00Z"},
    
    # ACCIDENTES
    {"id": "rep-006", "texto": "choque múltiple tres vehículos en el puente Tuxtla Berriozábal", "tipo": "accidente", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-10T10:00:00Z"},
    {"id": "rep-007", "texto": "accidente grave camión de carga volcado en la salida a Suchiapa heridos", "tipo": "accidente", "ruta_id": "tuxtla-suchiapa", "timestamp": "2026-06-11T06:30:00Z"},
    {"id": "rep-008", "texto": "accidente motociclista sin casco en Berriozábal heridas graves", "tipo": "accidente", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-12T11:00:00Z"},
    {"id": "rep-009", "texto": "choque leve entre taxi y particular en el centro de Tuxtla solo daños materiales", "tipo": "accidente", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-12T16:00:00Z"},
    {"id": "rep-010", "texto": "accidente en el crucero de Teopisca dos vehículos involucrados", "tipo": "accidente", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-11T09:00:00Z"},
    {"id": "rep-024", "texto": "colisión frontal en la carretera a Comitán dos camiones de carga involucrados", "tipo": "accidente", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-13T05:00:00Z"},
    {"id": "rep-025", "texto": "volcadura de tráiler en la curva del km 18 mercancía regada en la carretera", "tipo": "accidente", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-13T11:00:00Z"},
    
    # BACHES
    {"id": "rep-011", "texto": "bache enorme en la curva de Teopisca carril derecho completamente destruido", "tipo": "bache", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-10T12:00:00Z"},
    {"id": "rep-012", "texto": "baches múltiples en la carretera federal km 23 entre Tuxtla y Chiapa de Corzo", "tipo": "bache", "ruta_id": "tuxtla-chiapa-corzo", "timestamp": "2026-06-11T15:00:00Z"},
    {"id": "rep-013", "texto": "bache profundo en la entrada a Comitán rompió llanta de varios vehículos", "tipo": "bache", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-12T08:00:00Z"},
    {"id": "rep-014", "texto": "baches en toda la ruta de San Cristóbal a Teopisca está intransitable", "tipo": "bache", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-12T10:00:00Z"},
    {"id": "rep-026", "texto": "hoyo gigante en el libramiento de Berriozábal varios carros ponchados", "tipo": "bache", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-13T09:00:00Z"},
    {"id": "rep-027", "texto": "socavón en la carretera federal a la altura de Chiapa de Corzo peligroso", "tipo": "bache", "ruta_id": "tuxtla-chiapa-corzo", "timestamp": "2026-06-13T16:00:00Z"},
    
    # DERRUMBES
    {"id": "rep-015", "texto": "derrumbe por lluvia en la sierra de Comitán bloqueo total de la carretera", "tipo": "derrumbe", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-13T05:00:00Z"},
    {"id": "rep-016", "texto": "derrumbe de rocas grandes en la carretera a Motozintla después de la lluvia", "tipo": "derrumbe", "ruta_id": "comitan-motozintla", "timestamp": "2026-06-13T07:00:00Z"},
    {"id": "rep-017", "texto": "derrumbe en la curva del km 18 ruta a Comitán cayó mucha piedra", "tipo": "derrumbe", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-13T06:00:00Z"},
    {"id": "rep-028", "texto": "deslave en la sierra cerca de Teopisca bloqueo parcial un carril cerrado", "tipo": "derrumbe", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-14T06:00:00Z"},
    
    # OTROS
    {"id": "rep-018", "texto": "niebla densa en el tramo de montaña entre San Cristóbal y Teopisca cero visibilidad", "tipo": "niebla", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-10T06:00:00Z"},
    {"id": "rep-019", "texto": "sin iluminación todo el tramo de San Cristóbal a Teopisca muy peligroso de noche", "tipo": "sin_luz", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-11T20:00:00Z"},
    {"id": "rep-020", "texto": "pavimento completamente mojado y deslizado por las lluvias en la curva del km 18", "tipo": "otro", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-12T14:00:00Z"},
    {"id": "rep-029", "texto": "apagón en todo el tramo Tuxtla Berriozábal sin luz en las luminarias", "tipo": "sin_luz", "ruta_id": "tuxtla-berriozabal", "timestamp": "2026-06-14T19:00:00Z"},
    {"id": "rep-030", "texto": "neblina cerrada en la carretera a Comitán no se ve nada después del puente", "tipo": "niebla", "ruta_id": "san-cristobal-comitan", "timestamp": "2026-06-14T05:00:00Z"},
]

# ============================================================
# NER - RECONOCIMIENTO DE ENTIDADES NOMBRADAS
# ============================================================

ENTIDADES_CHIAPAS = {
    "LUGAR": [
        "Suchiapa", "Berriozábal", "Berriozabal", "Teopisca", "Comitán", "Comitan",
        "Tuxtla", "Tuxtla Gutiérrez", "San Cristóbal", "San Cristobal",
        "Chiapa de Corzo", "Tapachula", "Palenque", "Motozintla", "Tonalá", "Tonala",
        "Villaflores", "Catazajá", "Catazaja", "Playas de Catazajá",
    ],
    "UBICACION": [
        "km 12", "km 8", "km 15", "km 18", "km 22", "km 23", "km 40",
        "puente", "curva", "libramiento", "carretera federal", "autopista",
        "entrada", "salida", "carril derecho", "carril izquierdo",
        "ambos carriles", "paso a desnivel", "crucero", "avenida central",
        "zona baja", "zona centro", "mercado", "gasolinera", "embarcadero",
    ],
    "SEVERIDAD": [
        "severa", "grave", "graves", "leve", "leves", "total", "parcial",
        "múltiple", "multiples", "crítico", "critico", "profundo", "profunda",
        "gigante", "enorme", "grandes", "intransitable", "completamente",
        "medio metro", "un metro", "cero visibilidad",
    ],
    "INFRAESTRUCTURA": [
        "puente", "carril derecho", "carril izquierdo", "ambos carriles",
        "curva", "libramiento", "paso a desnivel", "carretera", "autopista",
        "avenida", "calle", "entrada", "salida", "drenaje", "alcantarillado",
        "luminarias", "pavimento", "asfalto", "talud", "barranco",
    ],
    "VEHICULOS": [
        "camión", "camiones", "camion", "camiones", "taxi", "taxis",
        "moto", "motociclista", "motociclistas", "tráiler", "trailer",
        "vehículos", "vehiculos", "autobús", "autobus", "camioneta",
        "volteo", "volteos", "particular", "auto", "carro", "carros",
    ],
    "CONSECUENCIA": [
        "heridos", "herido", "muertos", "muerto", "bloqueo total",
        "bloqueo parcial", "bloqueada", "bloqueado", "daños materiales",
        "daños", "tráfico detenido", "tráfico lento", "intransitable",
        "no se puede pasar", "solo pasan", "pérdida de cosecha",
        "pérdida", "rompió llanta", "ponchados", "ponchado",
    ],
    "CONDICION": [
        "lluvia", "lluvias", "lluvia torrencial", "niebla", "neblina",
        "mojado", "deslizado", "resbaloso", "noche", "madrugada",
        "sin iluminación", "sin luz", "apagón", "oscuridad",
    ],
}

class NERChiapas:
    """
    Reconocedor de Entidades Nombradas para el dominio vial chiapaneco.
    Basado en reglas + diccionario.
    """
    
    def __init__(self):
        self.patrones = {}
        for categoria, entidades in ENTIDADES_CHIAPAS.items():
            for entidad in entidades:
                key = entidad.lower()
                self.patrones[key] = categoria
    
    def extraer_entidades(self, texto: str) -> dict:
        texto_lower = texto.lower()
        entidades_encontradas = {cat: [] for cat in ENTIDADES_CHIAPAS}
        
        for entidad, categoria in self.patrones.items():
            if entidad in texto_lower:
                if entidad not in entidades_encontradas[categoria]:
                    entidades_encontradas[categoria].append(entidad)
        
        return {k: v for k, v in entidades_encontradas.items() if v}
    
    def extraer_tipo_incidente(self, texto: str) -> str:
        texto_lower = texto.lower()

        if any(p in texto_lower for p in ["inundación", "inundacion", "agua cubre", "desbord", "anegada"]):
            return "inundacion"
        if any(p in texto_lower for p in ["derrumbe", "deslave", "rocas", "piedras", "bloqueo total"]):
            return "derrumbe"
        if any(p in texto_lower for p in ["accidente", "choque", "volcadura", "atropell", "colisión"]):
            return "accidente"
        if any(p in texto_lower for p in ["bache", "hoyo", "socavón", "ponchado", "baches"]):
            return "bache"
        if any(p in texto_lower for p in ["sin iluminación", "sin iluminacion", "sin luz", "apagón", "apagon", "oscur"]):
            return "sin_luz"
        if any(p in texto_lower for p in ["niebla", "neblina", "visibilidad"]):
            return "niebla"
        return "otro"
    
    def extraer_severidad(self, texto: str) -> str:
        texto_lower = texto.lower()
        
        if any(p in texto_lower for p in ["severa", "grave", "total", "crítico", "critico", 
                                           "bloqueo total", "no se puede pasar", "cero visibilidad",
                                           "un metro", "intransitable","muy peligroso"]):
            return "critica"
        if any(p in texto_lower for p in ["múltiple", "multiples", "medio metro", "profundo",
                                           "gigante", "enorme", "grandes", "bloqueo parcial"]):
            return "alta"
        if any(p in texto_lower for p in ["leve", "parcial", "daños materiales", "solo daños"]):
            return "media"
        return "baja"
    
    def generar_resumen_entidades(self, texto: str) -> str:
        entidades = self.extraer_entidades(texto)
        tipo = self.extraer_tipo_incidente(texto)
        severidad = self.extraer_severidad(texto)
        
        partes = []
        partes.append(f"TIPO: {tipo}")
        partes.append(f"SEVERIDAD: {severidad}")
        
        if "LUGAR" in entidades:
            partes.append(f"LUGAR: {', '.join(entidades['LUGAR'][:3])}")
        if "UBICACION" in entidades:
            partes.append(f"UBICACIÓN: {', '.join(entidades['UBICACION'][:3])}")
        if "VEHICULOS" in entidades:
            partes.append(f"VEHÍCULOS: {', '.join(entidades['VEHICULOS'][:3])}")
        if "INFRAESTRUCTURA" in entidades:
            partes.append(f"AFECTACIÓN: {', '.join(entidades['INFRAESTRUCTURA'][:3])}")
        if "CONSECUENCIA" in entidades:
            partes.append(f"CONSECUENCIA: {', '.join(entidades['CONSECUENCIA'][:3])}")
        
        return " | ".join(partes)

ner_chiapas = NERChiapas()

# ============================================================
# CLASIFICACIÓN POR REGLAS
# ============================================================

def clasificar_por_reglas(texto: str) -> dict:
    """Clasificación por palabras clave."""
    texto_lower = texto.lower()
    
    reglas = [
        (["inundación", "inundacion", "agua", "desbord", "lluvia", "anegada", "inundada"], "inundacion"),
        (["accidente", "choque", "volcadura", "atropell", "colisión", "impacto"], "accidente"),
        (["bache", "hoyo", "socavón", "ponchado", "roto", "dañado"], "bache"),
        (["derrumbe", "deslave", "piedra", "roca", "bloqueo", "cerrado"], "derrumbe"),
        (["sin luz", "sin_luz", "apagón", "oscur", "iluminación"], "sin_luz"),
        (["niebla", "neblina", "visibilidad"], "niebla"),
    ]
    
    for palabras, tipo in reglas:
        if any(p in texto_lower for p in palabras):
            return {"tipo": tipo, "confianza": 0.8, "modelo": "reglas"}
    
    return {"tipo": "otro", "confianza": 0.5, "modelo": "reglas"}

# ============================================================
# BM25 RANKER
# ============================================================

class BM25Ranker:
    """
    Implementación BM25 para ranking de documentos.
    
    BM25(t, d) = IDF(t) × (f(t,d) × (k₁ + 1)) / (f(t,d) + k₁ × (1 - b + b × |d|/avgdl))
    
    Parámetros:
    - k₁: controla la saturación de TF (1.2-2.0 típico)
    - b: controla la normalización por longitud (0.75 típico)
    """
    
    def __init__(self, corpus_textos, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_textos
        self.N = len(corpus_textos)
        
        self.doc_lens = [len(doc.split()) for doc in corpus_textos]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        
        self.idf = {}
        self._calcular_idf()
        
        logger.info(f"BM25 inicializado: k1={k1}, b={b}, N={self.N}, avgdl={self.avgdl:.1f}")
    
    def _calcular_idf(self):
        df = {}
        
        for doc in self.corpus:
            terminos_unicos = set(doc.split())
            for termino in terminos_unicos:
                df[termino] = df.get(termino, 0) + 1
        
        for termino, freq in df.items():
            self.idf[termino] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def _score_documento(self, query_tokens, doc_idx):
        doc = self.corpus[doc_idx]
        doc_tokens = doc.split()
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        doc_counter = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = doc_counter.get(token, 0)
            if tf == 0:
                continue
            
            idf_val = self.idf[token]
            
            numerador = tf * (self.k1 + 1)
            denominador = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf_val * numerador / denominador
        
        return score
    
    def rank(self, query_texto, top_k=5):
        query_tokens = query_texto.split()
        
        scores = []
        for doc_idx in range(self.N):
            score = self._score_documento(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ============================================================
# FUNCIONES DE CARGA DE REPORTES
# ============================================================

def cargar_reportes_reales():
    try:
        resp = requests.get(
            f"{GATEWAY_URL}/api/internal/reportes?vigente=true&limit=200",
            headers={"X-Internal-API-Key": GATEWAY_API_KEY},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            reportes = data.get("reportes", [])
            logger.info(f"Cargados {len(reportes)} reportes reales desde Gateway")
            return [
                {
                    "id": r["id"],
                    "texto": r.get("nota_voz") or r["tipo"],
                    "tipo": r["tipo"],
                    "ruta_id": r["ruta_id"],
                    "timestamp": r["timestamp"]
                }
                for r in reportes
            ]
    except Exception as e:
        logger.warning(f"No se pudieron cargar reportes del Gateway: {e}")
    return []

# ============================================================
# MOTOR NLP
# ============================================================

class MotorNLP:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = None
        self.matriz_tfidf = None
        self.bm25 = None
        self.textos_originales = [doc["texto"] for doc in corpus]
        self.textos_procesados = []
        self._preprocesar_corpus()
        self.ultima_actualizacion = None

    def _preprocesar_corpus(self):
        self.textos_procesados = []
        for doc in self.corpus:
            tokens = self._preprocesar_texto(doc["texto"])
            self.textos_procesados.append(" ".join(tokens))
        logger.info(f"Corpus preprocesado: {len(self.textos_procesados)} documentos")

    def _preprocesar_texto(self, texto: str) -> List[str]:
        texto = texto.lower().strip()
        
        reemplazos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'}
        for acento, sin_acento in reemplazos.items():
            texto = texto.replace(acento, sin_acento)
        
        texto = re.sub(r'[^\w\s]', ' ', texto)
        texto = re.sub(r'\d+', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        tokens = texto.split()
        
        tokens_limpios = []
        for t in tokens:
            if t in STOPWORDS or len(t) < 3:
                continue
            token_mapeado = MAPA_SEMANTICO.get(t, t)
            if token_mapeado.endswith('ciones'):
                token_mapeado = token_mapeado[:-6] + 'cion'
            elif token_mapeado.endswith('es') and len(token_mapeado) > 5:
                token_mapeado = token_mapeado[:-2]
            elif token_mapeado.endswith('s') and not token_mapeado.endswith('ss') and len(token_mapeado) > 5:
                token_mapeado = token_mapeado[:-1]
            tokens_limpios.append(token_mapeado)
        
        return tokens_limpios

    def _expandir_consulta(self, tokens: List[str]) -> List[str]:
        expandidos = list(tokens)
        for token in tokens:
            if token in EXPANSION_CONSULTA:
                expandidos.extend(EXPANSION_CONSULTA[token])
        return expandidos

    def inicializar_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1,
            sublinear_tf=True,
        )
        self.matriz_tfidf = self.vectorizer.fit_transform(self.textos_procesados)
        logger.info(f"TF-IDF: {self.matriz_tfidf.shape[0]} docs × {self.matriz_tfidf.shape[1]} términos")

    def buscar(self, query: str, top_k: int = 5, metodo: str = "bm25"):
        import time
        inicio = time.time()
    
        if self.vectorizer is None:
            self.inicializar_tfidf()
    
        lugar_buscado = None
        tokens_ubicacion = []
        
        patrones_lugar = [
            r'en\s+([a-záéíóúñ\s]+?)(?:\s+(?:con|por|despues|después|muy|bastante|varios|varias|todo|toda|km\d*|$))',
            r'cerca\s+de\s+([a-záéíóúñ\s]+?)(?:\s+(?:con|por|despues|después|muy|bastante|varios|varias|todo|toda|km\d*|$))',
            r'en\s+la\s+([a-záéíóúñ\s]+?)(?:\s+(?:con|por|despues|después|muy|bastante|varios|varias|todo|toda|km\d*|$))',
        ]
        
        query_lower = query.lower()
        for patron in patrones_lugar:
            match = re.search(patron, query_lower)
            if match:
                lugar_buscado = match.group(1).strip()
                break
            
        if not lugar_buscado:
            ciudades_conocidas = [
                "san cristobal", "san cristóbal", "tuxtla", "tuxtla gutierrez", 
                "tuxtla gutiérrez", "berriozabal", "berriozábal", "suchiapa",
                "chiapa de corzo", "comitan", "comitán", "teopisca", "villaflores",
                "tonala", "tonalá", "motozintla", "tapachula", "palenque",
                "ocozocoautla", "chicoasen", "chicoasén", "las margaritas",
            ]
            for ciudad in ciudades_conocidas:
                if ciudad in query_lower:
                    lugar_buscado = ciudad
                    break
                
        if lugar_buscado:
            logger.info(f"📍 Lugar detectado en consulta: '{lugar_buscado}'")
            tokens_ubicacion = self._preprocesar_texto(lugar_buscado)
    
        tokens = self._preprocesar_texto(query)
        
        tokens_no_ubicacion = [t for t in tokens if t not in tokens_ubicacion]
        
        tokens_expandidos = self._expandir_consulta(tokens_no_ubicacion)
        
        todos_tokens = tokens_expandidos + tokens_ubicacion
        consulta_expandida = " ".join(todos_tokens)
    
        if metodo == "bm25":
            if self.bm25 is None:
                self.bm25 = BM25Ranker(self.textos_procesados, k1=1.5, b=0.75)
    
            resultados_bm25 = self.bm25.rank(consulta_expandida, top_k=top_k * 2)
    
            resultados = []
            for doc_idx, score in resultados_bm25:
                doc = self.corpus[doc_idx]
                
                if lugar_buscado:
                    doc_texto = doc["texto"].lower()
                    doc_ruta = doc.get("ruta_id", "").lower()
                    
                    menciona_lugar = False
                    
                    partes_lugar = lugar_buscado.split()
                    if all(parte in doc_texto for parte in partes_lugar):
                        menciona_lugar = True
                    elif lugar_buscado in doc_texto:
                        menciona_lugar = True
                    elif any(parte in doc_ruta for parte in partes_lugar):
                        menciona_lugar = True
                    
                    if not menciona_lugar:
                        continue
                    
                resultados.append(ReporteResultado(
                    id=doc["id"],
                    texto=doc["texto"],
                    tipo=doc["tipo"],
                    ruta_id=doc["ruta_id"],
                    score=round(float(score), 4),
                    timestamp=doc["timestamp"]
                ))
                
                if len(resultados) >= top_k:
                    break
        else:
            query_vec = self.vectorizer.transform([consulta_expandida])
            similitudes = cosine_similarity(query_vec, self.matriz_tfidf).flatten()
    
            UMBRAL_MIN = 0.03
            indices = similitudes.argsort()[::-1]
    
            resultados = []
            for idx in indices:
                score = similitudes[idx]
                if score < UMBRAL_MIN:
                    continue
                
                doc = self.corpus[idx]
                
                if lugar_buscado:
                    doc_texto = doc["texto"].lower()
                    if lugar_buscado not in doc_texto:
                        partes_lugar = lugar_buscado.split()
                        if not all(parte in doc_texto for parte in partes_lugar):
                            continue
                        
                resultados.append(ReporteResultado(
                    id=doc["id"],
                    texto=doc["texto"],
                    tipo=doc["tipo"],
                    ruta_id=doc["ruta_id"],
                    score=round(float(score), 4),
                    timestamp=doc["timestamp"]
                ))
                
                if len(resultados) >= top_k:
                    break
                
        tiempo_ms = (time.time() - inicio) * 1000
        logger.info(f"Búsqueda [{metodo}]: '{query}' → {len(resultados)} resultados en {tiempo_ms:.1f}ms")
        return resultados, tiempo_ms, consulta_expandida

    def upsert_reporte(self, reporte: dict):
        reporte_id = reporte.get("id")
        if not reporte_id:
            return False

        vigente = reporte.get("vigente", True)
        self.corpus = [doc for doc in self.corpus if doc.get("id") != reporte_id]

        if vigente:
            texto = reporte.get("texto") or reporte.get("nota_voz") or reporte.get("tipo", "otro")
            self.corpus.append({
                "id": reporte_id,
                "texto": texto,
                "tipo": reporte.get("tipo", "otro"),
                "ruta_id": reporte.get("ruta_id", ""),
                "timestamp": reporte.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latitud": reporte.get("latitud"),
                "longitud": reporte.get("longitud"),
                "vigente": True,
            })

        self._reconstruir_indices()
        return True

    def actualizar_estado_reporte(self, reporte_id: str, vigente: bool):
        encontrado = False
        if not vigente:
            original_len = len(self.corpus)
            self.corpus = [doc for doc in self.corpus if doc.get("id") != reporte_id]
            encontrado = len(self.corpus) != original_len
        else:
            for doc in self.corpus:
                if doc.get("id") == reporte_id:
                    doc["vigente"] = True
                    encontrado = True
                    break

        if encontrado:
            self._reconstruir_indices()
        return encontrado

    def _reconstruir_indices(self):
        self.textos_originales = [doc["texto"] for doc in self.corpus]
        self._preprocesar_corpus()
        self.vectorizer = None
        self.matriz_tfidf = None
        self.bm25 = None
        if len(self.corpus) > 0:
            self.inicializar_tfidf()
        self.ultima_actualizacion = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"Indices NLP actualizados: {len(self.corpus)} reportes")

    def obtener_topicos(self, n_topicos=5):
        conteo_tipos = {}
        for doc in self.corpus:
            tipo = doc.get("tipo", "otro")
            conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + 1

        total = len(self.corpus)
        nombres_tipos = {
            "inundacion": ("Inundaciones", "Desviar rutas por vías alternas elevadas"),
            "bache": ("Baches", "Reducir velocidad y reportar a mantenimiento"),
            "accidente": ("Accidentes", "Reforzar manejo defensivo"),
            "derrumbe": ("Derrumbes", "Evitar rutas de sierra"),
            "sin_luz": ("Sin iluminación", "Evitar rutas nocturnas"),
            "niebla": ("Niebla", "Restringir viajes con baja visibilidad"),
            "otro": ("Otros", "Monitorear evolución"),
        }

        topicos = []
        for tipo, count in sorted(conteo_tipos.items(), key=lambda x: x[1], reverse=True):
            nombre, accion = nombres_tipos.get(tipo, ("Otros", "Monitorear"))
            pct = round(count / total * 100, 1)
            tendencia = "Emergente" if pct > 30 else ("Recurrente" if pct > 15 else "Normal")
            topicos.append({
                "id": len(topicos),
                "nombre": nombre,
                "frecuencia": count,
                "porcentaje": pct,
                "palabras_clave": [tipo],
                "tendencia": tendencia,
                "accion_sugerida": accion
            })

        topico_dominante = topicos[0] if topicos else {"nombre": "N/A", "frecuencia": 0, "porcentaje": 0}
        self.ultima_actualizacion = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        return topicos[:n_topicos], topico_dominante, total

# ============================================================
# INICIALIZACIÓN
# ============================================================

logger.info("Inicializando Motor NLP v2.0...")
corpus_inicial = CORPUS_BASE.copy()
reportes_reales = cargar_reportes_reales()
if reportes_reales:
    ids_existentes = {doc["id"] for doc in corpus_inicial}
    for rep in reportes_reales:
        if rep["id"] not in ids_existentes:
            corpus_inicial.append(rep)
            ids_existentes.add(rep["id"])

motor_nlp = MotorNLP(corpus_inicial)
motor_nlp.inicializar_tfidf()
logger.info(f"Motor NLP listo: {len(corpus_inicial)} reportes ({len(reportes_reales)} reales + {len(CORPUS_BASE)} base)")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "servicio": "SafeRoute Motor NLP",
        "version": "2.0.0",
        "reportes": len(motor_nlp.corpus),
        "tecnicas": ["TF-IDF + Similitud Coseno", "BM25 con expansión semántica"]
    }

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "reportes": len(motor_nlp.corpus)}

@app.post("/nlp/buscar", response_model=BusquedaResponse)
async def buscar_reportes(request: BusquedaRequest):
    resultados, tiempo_ms, consulta_exp = motor_nlp.buscar(
        request.query, request.top_k, metodo=request.metodo
    )
    return BusquedaResponse(
        resultados=resultados,
        total=len(resultados),
        consulta=request.query,
        consulta_expandida=consulta_exp,
        tiempo_ms=round(tiempo_ms, 2)
    )

@app.get("/nlp/comparar")
async def comparar_rankings(q: str = "inundaciones en Suchiapa", k: int = 5):
    res_tfidf, tiempo_tfidf, _ = motor_nlp.buscar(q, k, metodo="tfidf")
    res_bm25, tiempo_bm25, _ = motor_nlp.buscar(q, k, metodo="bm25")
    
    return {
        "consulta": q,
        "tfidf": {
            "tiempo_ms": round(tiempo_tfidf, 2),
            "resultados": [
                {"id": r.id, "score": r.score, "texto": r.texto[:60]}
                for r in res_tfidf
            ]
        },
        "bm25": {
            "tiempo_ms": round(tiempo_bm25, 2),
            "resultados": [
                {"id": r.id, "score": r.score, "texto": r.texto[:60]}
                for r in res_bm25
            ]
        },
        "parametros_bm25": {
            "k1": motor_nlp.bm25.k1 if motor_nlp.bm25 else 1.5,
            "b": motor_nlp.bm25.b if motor_nlp.bm25 else 0.75,
        }
    }

@app.get("/nlp/topicos")
async def obtener_topicos(n_topicos: int = 5):
    topicos, dominante, total = motor_nlp.obtener_topicos(n_topicos)
    return TopicosResponse(
        topicos=topicos,
        total_reportes=total,
        topico_dominante=dominante,
        fecha_analisis=motor_nlp.ultima_actualizacion or datetime.now().isoformat()
    )

@app.post("/nlp/ingest/reporte")
async def ingest_reporte(request: ReporteIngestRequest):
    ok = motor_nlp.upsert_reporte(request.dict())
    if not ok:
        raise HTTPException(status_code=400, detail="reporte invalido")
    return {
        "status": "ok",
        "accion": "upsert",
        "id": request.id,
        "total": len(motor_nlp.corpus),
        "actualizado_en": motor_nlp.ultima_actualizacion,
        "indice": "bm25+tfidf",
    }

@app.post("/nlp/ingest/reporte/estado")
async def ingest_reporte_estado(request: ReporteEstadoRequest):
    encontrado = motor_nlp.actualizar_estado_reporte(request.id, request.vigente)
    return {
        "status": "ok",
        "id": request.id,
        "vigente": request.vigente,
        "encontrado": encontrado,
        "total": len(motor_nlp.corpus),
        "actualizado_en": motor_nlp.ultima_actualizacion,
    }

@app.post("/nlp/recargar")
async def recargar_corpus(body: List[dict] = None):
    global motor_nlp, CORPUS_BASE
    if not body:
        return {"status": "error", "mensaje": "No se recibieron reportes"}
    for rep in body:
        if "texto" in rep and "id" in rep:
            CORPUS_BASE.append(rep)
    motor_nlp = MotorNLP(CORPUS_BASE)
    motor_nlp.inicializar_tfidf()
    return {"status": "ok", "total": len(CORPUS_BASE)}

@app.post("/nlp/clasificar", response_model=ClasificarResponse)
async def clasificar_reporte(request: ClasificarRequest):
    """
    Clasifica automáticamente un reporte usando reglas.
    """
    resultado = clasificar_por_reglas(request.texto)
    
    coincide = None
    if request.tipo_actual:
        coincide = resultado["tipo"] == request.tipo_actual
    
    return ClasificarResponse(
        texto=request.texto[:100],
        tipo_predicho=resultado["tipo"],
        confianza=resultado["confianza"],
        modelo=resultado["modelo"],
        similitudes=None,
        tipo_actual=request.tipo_actual,
        coincide=coincide
    )

@app.post("/nlp/clasificar/lote")
async def clasificar_lote(reportes: List[dict]):
    """Clasifica un lote de reportes."""
    resultados = []
    aciertos = 0
    
    for rep in reportes:
        texto = rep.get("texto", "")
        tipo_real = rep.get("tipo", "")
        
        resultado = clasificar_por_reglas(texto)
        coincide = resultado["tipo"] == tipo_real
        
        if coincide:
            aciertos += 1
        
        resultados.append({
            "id": rep.get("id", ""),
            "texto": texto[:80],
            "tipo_real": tipo_real,
            "tipo_predicho": resultado["tipo"],
            "confianza": resultado["confianza"],
            "coincide": coincide
        })
    
    total = len(reportes)
    precision = round(aciertos / total * 100, 1) if total > 0 else 0
    
    return {
        "resultados": resultados,
        "total": total,
        "aciertos": aciertos,
        "precision": precision,
        "modelo": "reglas"
    }

@app.post("/nlp/ner", response_model=NERResponse)
async def extraer_entidades(request: NERRequest):
    entidades = ner_chiapas.extraer_entidades(request.texto)
    tipo = ner_chiapas.extraer_tipo_incidente(request.texto)
    severidad = ner_chiapas.extraer_severidad(request.texto)
    resumen = ner_chiapas.generar_resumen_entidades(request.texto)
    
    return NERResponse(
        texto=request.texto[:150],
        tipo_incidente=tipo,
        severidad=severidad,
        entidades=entidades,
        resumen=resumen
    )

@app.post("/nlp/analizar")
async def analizar_reporte(request: NERRequest):
    """
    Análisis completo de un reporte: NER + clasificación + severidad.
    """
    entidades = ner_chiapas.extraer_entidades(request.texto)
    tipo_ner = ner_chiapas.extraer_tipo_incidente(request.texto)
    severidad = ner_chiapas.extraer_severidad(request.texto)
    
    clasificacion = clasificar_por_reglas(request.texto)
    
    similares = []
    if motor_nlp.vectorizer is not None:
        try:
            tokens = motor_nlp._preprocesar_texto(request.texto)
            consulta = " ".join(tokens)
            resultados, _, _ = motor_nlp.buscar(consulta, top_k=3, metodo="bm25")
            similares = [{"id": r.id, "texto": r.texto[:80], "score": r.score} for r in resultados]
        except:
            pass
    
    return {
        "texto": request.texto[:150],
        "clasificacion": {
            "tipo_ner": tipo_ner,
            "tipo_reglas": clasificacion["tipo"],
            "coinciden": tipo_ner == clasificacion["tipo"],
            "severidad": severidad,
        },
        "entidades": entidades,
        "resumen_estructurado": ner_chiapas.generar_resumen_entidades(request.texto),
        "reportes_similares": similares,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")