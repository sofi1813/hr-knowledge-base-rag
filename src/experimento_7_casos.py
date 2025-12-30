import chromadb
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate
import random

# --- CONFIGURACIÓN ---
CHROMA_DB_PATH = "./candidates_db"
COLLECTION_NAME = "cvu_candidatos"

# --- AJUSTES DE EXPERIMENTO ---
N_MUESTRA = 20
OBJETIVO_TITULO = "Software Engineer Developer"
OBJETIVO_SKILLS = "Python, SQL, Leadership"
OBJETIVO_EXP_MIN = 3
UMBRAL = 0.45

def cargar_contexto():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return col, model
    except Exception as e:
        print(f"❌ Error conectando a DB: {e}. ¿Ejecutaste el script de ingesta primero?")
        exit()

def evaluar_similitud(texto_candidato, texto_objetivo, model):
    """Devuelve 1 si hay match semántico, 0 si no."""
    if not texto_candidato or texto_candidato.strip() == "":
        return 0

    emb1 = model.encode(texto_objetivo, convert_to_tensor=True)
    emb2 = model.encode(texto_candidato, convert_to_tensor=True)

    similitud = util.cos_sim(emb1, emb2).item()
    return 1 if similitud >= UMBRAL else 0

def ejecutar_motor_inferencia():
    collection, model = cargar_contexto()

    # 1. RECUPERAR TODOS LOS IDs DISPONIBLES
    todos_los_ids = collection.get(include=[])['ids']
    total_documentos = len(todos_los_ids)

    if total_documentos == 0:
        print("📭 La base de datos está vacía. Carga documentos con 'ingesta_cvu.py'.")
        return

    # 2. SELECCIONAR LA MUESTRA ALEATORIA
    if N_MUESTRA >= total_documentos:
        ids_muestra = todos_los_ids
        print(f"⚠️ Muestra ({N_MUESTRA}) es mayor al total ({total_documentos}). Usando todos los documentos.")
    else:
        ids_muestra = random.sample(todos_los_ids, N_MUESTRA)
        print(f"✅ Se seleccionó una muestra aleatoria de {len(ids_muestra)} documentos (N={N_MUESTRA}).")

    # 3. RECUPERAR SOLO LOS METADATOS DE LA MUESTRA
    datos = collection.get(ids=ids_muestra, include=['metadatas'])

    print("="*80)
    print(f"🎯 OBJETIVOS DE LA PRUEBA:")
    print(f"   1. Título cercano a: '{OBJETIVO_TITULO}'")
    print(f"   2. Skills cercanas a: '{OBJETIVO_SKILLS}'")
    print(f"   3. Experiencia mínima: {OBJETIVO_EXP_MIN} años")
    print("="*80)

    # 4. PRE-CALCULO DE FACTORES (T, S, E)
    evaluaciones_base = []
    ids_muestras = datos['ids']
    metas = datos['metadatas']
    for i, doc_id in enumerate(ids_muestras):
        m = metas[i]
        val_t = evaluar_similitud(m.get('titles', ''), OBJETIVO_TITULO, model)
        val_s = evaluar_similitud(m.get('skills', ''), OBJETIVO_SKILLS, model)
        exp_real = m.get('years_experience', 0)
        val_e = 1 if exp_real >= OBJETIVO_EXP_MIN else 0

        # Truncar el nombre del candidato a 20 caracteres para alineación
        candidato = m.get('candidate_name', 'Unknown')[:20]

        evaluaciones_base.append({
            "ID_Archivo": doc_id,
            "Candidato": candidato,
            "T": val_t,
            "S": val_s,
            "E": val_e,
            "Debug": f"Exp:{exp_real}"
        })

    # 5. EJECUCIÓN DE LOS 7 CASOS DE USO
    casos_uso = [
        {"id": 1, "desc": "Solo Título (T)", "logica": lambda x: x['T']},
        {"id": 2, "desc": "Solo Skills (S)", "logica": lambda x: x['S']},
        {"id": 3, "desc": "Solo Experiencia (E)", "logica": lambda x: x['E']},
        {"id": 4, "desc": "Título AND Skills (T & S)", "logica": lambda x: x['T'] and x['S']},
        {"id": 5, "desc": "Título AND Exp (T & E)", "logica": lambda x: x['T'] and x['E']},
        {"id": 6, "desc": "Skills AND Exp (S & E)", "logica": lambda x: x['S'] and x['E']},
        {"id": 7, "desc": "TODO (T & S & E)", "logica": lambda x: x['T'] and x['S'] and x['E']},
    ]

    for caso in casos_uso:
        print(f"\n📊 CASO {caso['id']}: {caso['desc']}")
        rows = []
        for ev in evaluaciones_base:
            rows.append([
                ev["ID_Archivo"],
                ev["Candidato"],
                caso['logica'](ev)
            ])

        # Imprimir tabla con formato 'grid' y alineación a la izquierda
        print(tabulate(
            rows,
            headers=["ID Archivo", "Candidato", "Cumple (0/1)"],
            tablefmt='grid',
            stralign='left',
            numalign='left'
        ))

if __name__ == "__main__":
    ejecutar_motor_inferencia()
