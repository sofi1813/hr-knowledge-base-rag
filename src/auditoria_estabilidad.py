import chromadb
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate
import random
import time

# --- CONFIGURACIÓN ---
CHROMA_DB_PATH = "./candidates_db"
COLLECTION_NAME = "cvu_candidatos"

# PARÁMETROS DEL EXPERIMENTO
SEMILLA = 42          # Cambia esto para obtener un grupo diferente de CVs
TAMAÑO_MUESTRA = 20   # Cantidad de CVs a auditar
UMBRAL = 0.45

# OBJETIVOS DE LA PRUEBA
TARGET_TITULO = "Software Engineer Developer"
TARGET_SKILLS = "Python, SQL, Leadership"
TARGET_EXP = 3

# ¿Qué caso quieres auditar? (1=Titulo, 7=Todo, etc.)
CASO_A_EVALUAR = 1 

def conectar_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return col, model

def ver_muestra_seleccionada(ids_muestra, metas):
    """
    Función independiente para visualizar qué documentos entraron en el sorteo.
    """
    lista_datos = []
    print(f"\n📄 DETALLE DE LA MUESTRA SELECCIONADA (Semilla {SEMILLA}):")
    
    for i, doc_id in enumerate(ids_muestra):
        m = metas[doc_id]
        nombre = m.get('candidate_name', 'Unknown')[:25] # Cortar si es muy largo
        titulo = m.get('titles', '')[:30]
        exp = m.get('years_experience', 0)
        
        lista_datos.append([
            i + 1,          # Índice
            doc_id,         # ID del archivo
            nombre,         # Nombre candidato
            titulo,         # Título detectado
            f"{exp} años"   # Experiencia
        ])
    
    # Imprimimos tabla informativa
    print(tabulate(lista_datos, 
                   headers=["#", "ID Archivo", "Candidato", "Título Detectado", "Exp"], 
                   tablefmt="simple"))
    print("-" * 80)

def evaluar_candidato(meta, model):
    """
    Lógica de evaluación que responde a CASO_A_EVALUAR.
    """
    # 1. Similitud Título
    emb_t1 = model.encode(TARGET_TITULO, convert_to_tensor=True)
    emb_t2 = model.encode(meta.get('titles', ''), convert_to_tensor=True)
    val_t = 1 if util.cos_sim(emb_t1, emb_t2).item() >= UMBRAL else 0
    
    # 2. Similitud Skills
    emb_s1 = model.encode(TARGET_SKILLS, convert_to_tensor=True)
    emb_s2 = model.encode(meta.get('skills', ''), convert_to_tensor=True)
    val_s = 1 if util.cos_sim(emb_s1, emb_s2).item() >= UMBRAL else 0
    
    # 3. Experiencia
    val_e = 1 if meta.get('years_experience', 0) >= TARGET_EXP else 0

    # LÓGICA DE SELECCIÓN DE CASO
    if CASO_A_EVALUAR == 1: return val_t
    elif CASO_A_EVALUAR == 2: return val_s
    elif CASO_A_EVALUAR == 3: return val_e
    elif CASO_A_EVALUAR == 4: return 1 if (val_t and val_s) else 0
    elif CASO_A_EVALUAR == 5: return 1 if (val_t and val_e) else 0
    elif CASO_A_EVALUAR == 6: return 1 if (val_s and val_e) else 0
    elif CASO_A_EVALUAR == 7: return 1 if (val_t and val_s and val_e) else 0
    else: return 0

def ejecutar_auditoria():
    print(f"🔬 INICIANDO AUDITORÍA DE ESTABILIDAD")
    col, model = conectar_db()
    
    # 1. OBTENER MUESTRA
    todos_ids = col.get(include=[])['ids']
    
    if len(todos_ids) < TAMAÑO_MUESTRA:
        ids_muestra = todos_ids
    else:
        random.seed(SEMILLA) # Fijamos la aleatoriedad
        ids_muestra = random.sample(todos_ids, TAMAÑO_MUESTRA)

    # Recuperamos metadatos
    datos = col.get(ids=ids_muestra, include=['metadatas'])
    metas = {id_: meta for id_, meta in zip(datos['ids'], datos['metadatas'])}

    # --- NUEVA FUNCIÓN: MOSTRAR QUIÉNES SON ---
    ver_muestra_seleccionada(ids_muestra, metas)

    # --- FASE 1: RUN 1 ---
    print("\n🏃 Ejecutando Evaluación #1 ...")
    resultados_run_1 = {}
    for doc_id in ids_muestra:
        resultados_run_1[doc_id] = evaluar_candidato(metas[doc_id], model)

    print("⏳ Simulando espera...")
    time.sleep(0.5) 

    # --- FASE 2: RUN 2 ---
    print("🏃 Ejecutando Evaluación #2 (Re-test) ...")
    resultados_run_2 = {}
    for doc_id in ids_muestra:
        resultados_run_2[doc_id] = evaluar_candidato(metas[doc_id], model)

    # --- FASE 3: MATRIZ DE CONFUSIÓN (ESTABILIDAD) ---
    tp = 0 # Si - Si
    tn = 0 # No - No
    fp = 0 # No - Si (Inconsistencia)
    fn = 0 # Si - No (Inconsistencia)

    detalles_error = []

    for doc_id in ids_muestra:
        val1 = resultados_run_1[doc_id]
        val2 = resultados_run_2[doc_id]
        nombre = metas[doc_id].get('candidate_name', 'Unknown')

        if val1 == 1 and val2 == 1: tp += 1
        elif val1 == 0 and val2 == 0: tn += 1
        elif val1 == 0 and val2 == 1:
            fp += 1
            detalles_error.append([nombre, "Cambió a APROBADO"])
        elif val1 == 1 and val2 == 0:
            fn += 1
            detalles_error.append([nombre, "Cambió a RECHAZADO"])

    print("\n" + "="*50)
    print(f"📊 MATRIZ DE CONSISTENCIA (CASO {CASO_A_EVALUAR})")
    print("="*50)
    
    matriz = [
        ["", "Run 2: SÍ", "Run 2: NO"],
        ["Run 1: SÍ", f"✅ {tp}", f"❌ {fn}"],
        ["Run 1: NO", f"⚠️ {fp}", f"✅ {tn}"]
    ]
    print(tabulate(matriz, tablefmt="grid"))
    
    estabilidad = ((tp + tn) / len(ids_muestra)) * 100
    print(f"\n🎯 ESTABILIDAD DEL SISTEMA: {estabilidad:.2f}%")

    if detalles_error:
        print("\n🔍 INCONSISTENCIAS ENCONTRADAS:")
        print(tabulate(detalles_error, headers=["Candidato", "Tipo de Error"]))

if __name__ == "__main__":
    ejecutar_auditoria()
