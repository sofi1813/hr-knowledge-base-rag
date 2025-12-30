import chromadb
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate
import random
import csv
import os

# --- 1. CONFIGURACIÓN GENERAL ---
CHROMA_DB_PATH = "./candidates_db"
COLLECTION_NAME = "cvu_candidatos"
ARCHIVO_VERDAD = "verdad_terreno.csv" # Tu archivo manual de etiquetas

# --- 2. PARÁMETROS DE EXPERIMENTACIÓN ---
SEMILLA = 42          # Fija esta semilla para tomar SIEMPRE los mismos CVs
TAMAÑO_MUESTRA = 20   # Cantidad de CVs a evaluar
UMBRAL = 0.30         # Umbral de Similitud Coseno (de 0.0 a 1.0)

# --- 3. REQUISITOS FIJOS DEL PUESTO ---
TARGET_TITULO = "Software Engineer Developer"
TARGET_SKILLS = "Python, SQL, Leadership"
TARGET_EXP = 3

# --- 4. CASO DE USO A EVALUAR ---
# Define qué criterio de búsqueda auditarás
# 1=Título, 2=Skills, 3=Experiencia, 4=T&S, 7=T&S&E (TODO)
CASO_A_EVALUAR = 7 


def conectar_db():
    """Conecta a la base de datos Chroma y carga el modelo de embeddings."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        col = client.get_collection(name=COLLECTION_NAME)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return col, model
    except Exception as e:
        print(f"❌ Error conectando a DB: {e}. ¿Ejecutaste 'ingesta_cvu.py'?")
        exit()

def obtener_muestra_controlada(col):
    """Obtiene un conjunto fijo de IDs usando la semilla."""
    todos_ids = col.get(include=[])['ids']
    total = len(todos_ids)
    
    if total < TAMAÑO_MUESTRA:
        ids_muestra = todos_ids
        print(f"⚠️ Muestra ({TAMAÑO_MUESTRA}) mayor que el total ({total}). Usando todos.")
    else:
        random.seed(SEMILLA)
        ids_muestra = random.sample(todos_ids, TAMAÑO_MUESTRA)
        print(f"✅ Muestra de {len(ids_muestra)} CVs seleccionada (Semilla {SEMILLA}).")

    datos = col.get(ids=ids_muestra, include=['metadatas'])
    metadatas = {id_: meta for id_, meta in zip(datos['ids'], datos['metadatas'])}
    return ids_muestra, metadatas

def evaluar_candidato(meta, model):
    """
    Lógica de evaluación blindada con Debugging.
    """
    nombre = meta.get('candidate_name', 'Unknown')
    
    # --- 1. EVALUACIÓN TÍTULO ---
    emb_t1 = model.encode(TARGET_TITULO, convert_to_tensor=True)
    emb_t2 = model.encode(meta.get('titles', ''), convert_to_tensor=True)
    score_t = util.cos_sim(emb_t1, emb_t2).item()
    val_t = 1 if score_t >= UMBRAL else 0
    
    # --- 2. EVALUACIÓN SKILLS ---
    emb_s1 = model.encode(TARGET_SKILLS, convert_to_tensor=True)
    emb_s2 = model.encode(meta.get('skills', ''), convert_to_tensor=True)
    score_s = util.cos_sim(emb_s1, emb_s2).item()
    val_s = 1 if score_s >= UMBRAL else 0
    
    # --- 3. EVALUACIÓN EXPERIENCIA (Blindaje de Tipos) ---
    raw_exp = meta.get('years_experience', 0)
    try:
        # Convertimos a float primero por si viene como "3.5" y luego a int
        exp_num = int(float(raw_exp))
    except:
        exp_num = 0
        
    val_e = 1 if exp_num >= TARGET_EXP else 0

    # --- DEBUGGING (Para que veas por qué da 0) ---
    # Esto imprimirá en consola los valores internos
    # print(f"🔍 DEBUG {nombre[:15]}: T({score_t:.2f})={val_t} | S({score_s:.2f})={val_s} | Exp({exp_num})={val_e} --> CASO {CASO_A_EVALUAR}")

    # --- LÓGICA DE SELECCIÓN DE CASO ---
    resultado_final = 0
    
    if CASO_A_EVALUAR == 1: # Solo Título
        resultado_final = val_t
    elif CASO_A_EVALUAR == 2: # Solo Skills
        resultado_final = val_s
    elif CASO_A_EVALUAR == 3: # Solo Experiencia
        resultado_final = val_e
    elif CASO_A_EVALUAR == 4: # Título AND Skills
        resultado_final = 1 if (val_t and val_s) else 0
    elif CASO_A_EVALUAR == 5: # Título AND Exp
        resultado_final = 1 if (val_t and val_e) else 0
    elif CASO_A_EVALUAR == 6: # Skills AND Exp
        resultado_final = 1 if (val_s and val_e) else 0
    elif CASO_A_EVALUAR == 7: # TODO
        resultado_final = 1 if (val_t and val_s and val_e) else 0
        
    return resultado_final

def cargar_verdad_terreno(ids_muestra):
    """Carga tu evaluación manual desde el CSV y valida los IDs."""
    verdad = {}
    ids_en_csv = set()
    try:
        with open(ARCHIVO_VERDAD, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader) # Leer cabecera

            # Asegurar que el CSV tenga las dos columnas esperadas
            if len(headers) < 2 or 'ID_Archivo' not in headers[0] or 'Etiqueta_Humana' not in headers[1]:
                print(f"❌ Error en cabecera. Asegúrate de que las columnas son: ID_Archivo, Etiqueta_Humana")
                exit()
                
            for row in reader:
                if len(row) >= 2:
                    doc_id = row[0].strip()
                    try:
                        etiqueta = int(row[1].strip())
                        if doc_id in ids_muestra:
                            verdad[doc_id] = etiqueta
                            ids_en_csv.add(doc_id)
                    except ValueError:
                        print(f"⚠️ Ignorando fila de {doc_id}: La etiqueta humana debe ser 1 o 0.")
        
        if len(ids_en_csv) != len(ids_muestra):
            print(f"⚠️ Advertencia: El CSV solo tiene {len(ids_en_csv)} etiquetas, se esperaban {len(ids_muestra)}.")
            
        return verdad
    except FileNotFoundError:
        return None

def generar_reporte_muestra(ids_muestra, metadatas):
    """Genera el archivo CSV inicial para que el humano etiquete."""
    print("----------------------------------------------------------")
    print(f" PASO 1: GENERAR ARCHIVO BASE PARA ETIQUETADO MANUAL")
    print("----------------------------------------------------------")
    
    lista_datos = []
    for doc_id in ids_muestra:
        m = metadatas[doc_id]
        nombre = m.get('candidate_name', 'Unknown')
        lista_datos.append([
            doc_id, 
            0, # Columna Etiqueta Humana: DEBES rellenar con 1 (Cumple) o 0 (No Cumple)
            nombre,
            m.get('titles', '')[:40],
            m.get('skills', '')[:40],
            m.get('years_experience', 0)
        ])
    
    # Crear el CSV de salida
    headers = ["ID_Archivo", "Etiqueta_Humana", "Candidato", "Titulo_Detectado", "Skills_Detectadas", "Exp_Detectada"]
    try:
        with open(ARCHIVO_VERDAD, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(lista_datos)
        
        print(f"💾 Archivo '{ARCHIVO_VERDAD}' creado. Contiene {len(ids_muestra)} CVs.")
        print(f"👀 TAREA: Abre el CSV, evalúa la columna 'Etiqueta_Humana' (1 o 0) y vuelve a ejecutar el script.")
        print("----------------------------------------------------------")
        return False
        
    except Exception as e:
        print(f"❌ Error al escribir el CSV: {e}")
        return False

def ejecutar_auditoria_precision():
    col, model = conectar_db()
    
    # Obtener IDs y metadatos de la muestra fija
    ids_muestra, metadatas = obtener_muestra_controlada(col)
    
    # Intentar cargar la verdad terreno (etiquetas humanas)
    verdad_humana = cargar_verdad_terreno(ids_muestra)

    if verdad_humana is None or not verdad_humana:
        # Si el CSV no existe o está vacío, genera el archivo base y termina
        generar_reporte_muestra(ids_muestra, metadatas)
        return
    
    # --- PROCESO DE AUDITORÍA (Solo si hay etiquetas humanas) ---
    print("\n----------------------------------------------------------")
    print(f" PASO 2: COMPARANDO MÁQUINA vs HUMANO (Caso {CASO_A_EVALUAR})")
    print("----------------------------------------------------------")

    # Contadores Matriz Confusión
    tp = 0 # Humano: 1, Máquina: 1 (Verdadero Positivo - Acierto)
    tn = 0 # Humano: 0, Máquina: 0 (Verdadero Negativo - Acierto)
    fp = 0 # Humano: 0, Máquina: 1 (Falso Positivo - Error: Máquina es muy optimista)
    fn = 0 # Humano: 1, Máquina: 0 (Falso Negativo - Error: Máquina es muy estricta/fallo de extracción)

    detalles_error = []

    for doc_id in verdad_humana.keys():
        decision_maquina = evaluar_candidato(metadatas[doc_id], model)
        decision_humana = verdad_humana[doc_id]
        
        nombre = metadatas[doc_id].get('candidate_name', 'Unknown')

        if decision_humana == 1 and decision_maquina == 1: tp += 1
        elif decision_humana == 0 and decision_maquina == 0: tn += 1
        elif decision_humana == 0 and decision_maquina == 1:
            fp += 1
            detalles_error.append([nombre, "H:NO, M:SÍ", "Falso Positivo (Máquina optimista)"])
        elif decision_humana == 1 and decision_maquina == 0:
            fn += 1
            detalles_error.append([nombre, "H:SÍ, M:NO", "Falso Negativo (Máquina estricta)"])

    # --- REPORTE DE MATRIZ DE CONFUSIÓN ---
    print("\n" + "="*60)
    print("📊 MATRIZ DE CONFUSIÓN (PRECISIÓN AI vs. Criterio Humano)")
    print("="*60)
    
    matriz = [
        ["", "Máquina: SÍ", "Máquina: NO"],
        ["Humano: SÍ", f"✅ {tp} (VP)", f"❌ {fn} (FN)"],
        ["Humano: NO", f"⚠️ {fp} (FP)", f"✅ {tn} (VN)"]
    ]
    print(tabulate(matriz, tablefmt="grid"))
    
    total_evaluado = tp + tn + fp + fn
    if total_evaluado > 0:
        accuracy = (tp + tn) / total_evaluado * 100
        print(f"\n🎯 EXACTITUD (ACCURACY) GLOBAL: {accuracy:.2f}%")
        
        # Métricas adicionales (Opcional, pero útil)
        precision_calc = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_calc = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"   - Precisión (Solo aciertos cuando dijo SÍ): {precision_calc:.2f}")
        print(f"   - Recall (Capacidad para encontrar SÍes): {recall_calc:.2f}")
    
    if detalles_error:
        print("\n🔍 DETALLE DE ERRORES:")
        print(tabulate(detalles_error, headers=["Candidato", "Cruce", "Tipo Error"], tablefmt="simple"))

if __name__ == "__main__":
    ejecutar_auditoria_precision()