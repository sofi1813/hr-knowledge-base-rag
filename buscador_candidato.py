import chromadb
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURACIÓN ---
CHROMA_DB_PATH = "./candidates_db"
COLLECTION_NAME = "cvu_candidatos"
UMBRAL_SEMANTICO = 0.4

def conectar_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return col, model
    except Exception as e:
        print(f"❌ Error: {e}. Ejecuta primero 'ingesta_cvu.py'")
        exit()

def calcular_similitud(texto_base, texto_objetivo, model):
    if not texto_base:
        return 0.0
    emb1 = model.encode(texto_objetivo, convert_to_tensor=True)
    emb2 = model.encode(texto_base, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def buscar_candidatos():
    collection, model = conectar_db()
    datos = collection.get()
    ids = datos['ids']
    metas = datos['metadatas']

    if not ids:
        print("📭 Base de datos vacía.")
        return

    while True:
        print("\n" + "═"*60)
        print(" 🔍  BUSCADOR DE MEJORES CANDIDATOS (RANKING AI)")
        print("═"*60)

        target_titulo = input("   🎯 Título deseado (ej. Software Engineer): ")
        target_skills = input("   🎯 Skills deseados (ej. Python, SQL): ")
        target_exp_str = input("   🎯 Años experiencia mínima (ej. 3): ")
        try:
            target_exp = int(target_exp_str)
        except:
            target_exp = 0

        print("\n--- SELECCIONE EL CASO DE USO (ESTRATEGIA) ---")
        print("1. Solo Título")
        print("2. Solo Skills")
        print("3. Solo Experiencia")
        print("4. Título + Skills")
        print("5. Título + Experiencia")
        print("6. Skills + Experiencia")
        print("7. TODO (Título + Skills + Experiencia)")
        print("0. Salir")

        opcion = input("\n👉 Seleccione opción (0-7): ")
        if opcion == '0':
            break

        resultados = []
        print("\n🔄 Analizando y Rankeando candidatos...")

        for i, doc_id in enumerate(ids):
            m = metas[i]
            nombre = m.get('candidate_name', 'Unknown')[:25]
            cv_titulo = m.get('titles', '')
            cv_skills = m.get('skills', '')
            cv_exp = m.get('years_experience', 0)

            score_t = calcular_similitud(cv_titulo, target_titulo, model)
            score_s = calcular_similitud(cv_skills, target_skills, model)
            score_e = 1.0 if cv_exp >= target_exp else 0.0

            final_score = 0.0
            detalles_calculo = ""

            if opcion == '1':
                final_score = score_t
                detalles_calculo = f"T({score_t:.2f})"
            elif opcion == '2':
                final_score = score_s
                detalles_calculo = f"S({score_s:.2f})"
            elif opcion == '3':
                final_score = score_e
                detalles_calculo = f"E({score_e})"
            elif opcion == '4':
                final_score = (score_t + score_s) / 2
                detalles_calculo = f"Avg(T:{score_t:.2f}, S:{score_s:.2f})"
            elif opcion == '5':
                final_score = (score_t + score_e) / 2
                detalles_calculo = f"Avg(T:{score_t:.2f}, E:{score_e})"
            elif opcion == '6':
                final_score = (score_s + score_e) / 2
                detalles_calculo = f"Avg(S:{score_s:.2f}, E:{score_e})"
            elif opcion == '7':
                final_score = (score_t + score_s + score_e) / 3
                detalles_calculo = f"Avg(T:{score_t:.2f}, S:{score_s:.2f}, E:{score_e})"

            resultados.append({
                "Ranking": 0,
                "Candidato": nombre,
                "Match %": final_score * 100,
                "Detalle Score": detalles_calculo,
                "Info": f"Rol: {cv_titulo[:15]}... | Exp: {cv_exp} | Skills: {cv_skills[:20]}..."
            })

        resultados.sort(key=lambda x: x["Match %"], reverse=True)
        top_candidatos = resultados[:10]

        print(f"\n🏆 TOP {len(top_candidatos)} CANDIDATOS - CASO {opcion}:")
        for idx, candidato in enumerate(top_candidatos, start=1):
            print(f"\n--- Candidato #{idx} ---")
            print(f"Nombre: {candidato['Candidato']}")
            print(f"Match: {candidato['Match %']:.1f}% ({candidato['Detalle Score']})")
            print(f"Info: {candidato['Info']}")

        print(f"\n💡 Nota: Se evaluaron {len(ids)} documentos en total.")
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    buscar_candidatos()
