import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import chromadb
from sentence_transformers import SentenceTransformer
import spacy

# --- CONFIGURACIÓN ---
CHROMA_DB_PATH = "./candidates_db"
COLLECTION_NAME = "cvu_candidatos"
DIRECTORIO_PDFS = "./carpeta_cvus_test"

# Configuración OCR Windows
path_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.name == 'nt' and os.path.exists(path_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_tesseract

# CARGAMOS MODELOS NLP (Español e Inglés)
print("⏳ Cargando modelos de NLP...")
try:
    nlp_es = spacy.load("es_core_news_md")
    nlp_en = spacy.load("en_core_web_md")
except:
    print("❌ Error: Debes instalar los modelos de spacy.")
    print("Ejecuta: python -m spacy download es_core_news_md")
    print("Ejecuta: python -m spacy download en_core_web_md")
    exit()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class ExtractorPro:
    def __init__(self):
        self.titulos_conocidos = [
            "ingeniero", "engineer", "developer", "desarrollador", "programador",
            "architect", "arquitecto", "manager", "gerente", "analyst", "analista",
            "scientist", "científico", "administrator", "administrador", "technician",
            "técnico", "consultant", "consultor", "director", "coordinator", "coordinador",
            "specialist", "especialista", "designer", "diseñador", "bachelor", "licenciado",
            "master", "maestría", "phd", "doctorado"
        ]

        self.skills_conocidas = [
            "python", "java", "javascript", "typescript", "sql", "nosql", "aws", "azure", 
            "docker", "kubernetes", "react", "angular", "vue", "node", "django", "flask",
            "git", "linux", "excel", "power bi", "tableau", "salesforce", "sap",
            "leadership", "liderazgo", "communication", "comunicación", "english", "inglés",
            "agile", "scrum", "kanban", "marketing", "sales", "ventas"
        ]

    def _ocr_hibrido(self, pagina):
        # Usamos "blocks" para respetar columnas, pero get_text("blocks") devuelve tuplas
        # Para OCR necesitamos imagen. Esta función decide qué método usar.
        texto = pagina.get_text().strip()
        if len(texto) > 50: 
            return "DIGITAL" # Marcador para indicar que usemos extracción digital
        
        # OCR Fallback
        pix = pagina.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            return pytesseract.image_to_string(img, lang='spa+eng')
        except:
            return ""

    def extraer_texto_ordenado(self, ruta_pdf):
        """
        Usa lógica de BLOQUES para leer columnas correctamente.
        """
        texto_completo = ""
        with fitz.open(ruta_pdf) as doc:
            for pagina in doc:
                res_ocr = self._ocr_hibrido(pagina)
                
                if res_ocr == "DIGITAL":
                    # Extraer bloques de texto ordenados por posición (arriba->abajo, izq->der)
                    # Esto evita mezclar columnas.
                    bloques = pagina.get_text("blocks", sort=True)
                    for b in bloques:
                        # b[4] contiene el texto del bloque
                        texto_completo += b[4] + "\n"
                else:
                    # Es OCR, añadimos el resultado directo
                    texto_completo += res_ocr + "\n"
        return texto_completo

    def extraer_nombre_con_nlp(self, texto):
        """
        Usa Inteligencia Artificial (spaCy) para encontrar personas.
        """
        # Tomamos solo el inicio del texto para buscar el nombre (primeros 500 caracteres)
        inicio_texto = texto[:800]
        
        # Limpieza básica
        inicio_texto = inicio_texto.replace('\n', ' ').strip()
        
        # Procesamos con ambos modelos (ES y EN) por si el CV está en inglés
        docs = [nlp_es(inicio_texto), nlp_en(inicio_texto)]
        
        candidatos_nombre = []

        for doc in docs:
            for ent in doc.ents:
                # Buscamos entidades etiquetadas como PER (Persona)
                if ent.label_ == "PER" or ent.label_ == "PERSON":
                    nombre = ent.text.strip().title()
                    # Filtros extra de seguridad
                    if "Curriculum" in nombre or "Resume" in nombre or "Cv" in nombre: continue
                    if len(nombre.split()) < 2: continue # Un nombre suele tener Nombre+Apellido
                    if len(nombre) > 40: continue
                    if any(char.isdigit() for char in nombre): continue
                    
                    candidatos_nombre.append(nombre)

        if candidatos_nombre:
            # Retornamos el primero encontrado (suelen aparecer arriba)
            return candidatos_nombre[0]
            
        return "Unknown Candidate"

    def extraer_experiencia_regex(self, texto):
        # Regex más estricto: Busca números de 1 o 2 dígitos seguidos explícitamente de "años" o "years"
        # Ignora fechas como "2015-2018" (que dan 2000 años de experiencia si no se cuida)
        texto_lower = texto.lower()
        patron = r'(\d{1,2})\+?\s*(?:años|years|yrs|ans)\s+(?:de\s+)?(?:experiencia|experience)?'
        
        coincidencias = re.findall(patron, texto_lower)
        numeros = [int(x) for x in coincidencias if int(x) < 50] # Filtramos errores (nadie tiene 99 años exp)
        
        return max(numeros) if numeros else 0

    def procesar_cv(self, ruta_archivo):
        texto = self.extraer_texto_ordenado(ruta_archivo)
        
        # 1. Extracción de Nombre con IA
        nombre = self.extraer_nombre_con_nlp(texto)
        
        # 2. Extracción de Experiencia mejorada
        anios = self.extraer_experiencia_regex(texto)
        
        # 3. Skills y Títulos (Búsqueda difusa simple)
        texto_lower = texto.lower()
        skills = list(set([s for s in self.skills_conocidas if s in texto_lower]))
        titulos = list(set([t for t in self.titulos_conocidos if t in texto_lower]))
        
        return texto, {
            "candidate_name": nombre,
            "years_experience": anios,
            "skills": ", ".join(skills),
            "titles": ", ".join(titulos)
        }

def vaciar_base_datos():
    try: chroma_client.delete_collection(COLLECTION_NAME)
    except: pass
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def main():
    collection = vaciar_base_datos()
    extractor = ExtractorPro()
    
    if not os.path.exists(DIRECTORIO_PDFS):
        os.makedirs(DIRECTORIO_PDFS)
        return

    archivos = [f for f in os.listdir(DIRECTORIO_PDFS) if f.lower().endswith(".pdf")]
    print(f"--- PROCESANDO {len(archivos)} ARCHIVOS CON NLP AVANZADO ---")

    for archivo in archivos:
        try:
            ruta = os.path.join(DIRECTORIO_PDFS, archivo)
            
            # Procesamiento centralizado
            texto_full, meta = extractor.procesar_cv(ruta)
            
            meta["filename"] = archivo
            vector = embedding_model.encode(texto_full).tolist()

            collection.upsert(
                ids=[archivo],
                embeddings=[vector],
                documents=[texto_full],
                metadatas=[meta]
            )
            
            print(f"✅ {meta['candidate_name']:<30} | Exp: {meta['years_experience']} | Skills: {len(meta['skills'].split(','))}")

        except Exception as e:
            print(f"❌ Error en {archivo}: {e}")

if __name__ == "__main__":
    main()
