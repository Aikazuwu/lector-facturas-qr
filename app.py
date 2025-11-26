import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
import numpy as np
import cv2
import base64
import re
import json
import pandas as pd
from urllib.parse import urlparse, parse_qs
import pytesseract
import unicodedata
import tempfile
import zipfile
import io

# --------- Configuración Tesseract ---------
# Si lo corres en local y no detecta tesseract, descomenta la siguiente línea:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------- Constantes ---------
ZOOM = 4.0
ROI_CENTRAL = (0.1, 0.3, 0.8, 0.4)
TIPO_MAP = {'1':'1','4':'2','6':'3','9':'4','11':'5','15':'6','51':'7','54':'8'}
REGIONES_QR = [
    (0.00, 0.00, 0.25, 0.25), (0.00, 0.00, 0.50, 0.50),
    (0.00, 0.75, 0.25, 0.25), (0.00, 0.75, 0.50, 0.25),
    (0.00, 0.75, 1.00, 0.25), (0.00, 0.50, 0.50, 0.50)
]
MES_MAP = {'enero':'01','febrero':'02','marzo':'03','abril':'04','mayo':'05','junio':'06',
           'julio':'07','agosto':'08','septiembre':'09','setiembre':'09','octubre':'10',
           'noviembre':'11','diciembre':'12'}
ABBR_MAP = {'ene':'01','feb':'02','mar':'03','abr':'04','may':'05','jun':'06','jul':'07',
            'ago':'08','sep':'09','sept':'09','set':'09','oct':'10','nov':'11','dic':'12'}

# --------- Funciones Auxiliares ---------

def _normalize_digits(s):
    return re.sub(r"\D", "", str(s)) if s else ""

def _dni_from_cuil(cuil):
    digs = _normalize_digits(cuil)
    return digs[2:10] if len(digs) == 11 else ""

def _normalize_str(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

def extraer_texto_pdf_y_roi(ruta_pdf):
    texto = ""
    try:
        doc = fitz.open(ruta_pdf)
        for p in doc: texto += p.get_text()
        doc.close()
    except: pass
    
    try:
        doc = fitz.open(ruta_pdf)
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            h, w = img.shape[:2]
            xf, yf, wf, hf = ROI_CENTRAL
            roi = img[int(h*yf):int(h*(yf+hf)), int(w*xf):int(w*(xf+wf))]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            texto += pytesseract.image_to_string(gray, lang='spa') + "\n"
        doc.close()
    except: pass
    return texto

# --- Lógica de DNI y Periodo ---
KEYWORDS_DNI = r"(dni|documento|doc\.?|nro\.?\s*doc|número\s*de\s*documento)"
REGEX_DNI_KEY = re.compile(rf"{KEYWORDS_DNI}\D{{0,30}}(\d{{1,2}}[.,]\d{{3}}[.,]\d{{3}}|\d{{8}})", re.IGNORECASE)
REGEX_DNI_PUNTOS = re.compile(r"\b(\d{1,2}\.\d{3}\.\d{3})\b")
REGEX_DNI_COMAS  = re.compile(r"\b(\d{1,2},\d{3},\d{3})\b")
REGEX_DNI_CORRIDO = re.compile(r"\b(\d{8})\b")

def es_dni_valido(d):
    if len(d) != 8 or not d.isdigit(): return False
    if d.startswith('0000') or int(d) < 10000000: return False
    return True

def limpiar_dni(cad): return cad.replace('.', '').replace(',', '').replace(' ', '')

def extraer_dni_de_pdf(ruta_pdf, allowed_dni_set):
    texto = extraer_texto_pdf_y_roi(ruta_pdf)
    candidatos = []
    for m in REGEX_DNI_KEY.finditer(texto):
        d = limpiar_dni(m.group(1))
        if es_dni_valido(d): candidatos.append(d)
    if not candidatos:
        for m in REGEX_DNI_PUNTOS.finditer(texto):
            d = limpiar_dni(m.group(1))
            if es_dni_valido(d): candidatos.append(d)
    if not candidatos:
        for m in REGEX_DNI_CORRIDO.finditer(texto):
            d = m.group(1)
            if es_dni_valido(d): candidatos.append(d)
            
    if allowed_dni_set:
        for cand in candidatos:
            if limpiar_dni(cand) in allowed_dni_set: return cand
    return candidatos[0] if candidatos else ''

# --- Periodo ---
REGEX_PERIODO_AAAAMM = re.compile(r"\b(20\d{2})(0[1-9]|1[0-2])\b")
REGEX_PERIODO_AAAA_MM = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
_MONTH_KEYS = list(MES_MAP.keys()) + list(ABBR_MAP.keys())
_MONTH_KEYS.sort(key=len, reverse=True)
MONTH_PATTERN = r"(" + "|".join(map(re.escape, _MONTH_KEYS)) + r")"
REGEX_PERIODO_TEXTO = re.compile(rf"\b{MONTH_PATTERN}\s*(?:,|\s+de)?\s*(20\d{{2}})\b", re.IGNORECASE)

def extraer_periodo_de_pdf(ruta_pdf):
    raw = extraer_texto_pdf_y_roi(ruta_pdf)
    texto = _normalize_str(raw)
    m = REGEX_PERIODO_AAAAMM.search(texto)
    if m: return m.group(1) + m.group(2)
    m = REGEX_PERIODO_AAAA_MM.search(texto)
    if m: return m.group(1) + m.group(2)
    m = REGEX_PERIODO_TEXTO.search(texto)
    if m:
        mes_txt = _normalize_str(m.group(1))
        yyyy = m.group(2)
        mm = MES_MAP.get(mes_txt, ABBR_MAP.get(mes_txt))
        if mm: return yyyy + mm
    return ''

# --- QR ---
def extraer_url_de_pdf(ruta_pdf):
    try:
        doc = fitz.open(ruta_pdf)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(ZOOM,ZOOM))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height,pix.width,pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        doc.close()
        detector = cv2.QRCodeDetector()
        
        # Intento rápido
        d, _, _ = detector.detectAndDecode(img)
        if d and d.startswith(('http','https')): return d
        
        # Intento con ROI
        for xf,yf,wf,hf in REGIONES_QR:
             h,w = img.shape[:2]
             roi = img[int(h*yf):int(h*(yf+hf)), int(w*xf):int(w*(xf+wf))]
             d, _, _ = detector.detectAndDecode(roi)
             if d and d.startswith(('http','https')): return d
    except: pass
    return None

def parsear_campos_qr(url):
    qs = parse_qs(urlparse(url).query)
    p = qs.get('p',[None])[0] or (re.search(r'[?&]p=([^&]+)',url) or [None])[0]
    if not p: return None,None,None,None
    try:
        dec = base64.b64decode(p).decode('utf-8')
    except: return None,None,None,None
    if dec.strip().startswith('{'):
        try:
            d = json.loads(dec)
            return (str(d.get('cuit','')), str(d.get('tipoCmp','')), str(d.get('ptoVta','')), str(d.get('nroCmp','')))
        except: return None,None,None,None
    parts = dec.split('|')
    return (parts[0], parts[1], parts[2], parts[3] if len(parts)>=4 else None)


# ================== INTERFAZ STREAMLIT ==================

st.set_page_config(page_title="Escáner AFIP", page_icon="📂")

st.title("📂 Escáner de Facturas QR AFIP")

# 1. Subida de Padrón
st.subheader("1. Base de datos Padrón (Opcional)")
padron_file = st.file_uploader("Subir Listado de Certificados/DNI (.xlsx)", type=['xlsx', 'xls'], help="Al cargar un archivo Excel con columnas CUIL, CUD, Vencimiento, Edad, Dependiencia (Si o No), les dara prioridad a estos DNI en la lectura.")
allowed_dni_set = set()

if padron_file:
    try:
        df_padron = pd.read_excel(padron_file, dtype=str)
        cuil_col = None
        for col in df_padron.columns:
            if "cuil" in col.lower():
                cuil_col = col
                break
        if cuil_col:
            for val in df_padron[cuil_col].dropna():
                dni = _dni_from_cuil(val)
                if len(dni) == 8: allowed_dni_set.add(dni)
            st.success(f"✅ Padrón cargado: {len(allowed_dni_set)} DNIs válidos.")
    except Exception as e:
        st.error(f"Error leyendo padrón: {e}")

# 2. Subida de Facturas y Opciones
st.subheader("2. Procesamiento de Facturas")
uploaded_files = st.file_uploader("Subir Facturas PDF", type="pdf", accept_multiple_files=True)

# --- OPCIONES ---
col1, col2, col3 = st.columns(3)
with col1:
    opt_renombrar = st.checkbox("Renombrar archivos PDF", value=True, help="Si se marca, los archivos se renombrarán a CUIT_TIPO_PTO_NRO.pdf")
with col2:
    opt_excel = st.checkbox("Generar Excel reporte", value=True, help="Genera un Excel con el detalle.")
with col3:
    opt_solo_original = st.checkbox("Factura original", value=True, help="Si se marca, se eliminan las hojas extra, dejando solo la primera página.")

if st.button("Procesar Facturas") and uploaded_files:
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Guardar archivos subidos en temp
        paths = []
        for uploaded_file in uploaded_files:
            p = os.path.join(temp_dir, uploaded_file.name)
            with open(p, "wb") as f:
                f.write(uploaded_file.getbuffer())
            paths.append(p)

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        rows = []
        files_to_zip = [] # Tuplas (nombre_en_zip, ruta_fisica)

        for i, path in enumerate(paths):
            fn = os.path.basename(path)
            status_text.text(f"Procesando: {fn}")
            
            error, url, dni, periodo = '', '', '', ''
            cuit, tipo, pto, nro, sss = '', '', '', '', ''

            # Extracciones
            dni = extraer_dni_de_pdf(path, allowed_dni_set)
            periodo = extraer_periodo_de_pdf(path)
            url_detect = extraer_url_de_pdf(path)

            if not url_detect:
                error = 'QR no detectado'
            else:
                url = url_detect
                cuit, tipo, pto, nro = parsear_campos_qr(url)
                sss = TIPO_MAP.get(tipo,'')
                missing = [n for n,v in zip(['CUIT','Tipo','Pto','Nro'], [cuit,tipo,pto,nro]) if not v]
                if missing: error = 'Faltan campos: ' + ', '.join(missing)

            if dni and nro and dni.lstrip('0') == (nro or '').lstrip('0'): dni = ''
            if dni and cuit:
                cuit_digits = re.sub(r'\D', '', cuit)
                if len(cuit_digits) == 11:
                    dni_prestador = cuit_digits[2:10]
                    if dni == dni_prestador: dni = ''

            # Determinar el nombre base detectado
            nombre_calculado = f"{cuit}_{sss}_{pto}_{nro}" if (cuit and sss and pto and nro) else ""

            # Lógica de guardado y manipulación PDF
            try:
                doc = fitz.open(path); new = fitz.open()
                new.insert_pdf(doc)
                
                # --- NUEVA LÓGICA: Eliminar hojas extra SOLO si el usuario lo pide ---
                if opt_solo_original:
                    for p in range(new.page_count-1,0,-1):
                        new.delete_page(p)
                
                # Decidir nombre final
                if opt_renombrar and not error and nombre_calculado:
                    nombre_final_zip = nombre_calculado + ".pdf"
                else:
                    nombre_final_zip = fn # Nombre original
                
                out_path = os.path.join(temp_dir, "temp_processed_" + fn)
                new.save(out_path)
                new.close(); doc.close()
                
                files_to_zip.append((nombre_final_zip, out_path))
                
            except Exception as e:
                # Fallback en caso de error critico con el PDF
                error = f"Error guardando PDF: {e}" if not error else error + f" | Error PDF: {e}"
                files_to_zip.append((fn, path))

            rows.append({
                'Archivo Original': fn,
                'Archivo Generado': nombre_calculado + ".pdf" if (not error and nombre_calculado) else '',
                'Codigo QR': url,
                'DNI': dni,
                'Periodo': periodo,
                'Error': error
            })
            
            progress_bar.progress((i + 1) / len(paths))

        # Generar ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            
            if opt_excel:
                df = pd.DataFrame(rows, columns=['Archivo Original', 'Archivo Generado', 'Codigo QR', 'DNI', 'Periodo', 'Error'])
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                zf.writestr("Reporte_Procesamiento.xlsx", excel_buffer.getvalue())
            
            for name, filepath in files_to_zip:
                zf.write(filepath, name)
        
        st.success("¡Procesamiento completado!")
        st.download_button(
            label="⬇️ Descargar ZIP",
            data=zip_buffer.getvalue(),
            file_name="facturas_procesadas.zip",
            mime="application/zip"
        )



