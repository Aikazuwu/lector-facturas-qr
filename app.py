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
# NOTA: En Streamlit Cloud esto se ignora porque se usa packages.txt.
# Si lo corres en local (Windows) y no detecta tesseract, descomenta la línea:
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

def extraer_dni_de_pdf(ruta_pdf, padron_dict):
    # padron_dict es un diccionario {dni: {datos...}}
    texto = extraer_texto_pdf_y_roi(ruta_pdf)
    candidatos = []
    
    # Búsquedas Regex
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
            
    # Prioridad: Si está en el padrón, ganamos.
    if padron_dict:
        for cand in candidatos:
            clean = limpiar_dni(cand)
            if clean in padron_dict:
                return clean
    
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

# --- QR (Actualizado para sacar Fecha, Importe, CAE) ---
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

def parsear_campos_qr_completo(url):
    """
    Retorna: cuit, tipo, pto, nro, fecha, importe, cae
    """
    qs = parse_qs(urlparse(url).query)
    p = qs.get('p',[None])[0] or (re.search(r'[?&]p=([^&]+)',url) or [None])[0]
    
    # Valores por defecto
    res = {'cuit':'', 'tipo':'', 'pto':'', 'nro':'', 'fecha':'', 'importe':'', 'cae':''}
    
    if not p: return res
    try:
        dec = base64.b64decode(p).decode('utf-8')
    except: return res

    if dec.strip().startswith('{'):
        try:
            d = json.loads(dec)
            res['cuit'] = str(d.get('cuit',''))
            res['tipo'] = str(d.get('tipoCmp',''))
            res['pto']  = str(d.get('ptoVta',''))
            res['nro']  = str(d.get('nroCmp',''))
            res['fecha']= str(d.get('fecha','')) # Formato YYYY-MM-DD usualmente en JSON AFIP
            res['importe'] = str(d.get('importe',''))
            res['cae'] = str(d.get('codAut',''))
            return res
        except: return res
        
    # Formato antiguo pipe (menos común en QR V1, pero posible)
    # orden usual: version|fecha|cuit|pto|tipo|nro|importe|moneda|cotiz|tipoDoc|nroDoc|tipoCAE|CAE
    parts = dec.split('|')
    if len(parts) >= 13:
        res['fecha'] = parts[1]
        res['cuit'] = parts[2]
        res['pto'] = parts[3]
        res['tipo'] = parts[4]
        res['nro'] = parts[5]
        res['importe'] = parts[6]
        res['cae'] = parts[12]
    return res


# ================== INTERFAZ STREAMLIT ==================

st.set_page_config(page_title="Escáner AFIP", page_icon="📂", layout="wide")

st.title("📂 Escáner de Facturas QR AFIP")

# --- BLOQUE 1: PADRÓN ---
st.subheader("1. Base de Datos: Padrón (Afiliados)")
padron_file = st.file_uploader("Subir Excel Padrón (.xlsx)", type=['xlsx', 'xls'], key="padron", help="Debe contener columnas: CUIL, CUD, Vencimiento, Dependencia.")
padron_data = {} # Diccionario: dni -> {cuil, cud, venc, dep}

if padron_file:
    try:
        df_padron = pd.read_excel(padron_file, dtype=str)
        # Normalizar columnas
        df_padron.columns = [_normalize_str(c) for c in df_padron.columns]
        
        # Buscar columnas clave
        col_cuil = next((c for c in df_padron.columns if 'cuil' in c), None)
        col_cud  = next((c for c in df_padron.columns if 'cud' in c), None)
        col_venc = next((c for c in df_padron.columns if 'venc' in c), None)
        col_dep  = next((c for c in df_padron.columns if 'dep' in c), None) # Dependencia

        if col_cuil:
            count = 0
            for _, row in df_padron.iterrows():
                cuil_val = _normalize_digits(row[col_cuil])
                if len(cuil_val) == 11:
                    dni_val = cuil_val[2:10]
                    padron_data[dni_val] = {
                        'cuil': cuil_val,
                        'cud': str(row[col_cud]) if col_cud and pd.notna(row[col_cud]) else "",
                        'venc': str(row[col_venc]) if col_venc and pd.notna(row[col_venc]) else "",
                        'dep': str(row[col_dep]) if col_dep and pd.notna(row[col_dep]) else ""
                    }
                    count += 1
            st.success(f"✅ Padrón procesado: {count} registros indexados por DNI.")
        else:
            st.error("❌ No se encontró la columna 'CUIL' en el Excel del Padrón.")
    except Exception as e:
        st.error(f"Error leyendo padrón: {e}")

# --- BLOQUE 2: PRESTACIONES ---
st.subheader("2. Base de Datos: Prestaciones (Opcional para TXT)")
prestaciones_file = st.file_uploader("Subir Excel Prestaciones (.xlsx)", type=['xlsx', 'xls'], key="prestaciones", help="Debe contener columnas: CUIL, CUIT, CODIGO, CANTIDAD")
prestaciones_data = {} # Clave: (cuil_afiliado, cuit_prestador) -> {codigo, cantidad}

if prestaciones_file:
    try:
        df_prest = pd.read_excel(prestaciones_file, dtype=str)
        df_prest.columns = [_normalize_str(c) for c in df_prest.columns]
        
        p_cuil = next((c for c in df_prest.columns if 'cuil' in c), None)
        p_cuit = next((c for c in df_prest.columns if 'cuit' in c), None)
        p_cod  = next((c for c in df_prest.columns if 'cod' in c), None) # Codigo
        p_cant = next((c for c in df_prest.columns if 'cant' in c), None) # Cantidad

        if p_cuil and p_cuit and p_cod and p_cant:
            # Detectar duplicados de clave (CUIL + CUIT)
            # Creamos una columna tupla para agrupar
            df_prest['key_tuple'] = list(zip(df_prest[p_cuil].apply(_normalize_digits), df_prest[p_cuit].apply(_normalize_digits)))
            
            # Contar ocurrencias
            counts = df_prest['key_tuple'].value_counts()
            
            loaded_count = 0
            for _, row in df_prest.iterrows():
                k_cuil = _normalize_digits(row[p_cuil])
                k_cuit = _normalize_digits(row[p_cuit])
                key = (k_cuil, k_cuit)
                
                # REGLA: Si la clave aparece más de una vez (ambigüedad), no cargamos nada
                if counts.get(key, 0) == 1:
                    prestaciones_data[key] = {
                        'codigo': str(row[p_cod]),
                        'cantidad': str(row[p_cant])
                    }
                    loaded_count += 1
            
            st.success(f"✅ Base Prestaciones procesada: {loaded_count} pares únicos (CUIL-CUIT) cargados. (Duplicados ignorados).")
        else:
            st.warning("⚠️ No se encontraron las columnas requeridas (CUIL, CUIT, CODIGO, CANTIDAD) en el archivo.")
    except Exception as e:
        st.error(f"Error leyendo prestaciones: {e}")

# --- BLOQUE 3: PROCESAMIENTO ---
st.subheader("3. Subida y Configuración")

col_rnos, col_blank = st.columns([1, 2])
with col_rnos:
    rnos_input = st.text_input("Ingresar RNOS (Requerido para TXT)", "")

uploaded_files = st.file_uploader("Subir Facturas PDF", type="pdf", accept_multiple_files=True)

# Opciones
c1, c2, c3, c4 = st.columns(4)
with c1:
    opt_renombrar = st.checkbox("Renombrar PDFs", value=True)
with c2:
    opt_excel = st.checkbox("Generar Excel", value=True)
with c3:
    opt_solo_original = st.checkbox("Limpiar hojas extra", value=True)
with c4:
    opt_txt = st.checkbox("Generar TXT Salida", value=False, help="Requiere RNOS, Padrón y Base de Prestaciones cargados.")

# Botón
if st.button("Procesar Facturas") and uploaded_files:
    
    # Validaciones previas para TXT
    generar_txt = False
    if opt_txt:
        errors_txt = []
        if not rnos_input: errors_txt.append("Falta RNOS")
        if not padron_data: errors_txt.append("Falta cargar Padrón")
        if not prestaciones_data: errors_txt.append("Falta cargar Prestaciones")
        
        if errors_txt:
            st.warning(f"⚠️ No se generará el TXT: {', '.join(errors_txt)}")
        else:
            generar_txt = True

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = []
        for uploaded_file in uploaded_files:
            p = os.path.join(temp_dir, uploaded_file.name)
            with open(p, "wb") as f:
                f.write(uploaded_file.getbuffer())
            paths.append(p)

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        rows = []
        files_to_zip = [] 
        txt_lines = [] # Aquí acumulamos las filas del TXT

        for i, path in enumerate(paths):
            fn = os.path.basename(path)
            status_text.text(f"Procesando: {fn}")
            
            error = ''
            
            # 1. Extracción Datos PDF
            dni = extraer_dni_de_pdf(path, padron_data)
            periodo = extraer_periodo_de_pdf(path)
            url_detect = extraer_url_de_pdf(path)

            qr_data = {'cuit':'', 'tipo':'', 'pto':'', 'nro':'', 'fecha':'', 'importe':'', 'cae':''}
            
            if not url_detect:
                error = 'QR no detectado'
            else:
                qr_data = parsear_campos_qr_completo(url_detect)
                # Validar campos mínimos del QR
                missing = [k for k,v in qr_data.items() if k in ['cuit','tipo','pto','nro'] and not v]
                if missing: error = 'Faltan campos QR: ' + ', '.join(missing)

            # Validaciones de lógica negocio
            if dni and qr_data['nro'] and dni.lstrip('0') == qr_data['nro'].lstrip('0'): dni = ''
            if dni and qr_data['cuit']:
                cuit_digits = re.sub(r'\D', '', qr_data['cuit'])
                if len(cuit_digits) == 11 and dni == cuit_digits[2:10]:
                    dni = ''
            
            sss = TIPO_MAP.get(qr_data['tipo'],'')
            
            # Nombre calculado
            nombre_calculado = ""
            if qr_data['cuit'] and sss and qr_data['pto'] and qr_data['nro']:
                nombre_calculado = f"{qr_data['cuit']}_{sss}_{qr_data['pto']}_{qr_data['nro']}"

            # --- GENERACIÓN LÍNEA TXT ---
            # Condiciones: Usuario quiere TXT + Bases cargadas + QR OK + DNI OK + Periodo OK
            if generar_txt and not error and dni and periodo:
                
                # Datos del Padrón
                p_info = padron_data.get(dni)
                
                if p_info:
                    cuil_afiliado = p_info['cuil']
                    cud_cod = p_info['cud']
                    cud_venc = p_info['venc']
                    dependencia = p_info['dep']
                    
                    # Datos Prestaciones (Key: CUIL Afiliado + CUIT Factura)
                    key_prest = (cuil_afiliado, _normalize_digits(qr_data['cuit']))
                    prest_info = prestaciones_data.get(key_prest)
                    
                    cod_prestacion = ""
                    cantidad = ""
                    
                    if prest_info:
                        cod_prestacion = prest_info['codigo']
                        cantidad = prest_info['cantidad']
                    
                    # Armado de columnas
                    col_1 = "DS"
                    col_2 = rnos_input
                    col_3 = cuil_afiliado
                    col_4 = cud_cod
                    col_5 = cud_venc
                    col_6 = periodo
                    col_7 = qr_data['cuit']
                    col_8 = sss # Tipo mapeado
                    col_9 = "E"
                    col_10 = qr_data['fecha'] # Fecha emision
                    col_11 = qr_data['cae']
                    col_12 = qr_data['pto']
                    col_13 = qr_data['nro']
                    col_14 = qr_data['importe']
                    col_15 = qr_data['importe'] # Solicitado = Total
                    col_16 = cod_prestacion
                    col_17 = cantidad
                    col_18 = "00"
                    col_19 = dependencia
                    
                    linea = f"{col_1}|{col_2}|{col_3}|{col_4}|{col_5}|{col_6}|{col_7}|{col_8}|{col_9}|{col_10}|{col_11}|{col_12}|{col_13}|{col_14}|{col_15}|{col_16}|{col_17}|{col_18}|{col_19}"
                    txt_lines.append(linea)

            # --- MANEJO DE ARCHIVOS (PDF) ---
            try:
                doc = fitz.open(path); new = fitz.open()
                new.insert_pdf(doc)
                if opt_solo_original:
                    for p in range(new.page_count-1,0,-1): new.delete_page(p)
                
                final_name = fn
                if opt_renombrar and not error and nombre_calculado:
                    final_name = nombre_calculado + ".pdf"
                
                out_path = os.path.join(temp_dir, "proc_" + fn)
                new.save(out_path)
                new.close(); doc.close()
                files_to_zip.append((final_name, out_path))
            except Exception as e:
                error = f"Error PDF: {e}"
                files_to_zip.append((fn, path))

            # Reporte Excel
            rows.append({
                'Archivo Original': fn,
                'Archivo Generado': nombre_calculado + ".pdf" if (not error and nombre_calculado) else '',
                'Codigo QR': url_detect,
                'DNI': dni,
                'Periodo': periodo,
                'Error': error
            })
            progress_bar.progress((i + 1) / len(paths))

        # --- ZIP FINAL ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # 1. Excel
            if opt_excel:
                df = pd.DataFrame(rows, columns=['Archivo Original', 'Archivo Generado', 'Codigo QR', 'DNI', 'Periodo', 'Error'])
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                zf.writestr("Reporte_Procesamiento.xlsx", excel_buffer.getvalue())
            
            # 2. TXT
            if generar_txt and txt_lines:
                txt_content = "\n".join(txt_lines)
                zf.writestr("salida_sistema.txt", txt_content)
            elif opt_txt and not txt_lines:
                # Si el usuario pidió TXT pero no salió ninguna línea (por errores o falta de match)
                zf.writestr("salida_sistema_VACIO.txt", "No se pudieron generar líneas. Verifique errores en Excel o cruce de datos.")

            # 3. PDFs
            for name, filepath in files_to_zip:
                zf.write(filepath, name)
        
        st.success("¡Proceso Finalizado!")
        st.download_button("⬇️ Descargar ZIP Completo", zip_buffer.getvalue(), "resultados_procesados.zip", "application/zip")



