from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import httpx
from models import AccidenteRequest, AccidenteResponse, CombinedResponse
from utils import procesar_csv, combinar_datos, identificar_zonas_peligrosas
import io
import os
from pathlib import Path
from datetime import datetime
import shutil
from math import radians, sin, cos, sqrt, atan2

app = FastAPI(title="Sistema de Análisis de Accidentes")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_BASE_URL = "http://localhost:8080/api/accidentes/cercanos"
CSV_DIRECTORY = "data"

Path(CSV_DIRECTORY).mkdir(exist_ok=True)

# ============================================
# 🔧 FUNCIÓN HAVERSINE - DEBE ESTAR AQUÍ
# ============================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en metros entre dos coordenadas usando la fórmula de Haversine
    
    Args:
        lat1, lon1: Coordenadas del primer punto (en grados)
        lat2, lon2: Coordenadas del segundo punto (en grados)
    
    Returns:
        float: Distancia en metros
    """
    R = 6371000  # Radio de la Tierra en metros
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def buscar_csv_defecto():
    """
    Busca el archivo CSV más reciente en el directorio de datos
    Retorna el path del CSV más reciente o None
    """
    csv_dir = Path(CSV_DIRECTORY)
    csv_files = list(csv_dir.glob("*.csv"))
    
    if csv_files:
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return csv_files[0]
    return None

def listar_todos_csv():
    """
    Lista todos los archivos CSV en el directorio de datos
    """
    csv_dir = Path(CSV_DIRECTORY)
    csv_files = list(csv_dir.glob("*.csv"))
    
    archivos = []
    for csv_file in csv_files:
        stat = csv_file.stat()
        archivos.append({
            "nombre": csv_file.name,
            "path": str(csv_file),
            "tamaño_bytes": stat.st_size,
            "fecha_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "es_mas_reciente": csv_file == buscar_csv_defecto()
        })
    
    archivos.sort(key=lambda x: x["fecha_modificacion"], reverse=True)
    return archivos

async def guardar_csv(archivo: UploadFile) -> Path:
    """
    Guarda el archivo CSV subido en el directorio de datos
    Retorna el path del archivo guardado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_original = archivo.filename.replace('.csv', '')
    nombre_archivo = f"{nombre_original}_{timestamp}.csv"
    
    file_path = Path(CSV_DIRECTORY) / nombre_archivo
    
    contenido = await archivo.read()
    with open(file_path, 'wb') as f:
        f.write(contenido)
    
    print(f"💾 CSV guardado: {nombre_archivo} ({len(contenido)} bytes)")
    
    return file_path

# ============================================
# PYDANTIC MODELS
# ============================================
from pydantic import BaseModel

class PuntoRuta(BaseModel):
    lat: float
    lng: float

class AnalisisRutaRequest(BaseModel):
    puntos_ruta: List[PuntoRuta]
    radio_deteccion: int = 300  # metros

# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Accidentes de Tránsito",
        "version": "1.0",
        "endpoints": {
            "analizar": "POST /analizar",
            "analizar_ruta_riesgo": "POST /analizar-ruta-riesgo/",
            "zonas_alto_riesgo": "POST /zonas-alto-riesgo/",
            "estadisticas": "POST /estadisticas-csv",
            "test_api": "GET /api-externa/test",
            "csv_disponible": "GET /csv-disponible",
            "listar_csv": "GET /listar-csv",
            "eliminar_csv": "DELETE /eliminar-csv/{nombre}"
        }
    }

@app.get("/csv-disponible")
async def verificar_csv_disponible():
    """
    Verifica si existe un CSV por defecto (el más reciente) en el directorio de datos
    """
    csv_path = buscar_csv_defecto()
    
    if csv_path:
        stat = csv_path.stat()
        return {
            "existe": True,
            "nombre": csv_path.name,
            "path": str(csv_path),
            "tamaño_bytes": stat.st_size,
            "fecha_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    return {
        "existe": False,
        "mensaje": f"No se encontró ningún archivo CSV en el directorio '{CSV_DIRECTORY}'"
    }

@app.get("/listar-csv")
async def listar_csv_guardados():
    """
    Lista todos los archivos CSV guardados en el servidor
    """
    archivos = listar_todos_csv()
    
    return {
        "total": len(archivos),
        "archivos": archivos,
        "directorio": CSV_DIRECTORY
    }

@app.delete("/eliminar-csv/{nombre_archivo}")
async def eliminar_csv(nombre_archivo: str):
    """
    Elimina un archivo CSV específico del servidor
    """
    try:
        file_path = Path(CSV_DIRECTORY) / nombre_archivo
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"El archivo '{nombre_archivo}' no existe"
            )
        
        file_path.unlink()
        print(f"🗑️ CSV eliminado: {nombre_archivo}")
        
        return {
            "success": True,
            "mensaje": f"Archivo '{nombre_archivo}' eliminado correctamente",
            "archivos_restantes": len(listar_todos_csv())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar archivo: {str(e)}")

@app.post("/analizar")
async def analizar_accidentes(
    archivo: Optional[UploadFile] = File(None),
    usar_csv_defecto: Optional[str] = Form(None),
    latitud: float = Form(default=-2.89264),
    longitud: float = Form(default=-78.77814),
    radio_km: float = Form(default=5.0)
):
    """
    Analiza accidentes combinando datos del CSV y la API externa
    """
    try:
        df_csv = None
        archivo_usado = None
        
        if usar_csv_defecto == "true":
            csv_path = buscar_csv_defecto()
            
            if not csv_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró ningún archivo CSV en el directorio '{CSV_DIRECTORY}'"
                )
            
            print(f"📂 Usando CSV por defecto: {csv_path.name}")
            df_csv = pd.read_csv(csv_path)
            archivo_usado = csv_path.name
            
        elif archivo:
            if not archivo.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
            
            saved_path = await guardar_csv(archivo)
            df_csv = pd.read_csv(saved_path)
            archivo_usado = saved_path.name
            
            print(f"📤 CSV nuevo guardado: {archivo_usado}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Debes proporcionar un archivo CSV o usar el CSV por defecto"
            )
        
        if len(df_csv) == 0:
            raise HTTPException(status_code=400, detail="El CSV no contiene datos")
        
        required_cols = ['latitud', 'longitud', 'tipo_accidente', 'provincia', 'ciudad']
        missing_cols = [col for col in required_cols if col not in df_csv.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {', '.join(missing_cols)}"
            )
        
        datos_csv = procesar_csv(df_csv, latitud, longitud, radio_km)
        
        datos_api = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    API_BASE_URL,
                    params={"lat": latitud, "lon": longitud}
                )
                
                if response.status_code == 200:
                    datos_api = response.json()
                    print(f"🌐 API externa retornó {len(datos_api)} accidentes")
                    
        except Exception as e:
            print(f"⚠️ Error API externa: {str(e)}")
        
        resultado = combinar_datos(datos_csv, datos_api, latitud, longitud)
        resultado["archivo_csv_usado"] = archivo_usado
        resultado["total_archivos_guardados"] = len(listar_todos_csv())
        
        return resultado
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/estadisticas-csv")
async def estadisticas_csv(archivo: UploadFile = File(...)):
    """
    Obtiene estadísticas básicas del CSV cargado
    """
    try:
        contenido = await archivo.read()
        df = pd.read_csv(io.BytesIO(contenido))
        
        return {
            "total_registros": len(df),
            "columnas": list(df.columns),
            "provincias": df['provincia'].value_counts().to_dict() if 'provincia' in df.columns else {},
            "ciudades": df['ciudad'].value_counts().head(10).to_dict() if 'ciudad' in df.columns else {},
            "tipos_accidente": df['tipo_accidente'].value_counts().to_dict() if 'tipo_accidente' in df.columns else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api-externa/test")
async def test_api_externa(lat: float = -2.89264, lon: float = -78.77814):
    """
    Prueba la conexión con la API externa
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                API_BASE_URL,
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            data = response.json()
            return {
                "status": "success",
                "total_accidentes": len(data) if isinstance(data, list) else 0,
                "data": data
            }
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502, 
            detail="No se pudo conectar a la API externa"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Verifica el estado de la API
    """
    csv_disponible = buscar_csv_defecto()
    total_csv = len(listar_todos_csv())
    
    return {
        "status": "healthy",
        "service": "Sistema de Análisis de Accidentes",
        "version": "1.0",
        "csv_por_defecto": {
            "disponible": csv_disponible is not None,
            "archivo": csv_disponible.name if csv_disponible else None
        },
        "total_archivos_csv": total_csv,
        "endpoints_disponibles": [
            "/analizar",
            "/analizar-ruta-riesgo/",
            "/zonas-alto-riesgo/",
            "/estadisticas-csv",
            "/api-externa/test",
            "/csv-disponible",
            "/listar-csv"
        ]
    }

# ============================================
# 🔥 ENDPOINT PRINCIPAL: ANÁLISIS DE RUTA
# ============================================
@app.post("/analizar-ruta-riesgo/")
async def analizar_ruta_riesgo(request: AnalisisRutaRequest):
    """
    Analiza una ruta específica y retorna las zonas de riesgo que la cruzan
    
    Args:
        request: Contiene puntos_ruta (lista de coordenadas) y radio_deteccion
    
    Returns:
        JSON con las zonas detectadas que impactan o están cerca de la ruta
    """
    try:
        print(f"🔍 Analizando ruta con {len(request.puntos_ruta)} puntos")
        
        csv_path = buscar_csv_defecto()
        
        if not csv_path:
            raise HTTPException(
                status_code=404,
                detail="No se encontró archivo CSV. Por favor, carga un archivo primero."
            )
        
        df_csv = pd.read_csv(csv_path)
        
        if len(df_csv) == 0:
            raise HTTPException(status_code=400, detail="El CSV no contiene datos")
        
        # Convertir puntos de ruta
        puntos_ruta = [(p.lat, p.lng) for p in request.puntos_ruta]
        
        # 🔥 PASO 1: Filtrar accidentes cercanos a la ruta (1km buffer)
        print("📊 Filtrando accidentes cerca de la ruta...")
        accidentes_cercanos = []
        for _, row in df_csv.iterrows():
            if pd.notna(row['latitud']) and pd.notna(row['longitud']):
                for punto in puntos_ruta:
                    dist = haversine(punto[0], punto[1], row['latitud'], row['longitud'])
                    if dist <= 1000:  # 1km de buffer
                        accidentes_cercanos.append((row['latitud'], row['longitud']))
                        break
        
        print(f"📊 Accidentes cerca de la ruta: {len(accidentes_cercanos)} de {len(df_csv)}")
        
        if len(accidentes_cercanos) == 0:
            print("✅ No hay accidentes cerca de esta ruta")
            return {
                "puntos_ruta_analizados": len(puntos_ruta),
                "zonas_en_ruta": [],
                "puntos_en_riesgo": 0,
                "radio_deteccion_metros": request.radio_deteccion,
                "archivo_csv_usado": csv_path.name
            }
        
        # 🔥 PASO 2: Identificar zonas de riesgo
        print("🗺️ Identificando zonas de riesgo...")
        zonas_detectadas = identificar_zonas_peligrosas(
            accidentes_cercanos, 
            radio_metros=300,
            min_accidentes=2
        )
        
        print(f"🗺️ Zonas detectadas: {len(zonas_detectadas)}")
        
        # 🔥 PASO 3: Clasificar zonas según impacto en la ruta
        zonas_en_ruta = []
        puntos_en_riesgo = 0
        
        for zona in zonas_detectadas:
            distancia_minima = float('inf')
            impacta_ruta = False
            
            # Verificar proximidad a cada punto de la ruta
            for punto in puntos_ruta:
                distancia = haversine(
                    punto[0], punto[1],
                    zona['latitud'], zona['longitud']
                )
                
                if distancia < distancia_minima:
                    distancia_minima = distancia
                
                # Si está dentro del radio de detección, impacta la ruta
                if distancia <= request.radio_deteccion:
                    impacta_ruta = True
                    puntos_en_riesgo += 1
            
            # Agregar zona con información de impacto
            zona_con_info = {
                **zona,
                'distancia_minima': round(distancia_minima, 2),
                'impacta_ruta': impacta_ruta
            }
            zonas_en_ruta.append(zona_con_info)
        
        # Ordenar: primero las que impactan, luego por cantidad de accidentes
        zonas_en_ruta.sort(key=lambda z: (not z['impacta_ruta'], -z['cantidad_accidentes']))
        
        print(f"✅ Zonas en ruta: {len(zonas_en_ruta)}")
        print(f"🚨 Zonas que IMPACTAN: {len([z for z in zonas_en_ruta if z['impacta_ruta']])}")
        
        return {
            "puntos_ruta_analizados": len(puntos_ruta),
            "zonas_en_ruta": zonas_en_ruta,
            "puntos_en_riesgo": puntos_en_riesgo,
            "radio_deteccion_metros": request.radio_deteccion,
            "archivo_csv_usado": csv_path.name
        }
        
    except Exception as e:
        print(f"❌ Error en analizar-ruta-riesgo: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al analizar ruta: {str(e)}")

# ============================================
# ENDPOINT: ZONAS DE ALTO RIESGO GENERAL
# ============================================
@app.post("/zonas-alto-riesgo/")
async def obtener_zonas_alto_riesgo(request: AccidenteRequest):
    """
    Obtiene zonas de alto riesgo en un área específica
    """
    try:
        csv_path = buscar_csv_defecto()
        
        if not csv_path:
            raise HTTPException(
                status_code=404,
                detail="No se encontró ningún archivo CSV"
            )
        
        df_csv = pd.read_csv(csv_path)
        
        if len(df_csv) == 0:
            raise HTTPException(status_code=400, detail="El CSV no contiene datos")
        
        required_cols = ['latitud', 'longitud']
        missing_cols = [col for col in required_cols if col not in df_csv.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas: {', '.join(missing_cols)}"
            )
        
        datos_csv = procesar_csv(df_csv, request.latitud, request.longitud, request.radio_km)
        todos_puntos = [(row['latitud'], row['longitud']) for _, row in datos_csv.iterrows()]
        
        print(f"🔍 Analizando {len(todos_puntos)} puntos")
        
        zonas = identificar_zonas_peligrosas(todos_puntos, radio_metros=300, min_accidentes=2)
        zonas_relevantes = [z for z in zonas if z['nivel_peligro'] in ['ALTO', 'MEDIO']]
        zonas_altas = [z for z in zonas if z['nivel_peligro'] == 'ALTO']
        
        print(f"   - Zonas ALTO+MEDIO: {len(zonas_relevantes)}")
        print(f"   - Zonas ALTO: {len(zonas_altas)}")
        
        return {
            "punto_referencia": {
                "latitud": request.latitud,
                "longitud": request.longitud
            },
            "radio_busqueda_km": request.radio_km,
            "total_accidentes_analizados": len(todos_puntos),
            "zonas_peligrosas": zonas_relevantes,
            "total_zonas_alto_riesgo": len(zonas_altas),
            "total_zonas_relevantes": len(zonas_relevantes),
            "archivo_csv_usado": csv_path.name
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")