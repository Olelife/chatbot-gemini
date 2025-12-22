import os
import json
import logging
from typing import List
from google.cloud import storage

logger = logging.getLogger(__name__)


# ===============================
# LISTAR ARCHIVOS EN GCS
# ===============================
def list_files_in_folder(bucket: str, folder: str) -> List[str]:
    """
    Lista todos los archivos dentro de una carpeta en GCS.
    """
    client = storage.Client()
    bucket_obj = client.bucket(bucket)

    files = [
        blob.name
        for blob in bucket_obj.list_blobs(prefix=folder)
        if not blob.name.endswith("/")
    ]

    if not files:
        logger.warning(f"⚠ No files found in folder: {folder}")

    return files


# ===============================
# CARGAR ARCHIVO COMO CHUNKS SEMÁNTICOS
# ===============================
def load_file_as_units(bucket: str, path: str) -> List[str]:
    """
    Carga un archivo desde GCS y lo convierte en chunks semánticos.

    - Si es .json → genera chunks por cobertura, por sección o por item.
    - Si es texto → chunk único.
    """
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(path)

    if not blob.exists():
        logger.warning(f"File not found: {path}")
        return []

    raw = blob.download_as_bytes()

    # Si NO es JSON → retornar texto como único chunk
    if not path.endswith(".json"):
        try:
            text = raw.decode("utf-8", errors="ignore")
            return [text] if text.strip() else []
        except:
            return []

    # Si es JSON → procesar
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error parsing JSON {path}: {e}")
        return []

    units = []

    # Caso: lista de items
    if isinstance(data, list):
        for entry in data:
            nombre = (
                entry.get("nombre")
                or entry.get("titulo")
                or entry.get("id")
                or "ITEM"
            )
            block = f"## ITEM: {nombre}\n\n{json.dumps(entry, ensure_ascii=False, indent=2)}"
            units.append(block)
        return units

    # Caso: diccionario → chunk único por archivo
    if isinstance(data, dict):

        # Si es cobertura (carpeta incluye "coberturas")
        if "coberturas" in path.lower():
            nombre = (
                data.get("nombre")
                or data.get("id")
                or os.path.basename(path).replace(".json", "")
            )
            block = f"## COBERTURA: {nombre}\n\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            return [block]

        # Caso general → sección
        nombre = os.path.basename(path).replace(".json", "").upper()
        block = f"## SECCIÓN: {nombre}\n\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        return [block]

    # Último recurso
    return [json.dumps(data)]
