import asyncio
import httpx
from pathlib import Path

CORPUS_DIR = Path("./corpus")
RESULTADOS_DIR = Path("./resultados")
URL = "http://localhost:8000/simplificar"

async def procesar_archivo(client, archivo):
    texto = archivo.read_text(encoding="utf-8")
    print(f"Procesando: {archivo.name}")
    try:
        response = await client.post(URL, json={"texto": texto})
        response.raise_for_status()
        resultado = response.json()

        if "resultado" not in resultado:
            print(f"ERROR en {archivo.name}: {resultado.get('error', resultado)}")
            return

        respuesta = resultado["resultado"]
        separador = "--- GLOSARIO ---"

        if separador in respuesta:
            texto_simplificado, glosario = respuesta.split(separador, 1)
            texto_simplificado = texto_simplificado.strip()
            glosario = glosario.strip()
        else:
            print(f"AVISO: {archivo.name} no contiene glosario.")
            texto_simplificado = respuesta.strip()
            glosario = ""

        RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
        nombre = archivo.stem

        (RESULTADOS_DIR / f"{nombre}.txt").write_text(texto_simplificado, encoding="utf-8")
        (RESULTADOS_DIR / f"{nombre}glosario.txt").write_text(glosario, encoding="utf-8")

        print(f" Texto: {RESULTADOS_DIR / f'{nombre}.txt'}")
        print(f" Glosario: {RESULTADOS_DIR / f'{nombre}glosario.txt'}")

    except Exception as e:
        print(f"ERROR procesando {archivo.name}: {e}")

async def main():
    archivos = [a for a in CORPUS_DIR.glob("*.txt") if not a.stem.endswith("glosario")]
    archivos.sort(key=lambda a: int(a.stem.lstrip("_")) if a.stem.lstrip("_").isdigit() else float("inf"))

    if not archivos:
        print("No se han encontrado archivos .txt en ./corpus")
        return

    print(f"Encontrados {len(archivos)} textos.\n")

    async with httpx.AsyncClient(timeout=None) as client:
        for archivo in archivos:
            await procesar_archivo(client, archivo)
            await asyncio.sleep(1)

    print("\nProcesamiento terminado.")

if __name__ == "__main__":
    asyncio.run(main())