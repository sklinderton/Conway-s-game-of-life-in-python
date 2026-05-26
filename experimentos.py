#!/usr/bin/env python3
# experimentos.py
"""
Script principal del proyecto: Juego de la Vida de Conway.

Ejecuta todos los experimentos en orden:
  1. Crea las carpetas de salida.
  2. Genera las animaciones de patrones clásicos (GIFs).
  3. Mide el rendimiento para distintos tamaños de cuadrícula.
  4. Genera las gráficas de rendimiento (escala lineal y log-log).

Cómo ejecutar:
--------------
    uv run python experimentos.py

    # O, si ya activaste el entorno virtual:
    python experimentos.py

    # Solo las animaciones (rápido):
    python experimentos.py --solo-animaciones

    # Solo el rendimiento (sin abrir ventanas):
    python experimentos.py --solo-rendimiento

    # Sin animaciones largas (modo rápido para pruebas):
    python experimentos.py --rapido
"""

import sys
import os
import argparse
import time

# Agregamos src/ al path para poder importar el paquete.
# Esto permite ejecutar el script desde la raíz del proyecto.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Importaciones del proyecto ──────────────────────────────────────────────
from src.juego_vida import (
    GameOfLife,
    crear_glider,
    crear_blinker,
    crear_toad,
    crear_aleatorio,
    NUMBA_DISPONIBLE,
)
from src.visualizacion import (
    animar_patron,
    guardar_animacion,
    crear_todas_las_animaciones,
    PALETA_VERDE,
    PALETA_CLASICA,
)
from src.rendimiento import medir_rendimiento, graficar_rendimiento

# ── Configuración de matplotlib sin ventana interactiva ─────────────────────
# Usamos el backend 'Agg' (Anti-Grain Geometry) cuando guardamos archivos,
# ya que no abre ventanas y funciona sin pantalla (servidores, CI, etc.).
import matplotlib
matplotlib.use("Agg")   # debe ir ANTES de importar pyplot


def separador(titulo: str) -> None:
    """Imprime una línea separadora decorativa para la consola."""
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"{'═'*60}")


def paso_animaciones(directorio: str, pasos: int, intervalo: int) -> None:
    """
    Paso 1: genera y guarda las animaciones de los patrones clásicos.

    Los GIFs se guardan en la carpeta `directorio`.
    """
    separador("PASO 1: Generando animaciones de patrones clásicos")
    os.makedirs(directorio, exist_ok=True)

    # Definición de los patrones a animar.
    # Cada tupla: (función_creadora, kwargs, nombre_gif, título)
    patrones = [
        (crear_glider,
         {"filas": 32, "columnas": 32, "offset_fila": 2, "offset_col": 2},
         "glider.gif",
         "Glider (Planeador)"),

        (crear_blinker,
         {"filas": 20, "columnas": 20, "offset_fila": 9, "offset_col": 8},
         "blinker.gif",
         "Blinker (Parpadeante)"),

        (crear_toad,
         {"filas": 20, "columnas": 20, "offset_fila": 8, "offset_col": 7},
         "toad.gif",
         "Toad (Sapo)"),

        (crear_aleatorio,
         {"filas": 64, "columnas": 64, "densidad": 0.25, "semilla": 2024},
         "aleatorio.gif",
         "Estado Aleatorio 64×64"),

        # Patrón aleatorio en grilla grande para mostrar escalabilidad
        (crear_aleatorio,
         {"filas": 128, "columnas": 128, "densidad": 0.20, "semilla": 99},
         "aleatorio_128.gif",
         "Estado Aleatorio 128×128"),
    ]

    for creador, kwargs, nombre_gif, titulo in patrones:
        print(f"\n  → {titulo}")
        juego = creador(**kwargs)

        # Creamos la animación sin mostrarla en pantalla (mostrar=False)
        anim = animar_patron(
            juego,
            pasos=pasos,
            intervalo=intervalo,
            titulo=titulo,
            paleta=PALETA_VERDE,
            mostrar=False,
        )

        # Guardamos como GIF
        ruta_gif = os.path.join(directorio, nombre_gif)
        guardar_animacion(anim, ruta_gif, fps=max(1, 1000 // intervalo))

        # Cerramos la figura para liberar memoria
        import matplotlib.pyplot as plt
        plt.close("all")

    print(f"\n  ✓ Todas las animaciones guardadas en: {directorio}/")


def paso_rendimiento(directorio: str, tamanos: list[int],
                     iteraciones: int, calentamiento: int) -> None:
    """
    Paso 2: mide el rendimiento y genera las gráficas.

    Ejecuta el juego para cada tamaño en `tamanos`, mide el tiempo
    promedio por iteración y guarda las gráficas en `directorio`.
    """
    separador("PASO 2: Midiendo rendimiento empírico")
    os.makedirs(directorio, exist_ok=True)

    # Medición de tiempos
    datos = medir_rendimiento(
        tamanos=tamanos,
        iteraciones=iteraciones,
        calentamiento=calentamiento,
    )

    # Generación de gráficas
    separador("PASO 3: Generando gráficas de rendimiento")
    graficar_rendimiento(datos, directorio_salida=directorio)


def demo_consola() -> None:
    """
    Muestra en la consola el estado del tablero para verificar que
    el juego funciona correctamente sin matplotlib.

    Útil para depuración rápida.
    """
    separador("DEMO: Verificación en consola (Blinker)")
    juego = crear_blinker(filas=7, columnas=9, offset_fila=3, offset_col=3)

    simbolos = {0: "·", 1: "█"}

    for gen in range(4):
        print(f"\n  Generación {gen}:")
        tablero = juego.get_state()
        for fila in tablero:
            print("    " + " ".join(simbolos[c] for c in fila))
        juego.step()

    print("\n  ✓ El Blinker oscila correctamente entre horizontal y vertical.")


def main():
    """Función principal: parsea argumentos y ejecuta los experimentos."""

    # ── Argumentos de línea de comandos ────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Juego de la Vida de Conway — Experimentos"
    )
    parser.add_argument("--solo-animaciones", action="store_true",
                        help="Solo genera las animaciones GIF")
    parser.add_argument("--solo-rendimiento", action="store_true",
                        help="Solo mide el rendimiento y genera gráficas")
    parser.add_argument("--rapido", action="store_true",
                        help="Modo rápido: menos frames y tamaños reducidos")
    parser.add_argument("--sin-grandes", action="store_true",
                        help="Omite los tamaños 512 y 1024 (más lento)")
    args = parser.parse_args()

    # ── Configuración según modo ────────────────────────────────────────────
    if args.rapido:
        pasos_anim   = 30
        intervalo    = 200
        iteraciones  = 20
        calentamiento = 3
        tamanos = [32, 64, 128, 256]
    else:
        pasos_anim   = 60
        intervalo    = 150
        iteraciones  = 60
        calentamiento = 5
        tamanos = [32, 64, 128, 256, 512, 1024]
        if args.sin_grandes:
            tamanos = [32, 64, 128, 256]

    # ── Bienvenida ──────────────────────────────────────────────────────────
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   JUEGO DE LA VIDA DE CONWAY — Experimentos            ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Numba disponible: {'SÍ ✓ (modo acelerado)' if NUMBA_DISPONIBLE else 'NO ✗ (modo NumPy puro)'}")
    print(f"  Tamaños a medir: {tamanos}")
    print(f"  Iteraciones por tamaño: {iteraciones} (warm-up: {calentamiento})")

    t_total_inicio = time.perf_counter()

    # ── Demo de consola (siempre se ejecuta, es instantánea) ───────────────
    demo_consola()

    # ── Paso 1: Animaciones ─────────────────────────────────────────────────
    if not args.solo_rendimiento:
        paso_animaciones(
            directorio="animaciones",
            pasos=pasos_anim,
            intervalo=intervalo,
        )

    # ── Paso 2 y 3: Rendimiento y Gráficas ─────────────────────────────────
    if not args.solo_animaciones:
        paso_rendimiento(
            directorio="figuras",
            tamanos=tamanos,
            iteraciones=iteraciones,
            calentamiento=calentamiento,
        )

    # ── Resumen final ───────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_inicio
    print(f"\n{'═'*60}")
    print(f"  ✓ Experimentos completados en {t_total:.1f} segundos.")
    print(f"  Archivos generados:")
    for carpeta in ["animaciones", "figuras"]:
        if os.path.isdir(carpeta):
            archivos = [f for f in os.listdir(carpeta) if not f.startswith(".")]
            for archivo in sorted(archivos):
                tam = os.path.getsize(os.path.join(carpeta, archivo))
                print(f"    {carpeta}/{archivo}  ({tam/1024:.1f} KB)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
