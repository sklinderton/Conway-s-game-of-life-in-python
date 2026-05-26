# src/visualizacion.py
"""
Módulo de visualización del Juego de la Vida de Conway.

Usa matplotlib.animation.FuncAnimation para generar animaciones de la evolución
del autómata celular. Las animaciones se pueden mostrar en pantalla o guardar
como GIF o MP4.

¿Cómo funciona FuncAnimation?
------------------------------
FuncAnimation recibe:
  - fig  : la figura de matplotlib donde se dibuja.
  - func : una función llamada en cada fotograma. Recibe el número de frame
           como argumento y debe actualizar los artistas de la figura.
  - frames: número total de fotogramas (generaciones a animar).
  - interval: milisegundos entre fotogramas. Menor → animación más rápida.
  - blit: si True, solo redibuja los artistas que cambiaron (más eficiente).

Cada vez que FuncAnimation llama a la función de actualización (update_frame),
nosotros avanzamos el juego una generación y actualizamos la imagen mostrada.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from .juego_vida import (
    GameOfLife,
    crear_glider,
    crear_blinker,
    crear_toad,
    crear_aleatorio,
)


# Paleta de colores: negro para celdas muertas, blanco para vivas.
# También podemos usar colores más vistosos cambiando esta lista.
PALETA_CLASICA = ListedColormap(["#1a1a2e", "#e8f4f8"])   # oscuro / claro
PALETA_VERDE   = ListedColormap(["#0d1117", "#39d353"])   # negro / verde (estilo Matrix)
PALETA_CALIDA  = ListedColormap(["#1a0a00", "#ff6b35"])   # negro / naranja


def animar_patron(juego: GameOfLife,
                  pasos: int = 100,
                  intervalo: int = 100,
                  titulo: str = "Juego de la Vida",
                  paleta=None,
                  mostrar: bool = True) -> animation.FuncAnimation:
    """
    Crea una animación de la evolución de un patrón del Juego de la Vida.

    Parámetros:
    -----------
    juego     : instancia de GameOfLife con el estado inicial cargado.
    pasos     : número de generaciones a animar.
    intervalo : milisegundos entre fotogramas (100 ms → 10 fps).
                Valor menor = animación más rápida.
    titulo    : texto que aparece en el título de la figura.
    paleta    : colormap de matplotlib. Por defecto usa PALETA_CLASICA.
    mostrar   : si True, llama a plt.show() al final.

    Retorna:
    --------
    anim : objeto FuncAnimation (necesario para guardar con guardar_animacion).

    Ejemplo:
    --------
    >>> juego = crear_glider(32, 32)
    >>> anim = animar_patron(juego, pasos=50, mostrar=False)
    >>> guardar_animacion(anim, "animaciones/glider.gif")
    """
    if paleta is None:
        paleta = PALETA_CLASICA

    # Creamos la figura. El tamaño se ajusta para que cada celda sea visible.
    # Para grillas muy grandes (512x512) reducimos el tamaño de figura.
    filas, cols = juego.filas, juego.columnas
    tam = min(8, max(4, 8 * 32 / max(filas, cols)))  # escala el tamaño de figura
    fig, ax = plt.subplots(figsize=(tam, tam))
    fig.patch.set_facecolor("#0d1117")   # fondo de la ventana
    ax.set_facecolor("#0d1117")

    # Configuramos los ejes: sin marcas de graduación, sin bordes.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Título con información de la cuadrícula
    ax.set_title(
        f"{titulo}  ({filas}×{cols})",
        color="white", fontsize=12, pad=10
    )

    # imshow dibuja el estado inicial del tablero.
    # vmin=0, vmax=1 asegura que 0 y 1 siempre mapeen a los extremos de la paleta.
    im = ax.imshow(
        juego.get_state(),
        cmap=paleta,
        vmin=0, vmax=1,
        interpolation="nearest",   # sin suavizado: bordes nítidos entre celdas
        aspect="equal"
    )

    # Texto en la esquina inferior que muestra la generación actual.
    texto_gen = ax.text(
        0.02, 0.02, "Gen: 0",
        transform=ax.transAxes,
        color="white", fontsize=9, va="bottom"
    )

    def update_frame(frame_num):
        """
        Función llamada por FuncAnimation en cada fotograma.

        Avanza el juego una generación y actualiza la imagen.
        Devuelve una lista de los artistas modificados (requerido por blit=True).

        frame_num : entero entre 0 y pasos-1 (lo pasa FuncAnimation automáticamente).
        """
        juego.step()                        # avanzamos una generación
        im.set_data(juego.get_state())      # actualizamos los datos de la imagen
        texto_gen.set_text(f"Gen: {juego.generacion}")
        return [im, texto_gen]              # artistas que cambiaron (para blit)

    # Creamos la animación.
    # - frames=pasos: llamará a update_frame pasos veces.
    # - interval=intervalo: esperará `intervalo` ms entre fotogramas.
    # - blit=True: solo redibuja lo que cambió (más eficiente en pantalla).
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=pasos,
        interval=intervalo,
        blit=True,
        repeat=True   # la animación se repite al terminar
    )

    plt.tight_layout()

    if mostrar:
        plt.show()

    return anim


def guardar_animacion(anim: animation.FuncAnimation,
                      ruta: str,
                      fps: int = 10,
                      dpi: int = 80) -> None:
    """
    Guarda una animación como archivo GIF o MP4.

    El formato se detecta automáticamente por la extensión del archivo:
    - .gif  → usa el escritor PillowWriter (requiere Pillow).
    - .mp4  → usa FFMpegWriter (requiere ffmpeg instalado en el sistema).

    Parámetros:
    -----------
    anim : objeto FuncAnimation generado por animar_patron().
    ruta : ruta del archivo de salida (ej: "animaciones/glider.gif").
    fps  : fotogramas por segundo del archivo guardado.
    dpi  : resolución en puntos por pulgada.

    Ejemplo:
    --------
    >>> guardar_animacion(anim, "animaciones/glider.gif", fps=10)
    """
    ruta_str = str(ruta)
    print(f"  Guardando animación en: {ruta_str} ...", end=" ", flush=True)

    if ruta_str.endswith(".gif"):
        # PillowWriter: usa la librería Pillow para escribir GIFs cuadro a cuadro.
        escritor = animation.PillowWriter(fps=fps)
    elif ruta_str.endswith(".mp4"):
        # FFMpegWriter: usa ffmpeg para codificar video H.264.
        escritor = animation.FFMpegWriter(fps=fps, bitrate=800)
    else:
        raise ValueError(f"Formato no soportado: {ruta_str}. Usa .gif o .mp4")

    anim.save(ruta_str, writer=escritor, dpi=dpi)
    print("✓")


def crear_todas_las_animaciones(directorio_salida: str = "animaciones",
                                pasos: int = 60,
                                intervalo: int = 150) -> None:
    """
    Genera y guarda animaciones de los cuatro patrones clásicos.

    Patrones generados:
    - Glider (planeador): se desplaza diagonalmente.
    - Blinker (parpadeante): oscila entre horizontal y vertical.
    - Toad (sapo): oscilador de período 2 con dos triángulos.
    - Aleatorio: tablero aleatorio 64×64.

    Parámetros:
    -----------
    directorio_salida : carpeta donde se guardan los GIFs.
    pasos             : generaciones a animar por patrón.
    intervalo         : ms entre fotogramas.
    """
    import os
    os.makedirs(directorio_salida, exist_ok=True)

    # Lista de (función_creadora, kwargs, nombre_archivo, título)
    patrones = [
        (crear_glider,   {"filas": 32, "columnas": 32},             "glider.gif",   "Glider (Planeador)"),
        (crear_blinker,  {"filas": 16, "columnas": 16},             "blinker.gif",  "Blinker (Parpadeante)"),
        (crear_toad,     {"filas": 16, "columnas": 16},             "toad.gif",     "Toad (Sapo)"),
        (crear_aleatorio,{"filas": 64, "columnas": 64, "semilla": 42}, "aleatorio.gif","Estado Aleatorio 64×64"),
    ]

    for creador, kwargs, nombre, titulo in patrones:
        print(f"\n→ Animando: {titulo}")
        juego = creador(**kwargs)

        # mostrar=False para no abrir ventanas durante la generación por lotes.
        anim = animar_patron(
            juego,
            pasos=pasos,
            intervalo=intervalo,
            titulo=titulo,
            paleta=PALETA_VERDE,
            mostrar=False
        )

        ruta = f"{directorio_salida}/{nombre}"
        guardar_animacion(anim, ruta, fps=max(1, 1000 // intervalo))

        # Cerramos la figura para liberar memoria
        plt.close("all")


def visualizar_estado(juego: GameOfLife,
                      titulo: str = "Estado actual",
                      paleta=None) -> None:
    """
    Muestra una imagen estática del estado actual del juego.

    Útil para depuración o para ver el estado inicial/final sin animar.

    Parámetros:
    -----------
    juego  : instancia de GameOfLife.
    titulo : título de la figura.
    paleta : colormap de matplotlib.
    """
    if paleta is None:
        paleta = PALETA_CLASICA

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(juego.get_state(), cmap=paleta, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{titulo} — Gen {juego.generacion} — Vivas: {juego.contar_vivas()}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
