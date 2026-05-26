# src/__init__.py
# Paquete principal del proyecto Juego de la Vida de Conway
# Exporta las clases y funciones más importantes para uso externo

from .juego_vida import GameOfLife
from .visualizacion import animar_patron, guardar_animacion
from .rendimiento import medir_rendimiento, graficar_rendimiento

__all__ = [
    "GameOfLife",
    "animar_patron",
    "guardar_animacion",
    "medir_rendimiento",
    "graficar_rendimiento",
]
