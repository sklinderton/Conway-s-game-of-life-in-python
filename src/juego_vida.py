# src/juego_vida.py
"""
Implementación del Juego de la Vida de Conway. Por Jason Jesus Barrantes Sanchez

Este módulo contiene la clase GameOfLife y la función núcleo acelerada con Numba
que calcula la siguiente generación del autómata celular.

¿Qué es Numba?
--------------
Numba es un compilador JIT (Just-In-Time) para Python. Cuando decoramos una función
con @njit, Numba la compila a código máquina nativo la primera vez que se llama.
Las llamadas siguientes usan ese código compilado, que es tan rápido como C o Fortran.

¿Por qué acelera el código?
---------------------------
Python normal es lento porque cada operación pasa por el intérprete. Con Numba,
los bucles anidados sobre arreglos NumPy se ejecutan directamente en la CPU, sin
overhead del intérprete.

Limitaciones de @njit:
-----------------------
- No se pueden usar objetos arbitrarios de Python (listas de listas, dicts, clases).
- No se puede llamar a funciones de Python que no sean compatibles con Numba.
- La primera llamada tiene latencia por la compilación JIT (warm-up).
- No soporta todas las funciones de NumPy (aunque soporta la mayoría de las básicas).
"""

import numpy as np

# Intentamos importar Numba. Si no está instalada, usamos una versión pura de Python.
try:
    from numba import njit
    NUMBA_DISPONIBLE = True
except ImportError:
    # Si Numba no está disponible, creamos un decorador "falso" que no hace nada.
    def njit(*args, **kwargs):
        def decorador(func):
            return func
        return decorador
    NUMBA_DISPONIBLE = False
    print("⚠  Numba no encontrada. Se usará la versión pura de NumPy (más lenta).")


# =============================================================================
# FUNCIÓN NÚCLEO: calcula la siguiente generación
# =============================================================================
# Esta función se define FUERA de la clase porque @njit no puede compilar métodos
# de clase (Numba no entiende `self`). La clase la llama como una función normal.
#
# Usamos bordes TOROIDALES (el tablero "se envuelve" como un donut):
#   - La celda de la fila 0 tiene como vecino superior a la última fila.
#   - La celda de la columna 0 tiene como vecino izquierdo a la última columna.
# Esto evita tener que manejar casos especiales en los bordes y es más elegante
# matemáticamente. Se implementa con el operador módulo (%).

@njit(cache=True)
def _calcular_siguiente_generacion(tablero, filas, columnas):
    """
    Calcula el siguiente estado del tablero aplicando las reglas de Conway.

    Parámetros:
    -----------
    tablero  : np.ndarray de uint8, forma (filas, columnas)
               Estado actual. 1 = viva, 0 = muerta.
    filas    : int, número de filas de la cuadrícula.
    columnas : int, número de columnas de la cuadrícula.

    Retorna:
    --------
    nuevo_tablero : np.ndarray de uint8 con el estado siguiente.

    Complejidad: O(filas × columnas) = O(n), donde n es el número total de celdas.
    Para cada celda hacemos exactamente 8 consultas a vecinos → trabajo constante por celda.
    """
    # Creamos un nuevo tablero vacío para guardar la siguiente generación.
    # Es CRUCIAL no modificar el tablero original mientras calculamos, porque
    # todas las celdas deben actualizarse "simultáneamente".
    nuevo_tablero = np.zeros((filas, columnas), dtype=np.uint8)

    # Recorremos cada celda de la cuadrícula con dos bucles anidados.
    # Gracias a @njit, estos bucles se compilan a código máquina nativo:
    # son equivalentes en velocidad a un bucle en C.
    for i in range(filas):
        for j in range(columnas):

            # Contamos los vecinos vivos de la celda (i, j).
            # Usamos el operador % para los bordes toroidales:
            #   (i - 1) % filas  → si i=0, da filas-1 (borde superior → inferior)
            #   (i + 1) % filas  → si i=filas-1, da 0 (borde inferior → superior)
            vecinos_vivos = (
                tablero[(i - 1) % filas, (j - 1) % columnas] +  # arriba-izquierda
                tablero[(i - 1) % filas,  j               ] +   # arriba
                tablero[(i - 1) % filas, (j + 1) % columnas] +  # arriba-derecha
                tablero[ i,              (j - 1) % columnas] +  # izquierda
                tablero[ i,              (j + 1) % columnas] +  # derecha
                tablero[(i + 1) % filas, (j - 1) % columnas] +  # abajo-izquierda
                tablero[(i + 1) % filas,  j               ] +   # abajo
                tablero[(i + 1) % filas, (j + 1) % columnas]    # abajo-derecha
            )

            # Aplicamos las 4 reglas de Conway:
            celda_actual = tablero[i, j]

            if celda_actual == 1:
                # Regla 2 (Soledad): muere con < 2 vecinos.
                # Regla 1 (Superpoblación): muere con > 3 vecinos.
                # Regla 3 (Supervivencia): sobrevive con 2 o 3 vecinos.
                if vecinos_vivos == 2 or vecinos_vivos == 3:
                    nuevo_tablero[i, j] = 1  # sobrevive
                # else: queda en 0 (muere)
            else:
                # Regla 4 (Reproducción): nace con exactamente 3 vecinos.
                if vecinos_vivos == 3:
                    nuevo_tablero[i, j] = 1  # nace
                # else: queda en 0 (sigue muerta)

    return nuevo_tablero


# =============================================================================
# CLASE PRINCIPAL: GameOfLife
# =============================================================================

class GameOfLife:
    """
    Clase que encapsula el estado y la lógica del Juego de la Vida de Conway.

    El tablero se representa como un arreglo NumPy de tipo uint8 (enteros sin signo
    de 8 bits). Usamos uint8 en lugar de bool porque Numba suma los valores directamente
    para contar vecinos (True+True = 2, que funciona, pero uint8 es más explícito).

    Condiciones de frontera: TOROIDALES (el tablero se envuelve en ambos ejes).
    Esto significa que las celdas del borde tienen vecinos en el lado opuesto,
    como si la cuadrícula fuera la superficie de un donut (toro).

    Ejemplo de uso:
    ---------------
    >>> juego = GameOfLife(32, 32)
    >>> juego.run(100)
    >>> estado = juego.get_state()
    """

    def __init__(self, filas, columnas, estado_inicial=None):
        """
        Inicializa el tablero del Juego de la Vida.

        Parámetros:
        -----------
        filas         : int, número de filas de la cuadrícula.
        columnas      : int, número de columnas de la cuadrícula.
        estado_inicial: np.ndarray opcional de forma (filas, columnas).
                        Si es None, se genera un estado aleatorio con ~20% de celdas vivas.

        Ejemplo:
        --------
        >>> juego = GameOfLife(10, 10)                          # aleatorio
        >>> juego = GameOfLife(10, 10, np.zeros((10,10)))      # todo muerto
        """
        self.filas = filas
        self.columnas = columnas
        self.generacion = 0  # contador de generaciones transcurridas

        if estado_inicial is None:
            # Estado aleatorio: cada celda tiene 20% de probabilidad de estar viva.
            # np.random.choice([0, 1], size=..., p=[0.8, 0.2]) es equivalente pero
            # random.random() < 0.2 es más rápido para arreglos grandes.
            rng = np.random.default_rng()  # generador de números aleatorios moderno
            self.tablero = rng.choice(
                np.array([0, 1], dtype=np.uint8),
                size=(filas, columnas),
                p=[0.8, 0.2]
            )
        else:
            # Usamos el estado provisto, asegurándonos del tipo correcto.
            self.tablero = np.array(estado_inicial, dtype=np.uint8)
            # Verificamos que las dimensiones coincidan.
            assert self.tablero.shape == (filas, columnas), (
                f"El estado inicial debe tener forma ({filas}, {columnas}), "
                f"pero se recibió {self.tablero.shape}"
            )

    def step(self):
        """
        Avanza el tablero una generación aplicando las reglas de Conway.

        Delega el cálculo pesado a la función _calcular_siguiente_generacion,
        que está acelerada con Numba (@njit). De esta forma, la clase puede
        usar orientación a objetos (self, métodos) mientras el núcleo de cálculo
        se beneficia de la compilación JIT.

        Modifica self.tablero en el lugar (in-place) y actualiza self.generacion.
        """
        # Llamamos a la función Numba con los datos del tablero.
        # Pasamos filas y columnas como parámetros separados porque @njit
        # puede acceder al atributo .shape del arreglo, pero pasarlos
        # explícitamente es más claro y ligeramente más eficiente.
        self.tablero = _calcular_siguiente_generacion(
            self.tablero, self.filas, self.columnas
        )
        self.generacion += 1

    def run(self, pasos):
        """
        Ejecuta múltiples generaciones del juego.

        Parámetros:
        -----------
        pasos : int, número de generaciones a ejecutar.

        Ejemplo:
        --------
        >>> juego = GameOfLife(64, 64)
        >>> juego.run(200)  # avanza 200 generaciones
        """
        for _ in range(pasos):
            self.step()

    def get_state(self):
        """
        Devuelve una COPIA del tablero actual como arreglo NumPy.

        Retorna una copia (no una referencia) para evitar que el código externo
        modifique accidentalmente el estado interno del juego.

        Retorna:
        --------
        np.ndarray de uint8, forma (filas, columnas).
        """
        return self.tablero.copy()

    def reset(self, estado_inicial=None):
        """
        Reinicia el juego con un nuevo estado inicial.

        Útil para reutilizar el mismo objeto GameOfLife con diferentes
        configuraciones sin crear una instancia nueva.

        Parámetros:
        -----------
        estado_inicial: np.ndarray opcional. Si None, genera estado aleatorio.
        """
        self.__init__(self.filas, self.columnas, estado_inicial)

    def contar_vivas(self):
        """
        Cuenta cuántas celdas están vivas en el estado actual.

        Retorna:
        --------
        int, número de celdas vivas.
        """
        # np.sum() sobre un arreglo uint8 suma los 1s (vivas) y 0s (muertas).
        return int(np.sum(self.tablero))

    def __repr__(self):
        """Representación de texto del objeto para depuración."""
        return (
            f"GameOfLife({self.filas}x{self.columnas}, "
            f"gen={self.generacion}, "
            f"vivas={self.contar_vivas()})"
        )


# =============================================================================
# PATRONES CLÁSICOS (funciones de fábrica)
# =============================================================================
# Estas funciones crean tableros con patrones conocidos del Juego de la Vida.
# Son útiles para verificar que la implementación es correcta.

def crear_glider(filas=32, columnas=32, offset_fila=1, offset_col=1):
    """
    Crea un Glider (planeador) en un tablero vacío.

    El Glider es el patrón más famoso del Juego de la Vida. Se mueve
    diagonalmente en la cuadrícula, completando un ciclo cada 4 generaciones
    y desplazándose 1 celda en diagonal.

    Patrón (coordenadas relativas al offset):
        . X .
        . . X
        X X X

    Parámetros:
    -----------
    filas, columnas : dimensiones del tablero.
    offset_fila, offset_col : posición donde colocar el glider.
    """
    tablero = np.zeros((filas, columnas), dtype=np.uint8)
    # Definimos el patrón como lista de (fila, columna) de celdas vivas
    patron = [
        (0, 1),
        (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    for df, dc in patron:
        tablero[(offset_fila + df) % filas, (offset_col + dc) % columnas] = 1
    return GameOfLife(filas, columnas, tablero)


def crear_blinker(filas=16, columnas=16, offset_fila=7, offset_col=6):
    """
    Crea un Blinker (parpadeante) en un tablero vacío.

    El Blinker es el oscilador más simple: alterna entre una línea horizontal
    y una línea vertical cada generación (período 2).

    Patrón inicial:
        X X X
    """
    tablero = np.zeros((filas, columnas), dtype=np.uint8)
    patron = [(0, 0), (0, 1), (0, 2)]
    for df, dc in patron:
        tablero[(offset_fila + df) % filas, (offset_col + dc) % columnas] = 1
    return GameOfLife(filas, columnas, tablero)


def crear_toad(filas=16, columnas=16, offset_fila=6, offset_col=5):
    """
    Crea un Toad (sapo) en un tablero vacío.

    El Toad es un oscilador de período 2 que consiste en dos triángulos
    que se alternan.

    Patrón inicial:
        . X X X
        X X X .
    """
    tablero = np.zeros((filas, columnas), dtype=np.uint8)
    patron = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2),
    ]
    for df, dc in patron:
        tablero[(offset_fila + df) % filas, (offset_col + dc) % columnas] = 1
    return GameOfLife(filas, columnas, tablero)


def crear_aleatorio(filas=64, columnas=64, densidad=0.2, semilla=None):
    """
    Crea un tablero con estado inicial aleatorio.

    Parámetros:
    -----------
    filas, columnas : dimensiones del tablero.
    densidad        : fracción de celdas vivas (0.0 a 1.0).
    semilla         : semilla para reproducibilidad.
    """
    rng = np.random.default_rng(semilla)
    tablero = (rng.random((filas, columnas)) < densidad).astype(np.uint8)
    return GameOfLife(filas, columnas, tablero)
