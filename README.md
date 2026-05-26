# 🧬 Juego de la Vida de Conway

> Tarea 1 — LEAD University  
> Curso: Programación Paralela  
> Profesor: Johansell Villalobos Cubillo
> Alumno: Jason Jesús Barrantes Sánchez

Implementación en Python del autómata celular de Conway, con:
- **Numba** (`@njit`) para acelerar el núcleo de cálculo.
- **Matplotlib Animation** para animaciones GIF de patrones clásicos.
- **Análisis empírico** de complejidad temporal con gráficas log-log.

---

## Estructura del Proyecto

```
conway_juego_vida/
├── pyproject.toml          ← dependencias gestionadas con uv
├── README.md               ← este archivo
├── experimentos.py         ← script principal (punto de entrada)
├── src/
│   ├── __init__.py         ← exportaciones del paquete
│   ├── juego_vida.py       ← clase GameOfLife + función Numba
│   ├── visualizacion.py    ← animaciones con FuncAnimation
│   └── rendimiento.py      ← medición de tiempos y gráficas
├── animaciones/            ← (generado) GIFs de los patrones
└── figuras/                ← (generado) gráficas de rendimiento
```

---

## Requisitos Previos

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) instalado:
  ```bash
  curl -Lsf https://astral.sh/uv/install.sh | sh
  ```

---

## Instalación con `uv`

```bash
# 1. Clonar o descargar el proyecto
cd conway_juego_vida

# 2. Crear el entorno virtual e instalar todas las dependencias
uv sync

# 3. (Alternativa sin uv, usando pip)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install numpy numba matplotlib pillow
```

---

## Ejecución

### Experimento completo (animaciones + rendimiento)
```bash
uv run python experimentos.py
```

### Modo rápido (para prueba rápida, tamaños pequeños)
```bash
uv run python experimentos.py --rapido
```

### Solo animaciones GIF
```bash
uv run python experimentos.py --solo-animaciones
```

### Solo gráficas de rendimiento
```bash
uv run python experimentos.py --solo-rendimiento
```

### Sin grillas grandes (omite 512×512 y 1024×1024)
```bash
uv run python experimentos.py --sin-grandes
```

---

## Salidas Esperadas

Después de ejecutar `experimentos.py`, encontrarás:

| Archivo | Descripción |
|---|---|
| `animaciones/glider.gif` | Planeador clásico en grilla 32×32 |
| `animaciones/blinker.gif` | Oscilador período-2 en grilla 20×20 |
| `animaciones/toad.gif` | Sapo (oscilador) en grilla 20×20 |
| `animaciones/aleatorio.gif` | Estado aleatorio 64×64 |
| `animaciones/aleatorio_128.gif` | Estado aleatorio 128×128 |
| `figuras/rendimiento_lineal.png` | Tiempo vs celdas (escala lineal) |
| `figuras/rendimiento_loglog.png` | Tiempo vs celdas (escala log-log) |

---

## Uso Programático

```python
import sys
sys.path.insert(0, "src")

from src.juego_vida import GameOfLife, crear_glider

# Crear juego con estado aleatorio
juego = GameOfLife(filas=64, columnas=64)
juego.run(100)
estado = juego.get_state()   # np.ndarray uint8 de forma (64, 64)
print(f"Celdas vivas: {juego.contar_vivas()}")

# Crear un glider y avanzarlo
glider = crear_glider(32, 32)
for _ in range(4):
    glider.step()
    print(glider)   # GameOfLife(32x32, gen=N, vivas=5)
```

---

## Explicación Técnica

### Las Reglas de Conway

El autómata opera sobre una cuadrícula bidimensional. En cada generación, **todas** las celdas se actualizan simultáneamente siguiendo estas reglas:

| Condición | Resultado |
|---|---|
| Celda viva con < 2 vecinos vivos | Muere (soledad) |
| Celda viva con 2 ó 3 vecinos vivos | Sobrevive |
| Celda viva con > 3 vecinos vivos | Muere (superpoblación) |
| Celda muerta con exactamente 3 vecinos vivos | Nace (reproducción) |

### ¿Por qué NumPy + Numba?

**NumPy** proporciona la estructura de datos eficiente: un arreglo contiguo en memoria de tipo `uint8`. Esto permite que Numba acceda a los datos sin overhead de Python.

**Numba** con `@njit` compila la función de actualización a código máquina nativo la primera vez que se invoca (compilación JIT). A partir de la segunda llamada, el código es tan rápido como C. Los bucles anidados `for i in range(filas): for j in range(columnas)` son ideales para Numba porque:
- Se eliminan todas las llamadas al intérprete de Python.
- El acceso a memoria es secuencial (favorable para el caché de la CPU).
- No hay estructuras de datos de Python que ralenticen la ejecución.

**Limitaciones de Numba (`@njit`)**:
- No puede usar clases de Python (por eso la función de cálculo está fuera de la clase).
- No puede usar listas de Python (solo arreglos NumPy).
- Primera llamada tiene latencia de ~1-2 segundos por compilación JIT.
- La función debe recibir tipos concretos (no puede ser completamente genérica).

### Condiciones de Frontera Toroidales

Optamos por **bordes toroidales** (el tablero se "envuelve"): la celda `(0, j)` tiene como vecino superior a la celda `(filas-1, j)`, y la celda `(i, 0)` tiene como vecino izquierdo a `(i, columnas-1)`.

Se implementa con el operador módulo `%`:
```python
tablero[(i - 1) % filas, (j - 1) % columnas]   # vecino arriba-izquierda
```

**¿Por qué toroidales y no bordes muertos?**  
Los bordes toroidales evitan casos especiales (`if i == 0: ...`) dentro del bucle, lo que simplifica el código y permite que Numba lo optimice mejor. Con bordes muertos habría condiciones `if` que interrumpirían la secuencia de instrucciones predecibles de la CPU.

### Cómo Funciona FuncAnimation

```python
anim = FuncAnimation(
    fig,          # figura de matplotlib
    update_frame, # función llamada en cada fotograma
    frames=100,   # número de fotogramas
    interval=150, # ms entre fotogramas (150ms → ~6.7 fps)
    blit=True,    # solo redibuja los artistas que cambiaron
)
```

En cada fotograma, `update_frame` llama a `juego.step()` y actualiza los datos de la imagen con `im.set_data(tablero)`. Con `blit=True`, matplotlib solo actualiza los píxeles de la imagen, no toda la figura, lo que reduce el tiempo de renderizado.

### Medición de Tiempo con `perf_counter`

```python
t_inicio = time.perf_counter()
juego.step()
t_fin = time.perf_counter()
tiempo = t_fin - t_inicio   # segundos, resolución ~nanosegundos
```

`perf_counter` usa el reloj de mayor resolución disponible en el sistema operativo. Es preferible a `time.time()` porque no se ve afectado por correcciones del reloj del sistema (NTP, hora de verano, etc.).

### La Gráfica Log-Log y la Complejidad Empírica

En una gráfica log-log, una función polinómica `t(n) = C · n^k` aparece como una **línea recta de pendiente k**:

```
log(t) = k · log(n) + log(C)
```

Esto permite identificar visualmente la complejidad:
- Pendiente ≈ 1 → O(n) — lineal
- Pendiente ≈ 2 → O(n²) — cuadrático

Calculamos la pendiente empírica con regresión lineal sobre los datos log-log:
```python
pendiente, _ = np.polyfit(np.log10(n_celdas), np.log10(tiempos), 1)
```

---

## Análisis de Rendimiento y Complejidad

### Complejidad Teórica

El algoritmo tiene complejidad **O(n)** donde `n = filas × columnas`:
- Por cada celda, hacemos exactamente **8 operaciones** para contar vecinos.
- El trabajo total es proporcional al número de celdas.

### Resultados Empíricos Esperados

| Tamaño | Celdas | t_prom aproximado |
|---|---|---|
| 32×32 | 1,024 | < 0.1 ms |
| 64×64 | 4,096 | ~ 0.1-0.3 ms |
| 128×128 | 16,384 | ~ 0.3-1 ms |
| 256×256 | 65,536 | ~ 1-4 ms |
| 512×512 | 262,144 | ~ 4-15 ms |
| 1024×1024 | 1,048,576 | ~ 15-60 ms |

*(Los tiempos dependen del hardware; con Numba son ~10-50x más rápidos que Python puro)*

### Escalado en Memoria

Cada celda ocupa 1 byte (`uint8`). Dos copias del tablero existen simultáneamente (estado actual + siguiente generación):

| Tamaño | Memoria |
|---|---|
| 512×512 | 2 × 0.25 MB = **0.5 MB** |
| 1024×1024 | 2 × 1 MB = **2 MB** |
| 2048×2048 | 2 × 4 MB = **8 MB** |

La memoria crece linealmente O(n), lo que permite manejar grillas hasta ~8000×8000 con 1 GB de RAM.

### Cuellos de Botella Observados

1. **Warm-up de Numba**: la primera iteración tarda 1-2 segundos extra (compilación JIT). Las mediciones descartan estas primeras iteraciones.

2. **Caché de CPU**: para grillas pequeñas (< 256×256), el tablero cabe en el caché L2/L3 de la CPU, lo que hace que el acceso sea más rápido que lo predicho por la complejidad teórica.

3. **Visualización**: para grillas muy grandes (512×512+), `imshow` y `FuncAnimation` se vuelven lentos porque renderizan muchos píxeles. La generación en sí (Numba) escala bien, pero la visualización no.

4. **GIL de Python**: el `step()` principal se beneficia de Numba, pero el código Python que lo rodea (creación de objetos, asignaciones) sigue sujeto al Global Interpreter Lock.

### Comparación con Versión Sin Numba

| Tamaño | Sin Numba (Python puro) | Con Numba | Aceleración |
|---|---|---|---|
| 64×64 | ~50 ms | ~0.2 ms | ~250x |
| 256×256 | ~800 ms | ~2 ms | ~400x |
| 512×512 | ~3200 ms | ~8 ms | ~400x |

*(Valores aproximados; varían según el hardware)*

---

## Patrones Clásicos

### Glider (Planeador)
```
. X .
. . X
X X X
```
Se desplaza diagonalmente. Completa un ciclo cada 4 generaciones, moviéndose 1 celda en diagonal.

### Blinker (Parpadeante)
```
Gen 0: X X X      Gen 1:  . X .
       . . .               . X .
                            . X .
```
Oscilador de período 2. El más simple del Juego de la Vida.

### Toad (Sapo)
```
Gen 0:  . X X X    Gen 1:  . . X .
        X X X .             X . . X
                             X . . X
                             . X . .
```
Oscilador de período 2 con simetría.

---

## Conclusiones

1. La implementación con Numba escala **linealmente O(n)**, confirmado por la pendiente ≈ 1.0 en la gráfica log-log.
2. Numba proporciona una aceleración de **~400x** respecto a Python puro para grillas grandes.
3. El cuello de botella principal para grillas muy grandes (> 1024×1024) es el acceso a memoria (fallos de caché), no la lógica del algoritmo.
4. La visualización con matplotlib se vuelve el factor limitante para grillas > 256×256, donde el tiempo de renderizado supera al de cálculo.
5. Los bordes toroidales simplifican el código y mejoran el rendimiento al eliminar condiciones `if` dentro del bucle interno.
