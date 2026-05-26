# src/rendimiento.py
"""
Módulo de medición de rendimiento y análisis de complejidad empírica.

Este módulo mide cuánto tarda cada iteración del Juego de la Vida para distintos
tamaños de cuadrícula, y genera gráficas que comparan el rendimiento empírico con
curvas teóricas de complejidad.

¿Cómo se mide el tiempo?
-------------------------
Usamos time.perf_counter(), que es el reloj de alta resolución de Python:
  - Más preciso que time.time() (resolución de nanosegundos en la mayoría de sistemas).
  - No se ve afectado por cambios en el reloj del sistema.
  - Mide tiempo de pared (wall-clock), no tiempo de CPU.

Patrón de medición:
    t_inicio = time.perf_counter()
    juego.step()
    t_fin = time.perf_counter()
    tiempo = t_fin - t_inicio

Descartamos las primeras 5 iteraciones porque Numba las usa para compilar el
código JIT (warm-up). A partir de la iteración 6, el código ya está compilado
y los tiempos son representativos del rendimiento real.

¿Por qué esperamos escalado O(n)?
----------------------------------
El algoritmo revisa cada celda exactamente una vez y hace 8 operaciones por celda
(contar vecinos). Esto da O(n) donde n = filas × columnas.
En la gráfica log-log, una función O(n) aparece como una línea recta de pendiente 1.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from .juego_vida import GameOfLife


def medir_rendimiento(tamanos: list[int] | None = None,
                      iteraciones: int = 60,
                      calentamiento: int = 5) -> dict:
    """
    Mide el tiempo promedio por iteración para distintos tamaños de cuadrícula.

    Para cada tamaño N, crea una cuadrícula N×N y mide el tiempo de `iteraciones`
    pasos del juego, descartando los primeros `calentamiento` para evitar el
    overhead de compilación JIT de Numba.

    Parámetros:
    -----------
    tamanos      : lista de tamaños de cuadrícula (lado). Por defecto:
                   [32, 64, 128, 256, 512, 1024].
    iteraciones  : total de iteraciones a ejecutar por tamaño.
    calentamiento: iteraciones iniciales descartadas (warm-up de Numba).

    Retorna:
    --------
    dict con las claves:
      - "tamanos"   : lista de tamaños de lado.
      - "n_celdas"  : lista de número de celdas (lado²).
      - "tiempos"   : lista de tiempos promedio por iteración (segundos).
      - "desv_std"  : lista de desviaciones estándar de los tiempos.
    """
    if tamanos is None:
        tamanos = [32, 64, 128, 256, 512, 1024]

    n_celdas_lista = []
    tiempos_lista  = []
    desv_std_lista = []

    print("\n" + "="*60)
    print("  MEDICIÓN DE RENDIMIENTO")
    print("="*60)
    print(f"  {'Tamaño':>10}  {'Celdas':>12}  {'t_prom (ms)':>14}  {'σ (ms)':>10}")
    print("-"*60)

    for lado in tamanos:
        n_celdas = lado * lado

        # Creamos el juego con estado aleatorio para este tamaño.
        juego = GameOfLife(lado, lado)

        # --- Fase de calentamiento (warm-up) ---
        # Las primeras `calentamiento` iteraciones activan la compilación JIT de Numba.
        # No las medimos porque incluirían el overhead de compilación, no el de ejecución.
        for _ in range(calentamiento):
            juego.step()

        # --- Fase de medición ---
        # Medimos cada iteración individualmente y guardamos todos los tiempos.
        tiempos_individuales = []
        for _ in range(iteraciones - calentamiento):
            t_inicio = time.perf_counter()   # ← reloj de alta resolución
            juego.step()
            t_fin = time.perf_counter()
            tiempos_individuales.append(t_fin - t_inicio)

        # Calculamos estadísticas
        t_prom = np.mean(tiempos_individuales)
        t_std  = np.std(tiempos_individuales)

        n_celdas_lista.append(n_celdas)
        tiempos_lista.append(t_prom)
        desv_std_lista.append(t_std)

        # Mostramos resultado en consola (convertido a milisegundos para legibilidad)
        print(f"  {lado:>8}×{lado:<4}  {n_celdas:>12,}  "
              f"{t_prom*1000:>12.4f}ms  {t_std*1000:>8.4f}ms")

    print("="*60)

    return {
        "tamanos"  : tamanos,
        "n_celdas" : n_celdas_lista,
        "tiempos"  : tiempos_lista,
        "desv_std" : desv_std_lista,
    }


def graficar_rendimiento(datos: dict,
                         directorio_salida: str = "figuras") -> None:
    """
    Genera dos gráficas de rendimiento y las guarda en disco.

    Gráfica 1: Escala lineal — Tiempo (ms) vs Número de celdas.
    Gráfica 2: Escala log-log — log(Tiempo) vs log(Celdas).

    Sobre ambas gráficas se superponen curvas teóricas de complejidad
    O(n), O(n log n) y O(n²), normalizadas para que coincidan con los
    datos en el punto medio.

    ¿Cómo se normalizan las curvas teóricas?
    -----------------------------------------
    Dada una función f(n) (ej: f(n) = n), la normalizamos para que pase
    por el punto medio de los datos:
        factor = t_medio / f(n_medio)
        curva  = factor * f(n)
    Así podemos comparar la FORMA de la curva (pendiente) con los datos,
    independientemente de las constantes multiplicativas.

    ¿Cómo interpretar la gráfica log-log?
    ----------------------------------------
    En una gráfica log-log, una función O(n^k) aparece como una línea recta
    con pendiente k:
        log(t) = k·log(n) + constante
    - Pendiente 1 → O(n) (lineal) ← lo que esperamos
    - Pendiente 2 → O(n²) (cuadrático)
    Si los datos siguen una línea recta de pendiente ≈1, la implementación escala linealmente.

    Parámetros:
    -----------
    datos            : dict retornado por medir_rendimiento().
    directorio_salida: carpeta donde guardar las gráficas.
    """
    os.makedirs(directorio_salida, exist_ok=True)

    n_celdas = np.array(datos["n_celdas"])
    tiempos  = np.array(datos["tiempos"]) * 1000   # convertimos a milisegundos
    desv_std = np.array(datos["desv_std"]) * 1000

    # -------------------------------------------------------------------------
    # Normalizamos las curvas teóricas en el punto MEDIO de los datos
    # -------------------------------------------------------------------------
    idx_medio  = len(n_celdas) // 2
    n_medio    = n_celdas[idx_medio]
    t_medio    = tiempos[idx_medio]

    # Definimos las funciones de complejidad (sin normalizar)
    # n_fino: arreglo denso de valores de n para trazar curvas suaves
    n_fino = np.linspace(n_celdas[0], n_celdas[-1], 500)

    def normalizar(func_vals, n_ref, t_ref):
        """Escala `func_vals` para que coincida en el punto de referencia."""
        f_ref = func_vals[np.searchsorted(n_fino, n_ref)]
        return func_vals * (t_ref / f_ref)

    curva_on     = normalizar(n_fino,                           n_medio, t_medio)
    curva_onlogn = normalizar(n_fino * np.log2(n_fino + 1),    n_medio, t_medio)
    curva_on2    = normalizar(n_fino**2,                        n_medio, t_medio)

    # =========================================================================
    # GRÁFICA 1: Escala LINEAL
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    # Datos empíricos con barras de error (±1 desviación estándar)
    ax1.errorbar(
        n_celdas, tiempos, yerr=desv_std,
        fmt="o-", color="#4c9be8", linewidth=2, markersize=7,
        capsize=4, label="Tiempo medido (±σ)", zorder=5
    )

    # Curvas teóricas
    ax1.plot(n_fino, curva_on,     "--", color="#2ecc71", linewidth=1.5, alpha=0.8, label="O(n) lineal")
    ax1.plot(n_fino, curva_onlogn, "--", color="#f39c12", linewidth=1.5, alpha=0.8, label="O(n log n)")
    ax1.plot(n_fino, curva_on2,    "--", color="#e74c3c", linewidth=1.5, alpha=0.8, label="O(n²)")

    ax1.set_xlabel("Número de celdas (filas × columnas)", fontsize=12)
    ax1.set_ylabel("Tiempo promedio por iteración (ms)", fontsize=12)
    ax1.set_title("Rendimiento del Juego de la Vida — Escala Lineal", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Formateamos el eje x con comas para miles
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    ruta1 = f"{directorio_salida}/rendimiento_lineal.png"
    fig1.savefig(ruta1, dpi=150, bbox_inches="tight")
    print(f"  Gráfica lineal guardada: {ruta1}")
    plt.close(fig1)

    # =========================================================================
    # GRÁFICA 2: Escala LOG-LOG
    # =========================================================================
    # En log-log, usamos el rango completo de n para las curvas teóricas.
    n_fino_log = np.logspace(
        np.log10(n_celdas[0] * 0.8),
        np.log10(n_celdas[-1] * 1.2),
        500
    )

    curva_on_log     = normalizar(n_fino_log,                        n_medio, t_medio)
    curva_onlogn_log = normalizar(n_fino_log * np.log2(n_fino_log + 1), n_medio, t_medio)
    curva_on2_log    = normalizar(n_fino_log**2,                      n_medio, t_medio)

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    ax2.loglog(
        n_celdas, tiempos,
        "o-", color="#4c9be8", linewidth=2, markersize=7,
        label="Tiempo medido", zorder=5
    )

    ax2.loglog(n_fino_log, curva_on_log,     "--", color="#2ecc71", linewidth=1.5, alpha=0.8,
               label="O(n) — pendiente 1")
    ax2.loglog(n_fino_log, curva_onlogn_log, "--", color="#f39c12", linewidth=1.5, alpha=0.8,
               label="O(n log n)")
    ax2.loglog(n_fino_log, curva_on2_log,    "--", color="#e74c3c", linewidth=1.5, alpha=0.8,
               label="O(n²) — pendiente 2")

    ax2.set_xlabel("Número de celdas (escala log)", fontsize=12)
    ax2.set_ylabel("Tiempo por iteración en ms (escala log)", fontsize=12)
    ax2.set_title("Rendimiento del Juego de la Vida — Escala Log-Log", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    ruta2 = f"{directorio_salida}/rendimiento_loglog.png"
    fig2.savefig(ruta2, dpi=150, bbox_inches="tight")
    print(f"  Gráfica log-log guardada: {ruta2}")
    plt.close(fig2)

    # =========================================================================
    # Estimamos la pendiente empírica en log-log (para reportar en consola)
    # =========================================================================
    # Si ajustamos log(t) = m·log(n) + b, la pendiente m nos dice la complejidad:
    #   m ≈ 1 → O(n), m ≈ 2 → O(n²), etc.
    log_n = np.log10(n_celdas)
    log_t = np.log10(tiempos)
    # Regresión lineal simple (mínimos cuadrados)
    pendiente, intercepto = np.polyfit(log_n, log_t, 1)
    print(f"\n  Pendiente empírica en log-log: {pendiente:.3f}")
    print(f"  (Pendiente ≈ 1.0 indica escalado O(n) lineal)")
    if 0.8 <= pendiente <= 1.2:
        print("  ✓ La implementación escala aproximadamente de forma LINEAL O(n).")
    elif pendiente < 0.8:
        print("  ✓ La implementación escala MEJOR que lineal (posiblemente por caché).")
    else:
        print("  ⚠ La implementación escala PEOR que lineal (cuello de botella detectado).")
