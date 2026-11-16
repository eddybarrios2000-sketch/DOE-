"""
Experimento DOE/RSM: ajuste de temperatura inicial para Recocido Simulado + 2-opt.
Matrícula/semilla: 2065883
Opción B: 5 niveles, 5 réplicas externas y 10 réplicas centrales por tamaño.
"""

import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# ---------------- CONFIGURACIÓN ----------------
MATRICULA = 2065883
RANGO_TEMPERATURA = (0.01, 2.00)
TAMANOS = [5, 10, 15]
REPLICAS_EXTERNAS = 5
REPLICAS_CENTRO = 10
ITERACIONES = 1500
BOOTSTRAP_MUESTRAS = 10000
ALPHA = 0.05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "results", "data")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------- NIVELES DOE ----------------
def calcular_niveles():
    """Genera r una sola vez y calcula los cinco niveles de temperatura."""
    random.seed(MATRICULA)
    r = random.random()

    fracciones = {
        "Axial bajo": 0.2 * r,
        "Bajo": 0.2 * r + 0.2,
        "Centro": 0.2 * r + 0.4,
        "Alto": 0.2 * r + 0.6,
        "Axial alto": 0.2 * r + 0.8,
    }

    t_min, t_max = RANGO_TEMPERATURA
    niveles = {
        nombre: t_min + fraccion * (t_max - t_min)
        for nombre, fraccion in fracciones.items()
    }
    return r, niveles


# ---------------- TSP Y 2-OPT ----------------
def create_random_tsp(n, seed):
    """Una instancia fija y reproducible para cada tamaño n."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def calculate_cost(route, coords):
    cost = 0.0
    for i in range(len(route)):
        p1 = coords[route[i]]
        p2 = coords[route[(i + 1) % len(route)]]
        cost += np.linalg.norm(p1 - p2)
    return float(cost)


def two_opt_swap(route, i, k):
    new_route = route[:]
    new_route[i:k + 1] = reversed(new_route[i:k + 1])
    return new_route


def simulated_annealing(coords, initial_temp, seed, iterations=ITERACIONES):
    """
    Recocido simulado con 2-opt.
    Se mantiene constante el número de iteraciones para comparar T0 justamente.
    El enfriamiento geométrico termina en min_temp=1e-4.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords debe contener pares de coordenadas (x, y).")
    if len(coords) < 2:
        raise ValueError("Se necesitan al menos dos ciudades.")
    if not np.isfinite(coords).all():
        raise ValueError("Las coordenadas deben ser valores finitos.")
    if not math.isfinite(initial_temp) or initial_temp <= 1e-4:
        raise ValueError("La temperatura inicial debe ser finita y mayor que 1e-4.")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 2:
        raise ValueError("iterations debe ser un entero mayor o igual que 2.")

    rng = random.Random(seed)
    n = len(coords)

    current_route = list(range(n))
    rng.shuffle(current_route)
    current_cost = calculate_cost(current_route, coords)

    best_route = current_route[:]
    best_cost = current_cost

    min_temp = 1e-4

    cooling_rate = (min_temp / initial_temp) ** (1 / max(1, iterations - 1))
    temp = initial_temp

    for _ in range(iterations):
        i, k = sorted(rng.sample(range(n), 2))
        new_route = two_opt_swap(current_route, i, k)
        new_cost = calculate_cost(new_route, coords)

        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_route = new_route
            current_cost = new_cost

            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best_cost, best_route


# ---------------- EXPERIMENTO ----------------
def ejecutar_experimento():
    r, niveles = calcular_niveles()
    print(f"Valor fijo de r: {r:.10f}")
    print("\nTemperaturas:")
    for nombre, temp in niveles.items():
        print(f"{nombre:12s}: {temp:.6f}")

    orden = ["Axial bajo", "Bajo", "Centro", "Alto", "Axial alto"]
    filas = []

    for n in TAMANOS:
        # Una instancia por tamaño, compartida por todas sus corridas.
        coords = create_random_tsp(n, seed=1000 + n)

        for nombre in orden:
            temp = niveles[nombre]
            replicas = REPLICAS_CENTRO if nombre == "Centro" else REPLICAS_EXTERNAS

            for replica in range(replicas):
                # Semilla distinta para cada corrida.
                seed = 2065883 + n * 100000 + orden.index(nombre) * 1000 + replica
                costo, _ = simulated_annealing(coords, temp, seed)

                filas.append({
                    "tamano": n,
                    "nivel": nombre,
                    "temperatura": temp,
                    "replica": replica + 1,
                    "semilla": seed,
                    "costo": costo
                })

        print(f"Completado n={n}")

    df = pd.DataFrame(filas)
    csv_path = os.path.join(DATA_DIR, "resultados_experimento.csv")
    df.to_csv(csv_path, index=False)
    return df, r, niveles


# ---------------- BOOTSTRAP ----------------
def bootstrap_mediana(valores, n_boot=BOOTSTRAP_MUESTRAS, seed=12345):
    """IC percentil del 95% para la mediana mediante remuestreo."""
    valores = np.asarray(valores, dtype=float)
    if valores.ndim != 1 or len(valores) == 0:
        raise ValueError("Se necesita al menos una observación para el bootstrap.")
    if not np.isfinite(valores).all():
        raise ValueError("Los valores del bootstrap deben ser finitos.")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot debe ser un entero positivo.")

    rng = np.random.default_rng(seed)
    muestras = rng.choice(valores, size=(n_boot, len(valores)), replace=True)
    medianas = np.median(muestras, axis=1)
    mediana = float(np.median(valores))
    li, ls = np.percentile(medianas, [2.5, 97.5])
    return mediana, float(li), float(ls)


# ---------------- ANÁLISIS RSM ----------------
def falta_de_ajuste(grupo, modelo):
    """
    Prueba clásica de falta de ajuste:
    SSE residual = falta de ajuste + error puro.
    Usa las réplicas dentro de cada nivel para estimar error puro.
    """
    sse_residual = float(np.sum(modelo.resid ** 2))
    niveles = grupo.groupby("temperatura")["costo"]
    sse_puro = float(sum(np.sum((g - g.mean()) ** 2) for _, g in niveles))

    n = len(grupo)
    p = 3  # intercepto, T y T^2
    m = grupo["temperatura"].nunique()
    gl_residual = n - p
    gl_puro = n - m
    gl_falta = m - p

    sse_falta = max(0.0, sse_residual - sse_puro)
    if gl_falta <= 0 or gl_puro <= 0 or sse_puro <= 0:
        return np.nan, np.nan, gl_falta, gl_puro

    ms_falta = sse_falta / gl_falta
    ms_puro = sse_puro / gl_puro
    f_stat = ms_falta / ms_puro
    p_value = float(stats.f.sf(f_stat, gl_falta, gl_puro))
    return float(f_stat), p_value, gl_falta, gl_puro


def analizar_y_graficar(df, niveles):
    resumen = []
    modelos_texto = []

    for n in TAMANOS:
        grupo = df[df["tamano"] == n].copy()

        costos = grupo["costo"].to_numpy(dtype=float)
        respuesta_constante = np.allclose(costos, costos[0], rtol=0, atol=1e-12)
        modelo = None
        anova_reg = None

        if respuesta_constante:
            beta0 = float(costos[0])
            beta1 = 0.0
            beta2 = 0.0
            p_curvatura = np.nan
            f_lof, p_lof = np.nan, np.nan
            gl_falta = int(grupo["temperatura"].nunique() - 3)
            gl_puro = int(len(grupo) - grupo["temperatura"].nunique())
            r2 = np.nan
        else:
            # Modelo cuadrático usando observaciones individuales.
            modelo = ols("costo ~ temperatura + I(temperatura**2)", data=grupo).fit()
            beta0 = modelo.params["Intercept"]
            beta1 = modelo.params["temperatura"]
            beta2 = modelo.params["I(temperatura ** 2)"]
            p_curvatura = float(modelo.pvalues["I(temperatura ** 2)"])
            f_lof, p_lof, gl_falta, gl_puro = falta_de_ajuste(grupo, modelo)
            r2 = float(modelo.rsquared)
            anova_reg = sm.stats.anova_lm(modelo, typ=2)

        # Confiar en el vértice solo si la curvatura es positiva y significativa.
        # Si no, seleccionar el mejor nivel realmente probado.
        if respuesta_constante:
            t_opt = np.nan
            metodo_opt = "Empate: costo constante; no hay óptimo identificable"
        elif beta2 > 0 and p_curvatura < ALPHA:
            t_opt = -beta1 / (2 * beta2)
            t_opt = float(np.clip(t_opt, grupo["temperatura"].min(),
                                  grupo["temperatura"].max()))
            metodo_opt = "Mínimo cuadrático (curvatura significativa)"
        else:
            medias = grupo.groupby(["nivel", "temperatura"])["costo"].mean()
            mejor = medias.idxmin()
            t_opt = float(mejor[1])
            metodo_opt = "Mejor nivel observado (curvatura no significativa)"

        # Nivel experimental más cercano a T* para el bootstrap solicitado.
        medias_nivel = grupo.groupby(["nivel", "temperatura"])["costo"].mean()
        if np.isfinite(t_opt):
            nivel_elegido, temp_elegida = min(
                medias_nivel.index, key=lambda x: abs(float(x[1]) - t_opt)
            )
        else:
            nivel_elegido = "Centro"
            temp_elegida = float(niveles[nivel_elegido])
        costos_punto = grupo[
            (grupo["nivel"] == nivel_elegido)
        ]["costo"].to_numpy()

        mediana, li, ls = bootstrap_mediana(costos_punto, seed=12345 + n)

        resumen.append({
            "tamano": n,
            "temperatura_optima_estimada": t_opt,
            "nivel_experimental_cercano": nivel_elegido,
            "temperatura_nivel_bootstrap": float(temp_elegida),
            "mediana_bootstrap": mediana,
            "IC95_inferior": li,
            "IC95_superior": ls,
            "beta2": float(beta2),
            "p_curvatura": p_curvatura,
            "p_falta_ajuste": p_lof,
            "F_falta_ajuste": f_lof,
            "gl_falta_ajuste": gl_falta,
            "gl_error_puro": gl_puro,
            "R2": r2,
            "metodo_optimo": metodo_opt
        })

        # Gráfica de curva ajustada, medias observadas e IC de medias.
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(grupo["temperatura"].min(),
                        grupo["temperatura"].max(), 250)
        ax.scatter(grupo["temperatura"], grupo["costo"], alpha=0.35,
                   label="Corridas individuales")
        if modelo is None:
            ax.axhline(beta0, linewidth=2.2,
                       label="Costo constante; sin ajuste identificable")
        else:
            pred_df = pd.DataFrame({"temperatura": x})
            pred = modelo.get_prediction(pred_df).summary_frame(alpha=ALPHA)
            ax.plot(x, pred["mean"], linewidth=2.2, label="Ajuste cuadrático")
            ax.fill_between(x, pred["mean_ci_lower"], pred["mean_ci_upper"],
                            alpha=0.2, label="IC 95% de la media ajustada")

        estad = grupo.groupby("temperatura")["costo"].agg(["mean", "std", "count"])
        sem = estad["std"] / np.sqrt(estad["count"])
        tcrit = stats.t.ppf(0.975, estad["count"] - 1)
        ax.errorbar(estad.index, estad["mean"], yerr=tcrit * sem,
                    fmt="o", capsize=4, label="Media observada ± IC 95%")

        if np.isfinite(t_opt):
            ax.axvline(t_opt, linestyle="--", linewidth=1.5,
                       label=f"T* estimada = {t_opt:.4f}")
        else:
            ax.text(0.02, 0.98, "Óptimo no identificable",
                    transform=ax.transAxes, va="top")
        ax.set_title(f"RSM: Costo vs Temperatura (n={n})")
        ax.set_xlabel("Temperatura inicial T₀")
        ax.set_ylabel("Costo de la ruta")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"curva_rsm_n{n}.png"), dpi=180)
        plt.close(fig)

        # Guardar resumen de regresión.
        if modelo is None:
            diagnostico_modelo = (
                "No se ajustó el modelo: todos los costos son iguales; "
                "no se puede estimar la curvatura ni identificar un óptimo.\n"
            )
        else:
            diagnostico_modelo = (
                f"{modelo.summary()}\n"
                f"ANOVA de regresión:\n{anova_reg.to_string()}\n"
            )
        modelos_texto.append(
            f"\n========== TAMAÑO n={n} ==========\n"
            f"{diagnostico_modelo}"
            f"Falta de ajuste: F={f_lof}, p={p_lof}, "
            f"gl=({gl_falta}, {gl_puro})\n"
            f"Temperatura óptima estimada: {t_opt:.6f}\n"
            f"Método: {metodo_opt}\n"
            f"Nivel cercano para bootstrap: {nivel_elegido} ({temp_elegida:.6f})\n"
        )

    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(os.path.join(DATA_DIR, "tabla_resumen.csv"), index=False)

    with open(os.path.join(DATA_DIR, "analisis_modelos.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(modelos_texto))

    return resumen_df


# ---------------- EFECTOS VISUALES ----------------
def grafica_niveles(df):
    """Gráfica descriptiva de medias por temperatura y tamaño."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in TAMANOS:
        g = df[df["tamano"] == n].groupby("temperatura")["costo"].mean()
        ax.plot(g.index, g.values, marker="o", label=f"n={n}")
    ax.set_title("Efecto de la temperatura según el tamaño")
    ax.set_xlabel("Temperatura inicial T₀")
    ax.set_ylabel("Costo promedio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparacion_tamanos.png"), dpi=180)
    plt.close(fig)


def main():
    df, r, niveles = ejecutar_experimento()
    resumen = analizar_y_graficar(df, niveles)
    grafica_niveles(df)

    print("\n===== TABLA RESUMEN =====")
    print(resumen[[
        "tamano", "temperatura_optima_estimada",
        "temperatura_nivel_bootstrap", "mediana_bootstrap",
        "IC95_inferior", "IC95_superior",
        "p_curvatura", "p_falta_ajuste"
    ]].to_string(index=False))

    print(f"\nDatos guardados en: {DATA_DIR}")
    print(f"Gráficas guardadas en: {FIG_DIR}")
    print("Total de corridas:", len(df))


if __name__ == "__main__":
    main()
