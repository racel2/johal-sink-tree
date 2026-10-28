# -*- coding: utf-8 -*-
"""
Laboratorio de Redes | Optimalidad y Sink Tree en Streamlit
Basado en: Johal, H. S. (2008). Optimality Principle Based Sink Tree Methodology
for Adaptive Static Routing Using Multiple Route Configuration Scheme.
DOI: 10.1109/WCECS.2008.21

Ejecutar:  python -m streamlit run app.py

AVISO: los valores en ms y Mbps y la función multicriterio son construcciones
didácticas de este laboratorio; NO son resultados experimentales de Johal.
"""

import io
import math

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# 1. DATOS DEL PROBLEMA
# ----------------------------------------------------------------------------
ORIGEN = "H"
DESTINO = "B"

# Posiciones fijas para dibujar la topología (x, y)
POS = {
    "H": (0.0, 0.0),
    "D": (1.2, 1.6),
    "A": (2.8, 1.6),
    "E": (1.2, 0.0),
    "J": (2.8, 0.0),
    "C": (4.2, 0.0),
    "O": (1.2, -1.6),
    "N": (2.8, -1.6),
    "B": (5.4, 0.0),
}
NODOS = list(POS.keys())

# Rutas preconfiguradas del artículo (prioridad estática)
RUTAS_ESTATICAS = {
    "PSRC": ["H", "D", "A", "B"],
    "SSRC": ["H", "E", "J", "C", "B"],
    "TSRC": ["H", "O", "N", "J", "C", "B"],
}

# Enlaces: (origen, destino, retardo_ms, banda_Mbps). Valores didácticos.
ENLACES_BASE = [
    ("H", "D", 8, 100),
    ("D", "A", 9, 100),
    ("A", "B", 7, 100),
    ("H", "E", 5, 50),
    ("E", "J", 10, 50),
    ("J", "C", 8, 100),
    ("C", "B", 6, 100),
    ("H", "O", 7, 50),
    ("O", "N", 9, 100),
    ("N", "J", 9, 100),
]

MODO_SALTOS = "Johal: mínimo de saltos"
MODO_RETARDO = "Mínimo retardo"
MODO_MULTI = "Multicriterio (α, β, γ)"
MODOS = [MODO_SALTOS, MODO_RETARDO, MODO_MULTI]


# ----------------------------------------------------------------------------
# 2. LÓGICA (sin dependencia de la interfaz)
# ----------------------------------------------------------------------------
# Escenarios de prueba: {(origen, destino): {columna: valor}} sobre la tabla base
ESCENARIOS = {
    "Base (datos del artículo)": {},
    "B. Retardo H–D = 40 ms": {("H", "D"): {"Retardo_ms": 40}},
    "C. Falla H–D": {("H", "D"): {"Disponible": False}},
    "D. Fallan H–D y H–E": {
        ("H", "D"): {"Disponible": False},
        ("H", "E"): {"Disponible": False},
    },
    "E. Cuello de botella E–J = 5 Mbps": {("E", "J"): {"Banda_Mbps": 5}},
    "Demo: cada criterio elige una ruta distinta": {
        ("H", "D"): {"Retardo_ms": 40, "Banda_Mbps": 20},
        ("D", "A"): {"Banda_Mbps": 20},
        ("A", "B"): {"Banda_Mbps": 20},
        ("H", "E"): {"Banda_Mbps": 10},
        ("E", "J"): {"Banda_Mbps": 10},
        ("H", "O"): {"Banda_Mbps": 100},
    },
}
NOMBRES_ESCENARIOS = list(ESCENARIOS.keys())


def tabla_inicial(escenario: str = "Base (datos del artículo)") -> pd.DataFrame:
    df = pd.DataFrame(
        ENLACES_BASE, columns=["Origen", "Destino", "Retardo_ms", "Banda_Mbps"]
    )
    df["Retardo_ms"] = df["Retardo_ms"].astype(float)
    df["Banda_Mbps"] = df["Banda_Mbps"].astype(float)
    df["Disponible"] = True
    for (u, v), cambios in ESCENARIOS.get(escenario, {}).items():
        fila = (df["Origen"] == u) & (df["Destino"] == v)
        for col, valor in cambios.items():
            df.loc[fila, col] = valor
    return df


def limpiar_tabla(df: pd.DataFrame):
    """Convierte tipos y devuelve (df_limpio, lista_de_errores)."""
    df = df.copy()
    df["Retardo_ms"] = pd.to_numeric(df["Retardo_ms"], errors="coerce")
    df["Banda_Mbps"] = pd.to_numeric(df["Banda_Mbps"], errors="coerce")
    df["Disponible"] = df["Disponible"].fillna(False).astype(bool)
    errores = []
    if df["Retardo_ms"].isna().any() or df["Banda_Mbps"].isna().any():
        errores.append("Hay celdas vacías o no numéricas en Retardo_ms o Banda_Mbps.")
    if (df["Retardo_ms"] < 0).any() or (df["Banda_Mbps"] < 0).any():
        errores.append("Retardo_ms y Banda_Mbps no pueden ser negativos (Dijkstra exige costos no negativos).")
    return df, errores


def referencias(df: pd.DataFrame):
    """d_ref y b_ref: máximos valores positivos de la tabla actual."""
    d_pos = df.loc[df["Retardo_ms"] > 0, "Retardo_ms"]
    b_pos = df.loc[df["Banda_Mbps"] > 0, "Banda_Mbps"]
    d_ref = float(d_pos.max()) if len(d_pos) else 1.0
    b_ref = float(b_pos.max()) if len(b_pos) else 1.0
    return d_ref, b_ref


def normalizar_pesos(alfa, beta, gamma):
    """Normaliza dividiendo cada peso entre la suma. Devuelve (a, b, g, suma_nula)."""
    s = alfa + beta + gamma
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3, True
    return alfa / s, beta / s, gamma / s, False


def costo_arista(d, b, alfa, beta, gamma, d_ref, b_ref):
    """C_ij = α + β·d/d_ref + γ·b_ref/b. Con b<=0 el enlace no es utilizable (inf)."""
    if b <= 0:
        return math.inf
    return alfa + beta * (d / d_ref) + gamma * (b_ref / b)


def dic_enlaces(df: pd.DataFrame) -> dict:
    return {
        frozenset((r.Origen, r.Destino)): {
            "d": float(r.Retardo_ms),
            "b": float(r.Banda_Mbps),
            "f": bool(r.Disponible),
        }
        for r in df.itertuples()
    }


def metricas_camino(camino, enlaces, d_ref, b_ref, alfa, beta, gamma):
    tramos = list(zip(camino[:-1], camino[1:]))
    datos = [enlaces[frozenset(t)] for t in tramos]
    costos = [costo_arista(x["d"], x["b"], alfa, beta, gamma, d_ref, b_ref) for x in datos]
    return {
        "Disponible": all(x["f"] for x in datos),
        "Saltos": len(tramos),
        "Retardo_ms": sum(x["d"] for x in datos),
        "Banda_min_Mbps": min(x["b"] for x in datos),
        "Costo_multicriterio": sum(costos),
    }


def valor_objetivo(modo, m):
    if modo == MODO_SALTOS:
        return float(m["Saltos"])
    if modo == MODO_RETARDO:
        return float(m["Retardo_ms"])
    return float(m["Costo_multicriterio"])


def construir_grafo(df, modo, alfa, beta, gamma, d_ref, b_ref):
    """Grafo solo con enlaces disponibles; 'peso' es el costo de la función objetivo."""
    G = nx.Graph()
    G.add_nodes_from(NODOS)
    for r in df.itertuples():
        if not r.Disponible:
            continue
        d, b = float(r.Retardo_ms), float(r.Banda_Mbps)
        if modo == MODO_SALTOS:
            w = 1.0
        elif modo == MODO_RETARDO:
            w = d
        else:
            w = costo_arista(d, b, alfa, beta, gamma, d_ref, b_ref)
            if not math.isfinite(w):
                continue  # capacidad 0: inutilizable en el modelo multicriterio
        G.add_edge(r.Origen, r.Destino, peso=w)
    return G


def camino_optimo(G):
    try:
        return nx.dijkstra_path(G, ORIGEN, DESTINO, weight="peso")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def sink_tree(G):
    """Devuelve (distancias, caminos_nodo_a_B). Un único siguiente salto por nodo."""
    dist, paths = nx.single_source_dijkstra(G, DESTINO, weight="peso")
    caminos = {n: list(reversed(p)) for n, p in paths.items()}  # nodo -> ... -> B
    return dist, caminos


def txt(camino):
    return "–".join(camino) if camino else "Sin ruta"


# ----------------------------------------------------------------------------
# 3. FIGURAS
# ----------------------------------------------------------------------------
def _segmentos(camino):
    xs, ys = [], []
    for u, v in zip(camino[:-1], camino[1:]):
        xs += [POS[u][0], POS[v][0], None]
        ys += [POS[u][1], POS[v][1], None]
    return xs, ys


def _base_figura(df, mostrar_etiquetas=True, tenue=False):
    fig = go.Figure()
    # enlaces disponibles y caídos
    for estado, color, dash in ((True, "#9aa5b1", "solid"), (False, "#d64545", "dot")):
        xs, ys = [], []
        for r in df.itertuples():
            if bool(r.Disponible) == estado:
                xs += [POS[r.Origen][0], POS[r.Destino][0], None]
                ys += [POS[r.Origen][1], POS[r.Destino][1], None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=2 if not tenue else 1.5, dash=dash),
                name="Enlace disponible" if estado else "Enlace caído",
                hoverinfo="skip", opacity=0.55 if tenue else 1,
            ))
    if mostrar_etiquetas:
        mx, my, tx = [], [], []
        for r in df.itertuples():
            mx.append((POS[r.Origen][0] + POS[r.Destino][0]) / 2)
            my.append((POS[r.Origen][1] + POS[r.Destino][1]) / 2)
            marca = "" if r.Disponible else " ✖"
            tx.append(f"{r.Retardo_ms:g} ms · {r.Banda_Mbps:g} Mbps{marca}")
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="text", text=tx, textposition="top center",
            textfont=dict(size=11), showlegend=False, hoverinfo="skip",
        ))
    return fig


def _agregar_nodos(fig, alcanzables=None):
    colores, textos = [], []
    for n in NODOS:
        if n == ORIGEN:
            colores.append("#2b6cb0")
        elif n == DESTINO:
            colores.append("#dd6b20")
        elif alcanzables is not None and n not in alcanzables:
            colores.append("#b0b0b0")
        else:
            colores.append("#4a5568")
        textos.append(n)
    fig.add_trace(go.Scatter(
        x=[POS[n][0] for n in NODOS], y=[POS[n][1] for n in NODOS],
        mode="markers+text", text=textos, textposition="middle center",
        textfont=dict(color="white", size=14),
        marker=dict(size=34, color=colores, line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="text",
        hovertext=[f"Router {n}" for n in NODOS],
    ))


def _formato(fig, titulo):
    fig.update_layout(
        title=titulo, height=520, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, range=[-0.7, 6.1]),
        yaxis=dict(visible=False, range=[-2.3, 2.3], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figura_rutas(df, c_opt, c_est):
    fig = _base_figura(df)
    if c_est:
        xs, ys = _segmentos(c_est)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="rgba(49,130,206,0.40)", width=14),
            name="Ruta estática (artículo)", hoverinfo="skip"))
    if c_opt:
        xs, ys = _segmentos(c_opt)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="#1a9850", width=5),
            name="Ruta óptima recalculada", hoverinfo="skip"))
    _agregar_nodos(fig)
    return _formato(fig, "Topología, ruta estática y ruta óptima H → B")


def figura_sink_tree(df, caminos):
    fig = _base_figura(df, mostrar_etiquetas=False, tenue=True)
    for n, cam in caminos.items():
        if n == DESTINO:
            continue
        sig = cam[1]
        (x0, y0), (x1, y1) = POS[n], POS[sig]
        k = 0.16  # recorta la flecha para no tapar los nodos
        ax, ay = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
        px, py = x1 - (x1 - x0) * k, y1 - (y1 - y0) * k
        fig.add_annotation(
            x=px, y=py, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=3.5,
            arrowcolor="#1a9850",
        )
    _agregar_nodos(fig, alcanzables=set(caminos.keys()))
    fig = _formato(fig, "Sink Tree T_B: un único siguiente salto de cada router hacia B")
    return fig


# ----------------------------------------------------------------------------
# 4. INTERFAZ STREAMLIT
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Laboratorio Sink Tree – Johal (2008)", layout="wide")

st.title("Optimalidad y Sink Tree en Streamlit")
st.caption(
    "Laboratorio de Redes | Basado en Johal, H. S. (2008), DOI 10.1109/WCECS.2008.21. "
    "Los datos en ms y Mbps y la función multicriterio son construcciones didácticas, "
    "no resultados experimentales del autor."
)

if "reset_n" not in st.session_state:
    st.session_state["reset_n"] = 0
n_reset = st.session_state["reset_n"]

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Controles")
    if st.button("Restablecer valores del artículo"):
        st.session_state["reset_n"] += 1
        st.rerun()

    escenario = st.selectbox(
        "Escenario de prueba (carga la tabla)", NOMBRES_ESCENARIOS, key=f"esc_{n_reset}"
    )
    idx_esc = NOMBRES_ESCENARIOS.index(escenario)
    if escenario.startswith("Demo"):
        st.caption(
            "Con los pesos por defecto: saltos → H–D–A–B, retardo → H–E–J–C–B, "
            "multicriterio → H–O–N–J–C–B."
        )
    else:
        st.caption("Al cambiar de escenario se pierden las ediciones manuales de la tabla.")

    modo = st.radio("Función objetivo", MODOS, index=0, key=f"modo_{n_reset}")
    es_multi = modo == MODO_MULTI

    st.markdown("**Pesos (solo multicriterio)**")
    alfa_in = st.slider("α  (saltos)", 0.0, 1.0, 0.50, 0.05, key=f"alfa_{n_reset}", disabled=not es_multi)
    beta_in = st.slider("β  (demora)", 0.0, 1.0, 0.35, 0.05, key=f"beta_{n_reset}", disabled=not es_multi)
    gamma_in = st.slider("γ  (baja capacidad)", 0.0, 1.0, 0.15, 0.05, key=f"gamma_{n_reset}", disabled=not es_multi)

    alfa, beta, gamma, suma_nula = normalizar_pesos(alfa_in, beta_in, gamma_in)
    if es_multi:
        if suma_nula:
            st.warning("Los tres pesos son 0: se usan α=β=γ=1/3.")
        st.caption(f"Pesos normalizados: α={alfa:.4f}, β={beta:.4f}, γ={gamma:.4f}")

# ---------------- Sección 1: tabla de enlaces ----------------
st.header("1. Tabla de enlaces")
st.write("Edite **Retardo_ms**, **Banda_Mbps** y **Disponible** directamente en la tabla.")

df_editado = st.data_editor(
    tabla_inicial(escenario),
    key=f"editor_{n_reset}_{idx_esc}",
    hide_index=True,
    num_rows="fixed",
    disabled=["Origen", "Destino"],
    column_config={
        "Retardo_ms": st.column_config.NumberColumn("Retardo_ms", min_value=0.0, step=1.0),
        "Banda_Mbps": st.column_config.NumberColumn("Banda_Mbps", min_value=0.0, step=5.0),
        "Disponible": st.column_config.CheckboxColumn("Disponible"),
    },
)

df, errores = limpiar_tabla(df_editado)
if errores:
    for e in errores:
        st.error(e)
    st.stop()

d_ref, b_ref = referencias(df)
enlaces = dic_enlaces(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("d_ref (ms)", f"{d_ref:g}")
c2.metric("b_ref (Mbps)", f"{b_ref:g}")
c3.metric("Función objetivo", modo.split(" (")[0])
c4.metric("α · β · γ", f"{alfa:.2f} · {beta:.2f} · {gamma:.2f}" if es_multi else "—")
st.caption("Registre d_ref y b_ref en cada experimento: cambian si modifica los retardos o capacidades máximos.")

if es_multi and (df["Banda_Mbps"] <= 0).any():
    st.warning("Hay enlaces con 0 Mbps: en el modelo multicriterio b_ref/b_ij no está definido y esos enlaces se excluyen del cálculo.")

# ---------------- Cálculos principales ----------------
G = construir_grafo(df, modo, alfa, beta, gamma, d_ref, b_ref)
c_opt = camino_optimo(G)

filas_rutas = []
for nombre, cam in RUTAS_ESTATICAS.items():
    m = metricas_camino(cam, enlaces, d_ref, b_ref, alfa, beta, gamma)
    m["Utilizable"] = m["Disponible"] and math.isfinite(valor_objetivo(modo, m))
    m["Valor_objetivo"] = valor_objetivo(modo, m)
    filas_rutas.append({"Ruta": nombre, "Recorrido": txt(cam), **m})
tabla_rutas = pd.DataFrame(filas_rutas)

# Selección estática: primera ruta preconfigurada disponible (prioridad PSRC > SSRC > TSRC)
c_est, nombre_est = None, None
for nombre, cam in RUTAS_ESTATICAS.items():
    fila = tabla_rutas.loc[tabla_rutas["Ruta"] == nombre].iloc[0]
    if fila["Disponible"]:
        c_est, nombre_est = cam, nombre
        break

# Mejor de las tres bajo la función objetivo actual
utilizables = tabla_rutas[tabla_rutas["Utilizable"]]
mejor_tres = None
if not utilizables.empty:
    mejor_tres = utilizables.sort_values("Valor_objetivo", kind="stable").iloc[0]

m_opt = metricas_camino(c_opt, enlaces, d_ref, b_ref, alfa, beta, gamma) if c_opt else None
m_est = metricas_camino(c_est, enlaces, d_ref, b_ref, alfa, beta, gamma) if c_est else None

# ---------------- Sección 2 ----------------
st.header("2. Función objetivo, rutas y comparación")

if modo == MODO_SALTOS:
    st.latex(r"\min_{P}\; H(P)=\sum_{(i,j)\in P} h_{ij},\qquad h_{ij}=1")
elif modo == MODO_RETARDO:
    st.latex(r"\min_{P}\; D(P)=\sum_{(i,j)\in P} d_{ij}")
else:
    st.latex(
        r"\min_{P}\; C(P)=\sum_{(i,j)\in P}\left[\alpha+\beta\,\frac{d_{ij}}{d_{ref}}"
        r"+\gamma\,\frac{b_{ref}}{b_{ij}}\right],\qquad \alpha+\beta+\gamma=1"
    )
st.latex(r"B(P)=\min_{(i,j)\in P} b_{ij}\qquad(\text{capacidad del camino; no es la penalización } b_{ref}/b_{ij})")

col_fig, col_res = st.columns([3, 2])
with col_fig:
    st.plotly_chart(figura_rutas(df, c_opt, c_est))
with col_res:
    st.subheader("Ruta estática (artículo)")
    if c_est:
        st.success(f"{nombre_est}: {txt(c_est)}")
        st.write(
            f"H = {m_est['Saltos']} saltos · D = {m_est['Retardo_ms']:g} ms · "
            f"B = {m_est['Banda_min_Mbps']:g} Mbps · C = {m_est['Costo_multicriterio']:.4f}"
        )
    else:
        st.error("Ninguna de las tres rutas preconfiguradas está disponible.")

    st.subheader("Ruta óptima recalculada")
    if c_opt:
        st.success(txt(c_opt))
        st.write(
            f"H = {m_opt['Saltos']} saltos · D = {m_opt['Retardo_ms']:g} ms · "
            f"B = {m_opt['Banda_min_Mbps']:g} Mbps · C = {m_opt['Costo_multicriterio']:.4f}"
        )
    else:
        st.error("No existe camino H → B con los enlaces disponibles.")

    if c_est and c_opt:
        if c_est == c_opt:
            st.info("La ruta estática seleccionada coincide con la óptima recalculada.")
        else:
            v_est = valor_objetivo(modo, m_est)
            v_opt = valor_objetivo(modo, m_opt)
            st.warning(
                f"La ruta estática ({nombre_est}) difiere de la óptima. "
                f"Valor objetivo: estática = {v_est:.4f}, óptima = {v_opt:.4f}."
            )

st.subheader("Rutas preconfiguradas bajo la función objetivo actual")
vista = tabla_rutas.drop(columns=["Utilizable"]).copy()
vista["Costo_multicriterio"] = vista["Costo_multicriterio"].round(4)
vista["Valor_objetivo"] = vista["Valor_objetivo"].round(4)
st.dataframe(vista, hide_index=True)

if mejor_tres is not None and c_opt is not None:
    cam_mejor = RUTAS_ESTATICAS[mejor_tres["Ruta"]]
    if cam_mejor == c_opt:
        st.caption(
            f"Mejor de las tres preconfiguradas: {mejor_tres['Ruta']} ({txt(cam_mejor)}). "
            "Coincide con el óptimo sobre todo el grafo."
        )
    else:
        st.caption(
            f"Mejor de las tres preconfiguradas: {mejor_tres['Ruta']} ({txt(cam_mejor)}). "
            f"Difiere del óptimo sobre todo el grafo ({txt(c_opt)}): registre ambos resultados."
        )

with st.expander("Desglose por arista (para el cálculo manual)"):
    opciones = list(RUTAS_ESTATICAS.keys()) + (["Óptima recalculada"] if c_opt else [])
    elegida = st.selectbox("Ruta a desglosar", opciones)
    cam = c_opt if elegida == "Óptima recalculada" else RUTAS_ESTATICAS[elegida]
    filas = []
    for u, v in zip(cam[:-1], cam[1:]):
        x = enlaces[frozenset((u, v))]
        c_term_a = alfa
        c_term_b = beta * x["d"] / d_ref
        c_term_g = gamma * b_ref / x["b"] if x["b"] > 0 else math.inf
        filas.append({
            "Enlace": f"{u}–{v}", "d_ij (ms)": x["d"], "b_ij (Mbps)": x["b"],
            "Disponible": x["f"], "α": round(c_term_a, 4),
            "β·d/d_ref": round(c_term_b, 4),
            "γ·b_ref/b": round(c_term_g, 4) if math.isfinite(c_term_g) else "∞",
            "C_ij": round(c_term_a + c_term_b + c_term_g, 4) if math.isfinite(c_term_g) else "∞",
        })
    desglose = pd.DataFrame(filas)
    st.dataframe(desglose, hide_index=True)
    st.caption("Los términos α, β y γ usan los pesos normalizados de la barra lateral. C(P) es la suma de la columna C_ij.")

# ---------------- Sección 3: Sink Tree ----------------
st.header("3. Sink Tree hacia B")
dist, caminos = sink_tree(G)

filas_tree = []
for n in NODOS:
    if n == DESTINO:
        filas_tree.append({
            "Router": n, "Siguiente_salto": "— (destino)", "Camino_hacia_B": "B",
            "Saltos": 0, "Retardo_ms": 0.0, "Banda_min_Mbps": None,
            "Costo_multicriterio": 0.0, "Valor_objetivo": 0.0,
        })
    elif n in caminos:
        cam = caminos[n]
        m = metricas_camino(cam, enlaces, d_ref, b_ref, alfa, beta, gamma)
        filas_tree.append({
            "Router": n, "Siguiente_salto": cam[1], "Camino_hacia_B": txt(cam),
            "Saltos": m["Saltos"], "Retardo_ms": m["Retardo_ms"],
            "Banda_min_Mbps": m["Banda_min_Mbps"],
            "Costo_multicriterio": round(m["Costo_multicriterio"], 4),
            "Valor_objetivo": round(valor_objetivo(modo, m), 4),
        })
    else:
        filas_tree.append({
            "Router": n, "Siguiente_salto": "Sin ruta", "Camino_hacia_B": "Sin ruta",
            "Saltos": None, "Retardo_ms": None, "Banda_min_Mbps": None,
            "Costo_multicriterio": None, "Valor_objetivo": None,
        })
tabla_tree = pd.DataFrame(filas_tree)

col_t1, col_t2 = st.columns([3, 2])
with col_t1:
    st.plotly_chart(figura_sink_tree(df, caminos))
with col_t2:
    st.dataframe(tabla_tree[["Router", "Siguiente_salto", "Camino_hacia_B", "Valor_objetivo"]], hide_index=True)
    st.caption(
        "El Sink Tree fija un solo siguiente salto por router; no incluye rutas de respaldo. "
        "Los routers en gris no tienen camino hacia B con los enlaces disponibles. "
        "Si hay empates de costo, se conserva el que devuelve Dijkstra (determinista)."
    )

with st.expander("Verificación del principio de optimalidad"):
    if c_opt:
        filas_v = []
        for i, nodo in enumerate(c_opt[:-1]):
            sub = c_opt[i:]
            m_sub = metricas_camino(sub, enlaces, d_ref, b_ref, alfa, beta, gamma)
            v_sub = valor_objetivo(modo, m_sub)
            v_tree = dist[nodo]
            filas_v.append({
                "Router": nodo, "Subcamino de la ruta óptima": txt(sub),
                "Valor del subcamino": round(v_sub, 4),
                "Valor óptimo desde el router": round(v_tree, 4),
                "¿Coinciden?": "Sí" if math.isclose(v_sub, v_tree, rel_tol=1e-9, abs_tol=1e-9) else "No",
            })
        st.dataframe(pd.DataFrame(filas_v), hide_index=True)
        st.caption("Si la ruta H → B es óptima, cada subcamino hacia B también lo es desde su propio router.")
    else:
        st.info("No hay ruta H → B que verificar.")

# ---------------- Sección 4: definiciones y exportación ----------------
st.header("4. Definiciones y exportación")

definiciones = pd.DataFrame(
    [
        ("G=(V,E)", "Grafo de la red", "—"),
        ("s, t", "Origen H, destino B", "—"),
        ("h_ij", "Costo por salto (1)", "saltos"),
        ("d_ij", "Retardo de enlace (ejemplo)", "ms"),
        ("b_ij", "Capacidad de enlace (ejemplo)", "Mbps"),
        ("f_ij", "Disponibilidad: 1 activo, 0 caído", "booleano"),
        ("H(P)", "Cantidad de saltos del camino", "saltos"),
        ("D(P)", "Retardo sumado del camino", "ms"),
        ("B(P)", "Mínimo ancho de banda de sus aristas", "Mbps"),
        ("C(P)", "Costo multicriterio normalizado", "adimensional"),
        ("α, β, γ", "Pesos de saltos, demora y baja capacidad", "adimensional"),
        ("d_ref, b_ref", "Máximos positivos de retardo y capacidad en la tabla actual", "ms, Mbps"),
        ("T_B", "Árbol de siguientes saltos hacia B", "—"),
    ],
    columns=["Símbolo", "Definición", "Unidad"],
)
st.dataframe(definiciones, hide_index=True)

# CSV de resultados (rutas + Sink Tree en un solo archivo)
contexto = {
    "Funcion_objetivo": modo, "alfa": round(alfa, 6), "beta": round(beta, 6),
    "gamma": round(gamma, 6), "d_ref_ms": d_ref, "b_ref_Mbps": b_ref,
}
filas_csv = []
for _, r in tabla_rutas.iterrows():
    filas_csv.append({
        "Tipo": "Ruta preconfigurada", "Nombre": r["Ruta"], "Camino": r["Recorrido"],
        "Siguiente_salto": "", "Disponible": r["Disponible"], "Saltos": r["Saltos"],
        "Retardo_ms": r["Retardo_ms"], "Banda_min_Mbps": r["Banda_min_Mbps"],
        "Costo_multicriterio": r["Costo_multicriterio"], "Valor_objetivo": r["Valor_objetivo"],
        "Seleccionada_estatica": r["Ruta"] == nombre_est, **contexto,
    })
if c_opt:
    filas_csv.append({
        "Tipo": "Ruta óptima recalculada", "Nombre": "Óptima", "Camino": txt(c_opt),
        "Siguiente_salto": c_opt[1], "Disponible": True, "Saltos": m_opt["Saltos"],
        "Retardo_ms": m_opt["Retardo_ms"], "Banda_min_Mbps": m_opt["Banda_min_Mbps"],
        "Costo_multicriterio": m_opt["Costo_multicriterio"],
        "Valor_objetivo": valor_objetivo(modo, m_opt),
        "Seleccionada_estatica": False, **contexto,
    })
for _, r in tabla_tree.iterrows():
    filas_csv.append({
        "Tipo": "Sink Tree", "Nombre": r["Router"], "Camino": r["Camino_hacia_B"],
        "Siguiente_salto": r["Siguiente_salto"], "Disponible": r["Camino_hacia_B"] != "Sin ruta",
        "Saltos": r["Saltos"], "Retardo_ms": r["Retardo_ms"],
        "Banda_min_Mbps": r["Banda_min_Mbps"], "Costo_multicriterio": r["Costo_multicriterio"],
        "Valor_objetivo": r["Valor_objetivo"], "Seleccionada_estatica": False, **contexto,
    })
csv_resultados = pd.DataFrame(filas_csv).to_csv(index=False).encode("utf-8-sig")

buf = io.StringIO()
df.to_csv(buf, index=False)
csv_enlaces = buf.getvalue().encode("utf-8-sig")

b1, b2 = st.columns(2)
b1.download_button("Descargar CSV de resultados", csv_resultados,
                   file_name="resultados_sink_tree.csv", mime="text/csv")
b2.download_button("Descargar CSV de la tabla de enlaces", csv_enlaces,
                   file_name="tabla_enlaces.csv", mime="text/csv")
st.caption("El CSV de resultados incluye la función objetivo, los pesos normalizados, d_ref y b_ref del experimento.")
