# app.py — Pokédex Competitiva (Streamlit)
# Requisitos: pip install streamlit pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

# -----------------------------
# Configuración de rutas
# -----------------------------
# Obtener el directorio base del proyecto (donde está app.py)
BASE_DIR = Path(__file__).parent.absolute()

# -----------------------------
# Config básica de la app
# -----------------------------
st.set_page_config(page_title="Pokédex Competitiva", page_icon="🎮", layout="wide")


# -----------------------------
# Fondo global embebido
# -----------------------------
def inject_bg_image(png_path: str, opacity: float = 0.08, size: str = "800px"):
    """Embebe la imagen como data URI y la aplica de fondo translúcido."""
    p = Path(png_path)
    if not p.exists():
        st.warning(f"No encuentro la imagen de fondo: {png_path}")
        return
    b64 = base64.b64encode(p.read_bytes()).decode()
    css = f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/png;base64,{b64}");
        background-repeat: no-repeat;
        background-position: right -120px bottom -120px;
        background-size: {size};
        opacity: {opacity};
        pointer-events: none;
        z-index: 0;
    }}
    .stApp > div {{
        position: relative;
        z-index: 1;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_bg_image(str(BASE_DIR / "img" / "pokeball_bg.png"), opacity=0.09, size="900px")


# -----------------------------
# Carga de datos
# -----------------------------
@st.cache_data
def load_data(path=None):
    if path is None:
        path = BASE_DIR / "data" / "pokemon_unified_enriched.csv"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
    df = pd.read_csv(path)
    for c in ["name", "type1", "type2"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    need = [
        "id",
        "name",
        "type1",
        "type2",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
        "base_total",
        "height",
        "weight",
        "abilities",
        "sprite_png",
        "sprite_svg",
        "generation",
        "legendary",
        "weak_to",
        "resist_to",
        "immune_to",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    if "rol" not in df.columns:
        df["rol"] = infer_roles(df)
    return df


def infer_roles(df):
    pct = {
        col: {
            "hi": df[col].quantile(0.75),
            "mid": df[col].quantile(0.50),
            "lo": df[col].quantile(0.25),
        }
        for col in ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    }

    def _rol(r):
        atk_hi = (r["attack"] >= pct["attack"]["hi"]) or (
            r["sp_attack"] >= pct["sp_attack"]["hi"]
        )
        spd_hi = r["speed"] >= pct["speed"]["hi"]
        hp_hi = r["hp"] >= pct["hp"]["hi"]
        def_hi = r["defense"] >= pct["defense"]["hi"]
        spdef_hi = r["sp_defense"] >= pct["sp_defense"]["hi"]
        spd_lo = r["speed"] <= pct["speed"]["lo"]
        def_mid = (r["defense"] >= pct["defense"]["mid"]) or (
            r["sp_defense"] >= pct["sp_defense"]["mid"]
        )

        if atk_hi and spd_hi:
            return "Sweeper"
        if hp_hi and def_hi:
            return "Tanque físico"
        if hp_hi and spdef_hi:
            return "Tanque especial"
        if spd_lo and def_mid:
            return "Soporte"
        return "Equilibrado"

    return df.apply(_rol, axis=1)


df = load_data()

# -----------------------------
# Utilidades de visualización
# -----------------------------
GEN_INFO = {
    1: ("Gen I", "Kanto", "1996–1999"),
    2: ("Gen II", "Johto", "1999–2002"),
    3: ("Gen III", "Hoenn", "2002–2006"),
    4: ("Gen IV", "Sinnoh", "2006–2010"),
    5: ("Gen V", "Unova", "2010–2013"),
    6: ("Gen VI", "Kalos", "2013–2016"),
    7: ("Gen VII", "Alola", "2016–2019"),
    8: ("Gen VIII", "Galar", "2019–2022"),
    9: ("Gen IX", "Paldea", "2022–"),
}


def gen_label(gen_value):
    try:
        g = int(gen_value)
    except Exception:
        return "-"
    name, region, years = GEN_INFO.get(g, (f"Gen {g}", "-", "-"))
    return f"{name} — {region} ({years})"


def tipo_str(row):
    t1 = str(row.get("type1", "")).capitalize() if pd.notna(row.get("type1")) else "-"
    t2v = row.get("type2")
    t2 = f"/{str(t2v).capitalize()}" if pd.notna(t2v) and str(t2v) != "nan" else ""
    return f"{t1}{t2}"


def pick_image(row, width=150):
    svg = row.get("sprite_svg")
    png = row.get("sprite_png")
    if pd.notna(svg) and str(svg).startswith("http"):
        st.image(svg, width=width)
    elif pd.notna(png) and str(png).startswith("http"):
        st.image(png, width=width)
    else:
        st.write("—")


def parse_types(val):
    if pd.isna(val):
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]


def type_advantage(a_row, b_row):
    a_types = [t for t in [a_row.get("type1"), a_row.get("type2")] if pd.notna(t)]
    b_types = [t for t in [b_row.get("type1"), b_row.get("type2")] if pd.notna(t)]

    a_weak = parse_types(a_row.get("weak_to"))
    a_resist = parse_types(a_row.get("resist_to"))
    a_immune = parse_types(a_row.get("immune_to"))
    b_weak = parse_types(b_row.get("weak_to"))
    b_resist = parse_types(b_row.get("resist_to"))
    b_immune = parse_types(b_row.get("immune_to"))

    score_a, score_b = 0, 0
    reasons_a, reasons_b = [], []

    for t in a_types:
        if t in b_weak:
            score_a += 2
            reasons_a.append(f"{t.capitalize()} es supereficaz vs {', '.join(b_types)}")
        if t in b_resist:
            score_a -= 1
            reasons_a.append(f"{t.capitalize()} es poco eficaz vs {', '.join(b_types)}")
        if t in b_immune:
            score_a -= 3
            reasons_a.append(f"{t.capitalize()} no afecta a {', '.join(b_types)}")

    for t in b_types:
        if t in a_weak:
            score_b += 2
            reasons_b.append(f"{t.capitalize()} es supereficaz vs {', '.join(a_types)}")
        if t in a_resist:
            score_b -= 1
            reasons_b.append(f"{t.capitalize()} es poco eficaz vs {', '.join(a_types)}")
        if t in a_immune:
            score_b -= 3
            reasons_b.append(f"{t.capitalize()} no afecta a {', '.join(a_types)}")

    return (score_a, reasons_a), (score_b, reasons_b)


def decide_duel(row_a, row_b):
    score_a = (
        float(row_a.get("base_total", 0)) / 100.0 + float(row_a.get("speed", 0)) / 50.0
    )
    score_b = (
        float(row_b.get("base_total", 0)) / 100.0 + float(row_b.get("speed", 0)) / 50.0
    )
    (ta, ra), (tb, rb) = type_advantage(row_a, row_b)
    score_a += ta
    score_b += tb

    if score_a > score_b:
        return "A", score_a - score_b, ra, score_a, score_b
    elif score_b > score_a:
        return "B", score_b - score_a, rb, score_a, score_b
    else:
        return "tie", 0.0, ra + rb, score_a, score_b


# -----------------------------
# Sidebar navegación + música
# -----------------------------
with st.sidebar:
    st.header("⚙️ Navegación")
    menu = st.radio(
        "Selecciona una vista:",
        [
            "🏠 Portada",
            "🧭 Idea de negocio",
            "📄 Ficha Pokémon",
            "⚔️ Comparador",
            "📊 Análisis competitivo",
            "📐 Conclusión estadística",
        ],
        index=0,
    )
    st.divider()
    st.caption("🎵 Música (haz click en ▶️ para reproducir)")
    st.audio(str(BASE_DIR / "music" / "pokemon_theme.mp3"))

# -----------------------------
# 1) Portada
# -----------------------------
if menu == "🏠 Portada":
    st.title("🎮 Pokédex Competitiva — Análisis EDA Pokémon")

    st.markdown("""
    ### Bienvenido a la **Pokédex Competitiva**
    Este proyecto explora el universo Pokémon desde una perspectiva analítica y competitiva.

    - **Compara** dos Pokémon según sus estadísticas base, rol estimado y **ventajas de tipo**.
    - **Explora** debilidades, resistencias e inmunidades por tipo.
    - **Analiza** su **rol competitivo** dentro del entorno de los eSports Pokémon.

    ---
    **Fuentes de datos:**
    - [PokeAPI](https://pokeapi.co/)
    - [Kaggle Pokémon Dataset](https://www.kaggle.com/abcsds/pokemon)
    """)
    st.info("Selecciona una opción en la barra lateral para comenzar.")

# -----------------------------
# 2) Idea de negocio
# -----------------------------
elif menu == "🧭 Idea de negocio":
    st.title("🧭 Pokédex Competitiva — Propuesta de valor")

    st.subheader("🎯 Problema actual en eSports Pokémon")
    st.markdown("""
El ecosistema competitivo crece (más formatos, más temporadas, más jugadores), pero **la información clave está fragmentada**:
- Listas de equipos, spreads y roles se publican en **foros, vídeos y hojas sueltas**, con **criterios no estandarizados**.
- Los jugadores pasan **mucho tiempo** buscando counters, preparando matchups y comparando opciones **sin un panel unificado**.
- Los entrenadores necesitan **argumentar decisiones** (por qué A sobre B) con datos objetivos y reproducibles.
- Organizadores y analistas carecen de una **visión agregada** del metajuego por tipo o rol.

Resultado: **decisiones lentas**, **sesgos**, y **baja eficiencia competitiva**.
    """)

    st.subheader("💡 Solución propuesta")
    st.markdown("""
**Pokédex Competitiva** centraliza y estandariza:
- **Ficha unificada** por Pokémon (stats base, **rol estimado**, tipos, debilidades/resistencias/inmunidades).
- **Comparador instantáneo** con **veredicto explicable** (stats + ventaja de tipos).
- **Panel analítico** por tipo y rol para detectar estilos dominantes.
- Interfaz **rápida y visual** (Streamlit) sobre datos abiertos (*PokeAPI + Kaggle*).
    """)

    st.subheader("🧪 Enfoque analítico e hipótesis")
    st.markdown("""
- **H₀:** Tipo y rol competitivo son independientes.
- **H₁:** Existe relación entre tipo y rol (p. ej., ciertos tipos concentran *Sweepers* o *Tanques*).
El EDA aporta **evidencia cuantitativa y reproducible** para contrastarlo.
    """)

    st.subheader("📈 KPI de utilidad (MVP)")
    st.markdown("""
- Tiempo de preparación ↓
- Precisión en predicción de matchups ↑
- Adopción: usuarios activos y consultas por sesión ↑
- Cobertura: % de Pokémon con rol y ficha completa ↑
    """)

    st.subheader("💼 Modelo de producto")
    st.markdown("""
1) **MVP educativo** (gratuito): EDA + comparador + paneles.
2) **Pro (suscripción):** guardado de equipos, reportes y filtros por formato/regulación.
3) **Club/Team:** espacios compartidos, *scrims* con analítica y *playbooks*.
    """)

# -----------------------------
# 3) Ficha Pokémon
# -----------------------------
elif menu == "📄 Ficha Pokémon":
    st.title("📄 Ficha Pokémon")

    poke_list = sorted(df["name"].dropna().unique())
    default_idx = poke_list.index("garchomp") if "garchomp" in poke_list else 0

    poke = st.selectbox("Busca un Pokémon:", poke_list, index=default_idx)
    row = df[df["name"] == poke].iloc[0]

    c1, c2 = st.columns([1, 2])
    with c1:
        pick_image(row, width=200)
    with c2:
        st.subheader(f"{row['name'].capitalize()} (#{int(row['id'])})")
        st.write(f"**Tipo:** {tipo_str(row)}")
        st.write(f"**Rol estimado:** {row['rol']}")
        st.write(f"**Generación:** {gen_label(row.get('generation'))}")
        st.write(f"**Altura/Peso:** {row['height']} / {row['weight']}")
        leg = row.get("legendary")
        leg_str = "-" if pd.isna(leg) else ("Sí" if bool(leg) else "No")
        st.write(f"**Legendario:** {leg_str}")
        st.write(f"**Habilidades:** {row.get('abilities', '-')}")

    st.markdown("### Estadísticas base")
    stats = row[["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]].astype(
        float
    )
    st.bar_chart(stats)

    st.markdown("### Tipos: debilidades / resistencias / inmunidades")
    colw, colr, coli = st.columns(3)
    with colw:
        st.write("**Débil a:**", row.get("weak_to", "-"))
    with colr:
        st.write("**Resiste:**", row.get("resist_to", "-"))
    with coli:
        st.write("**Inmune a:**", row.get("immune_to", "-"))

# -----------------------------
# 4) Comparador
# -----------------------------
elif menu == "⚔️ Comparador":
    st.title("⚔️ Comparador de Pokémon")

    colA, colB = st.columns(2)
    with colA:
        poke_a = st.selectbox(
            "Pokémon A", sorted(df["name"].dropna().unique()), index=0
        )
        row_a = df[df["name"] == poke_a].iloc[0]
        pick_image(row_a, width=180)
        st.subheader(f"{row_a['name'].capitalize()}")
        st.write(f"**Tipo:** {tipo_str(row_a)}")
        st.write(f"**Rol:** {row_a['rol']}")
        st.write(f"**Base Total:** {int(row_a['base_total'])}")

    with colB:
        poke_b = st.selectbox(
            "Pokémon B", sorted(df["name"].dropna().unique()), index=1
        )
        row_b = df[df["name"] == poke_b].iloc[0]
        pick_image(row_b, width=180)
        st.subheader(f"{row_b['name'].capitalize()}")
        st.write(f"**Tipo:** {tipo_str(row_b)}")
        st.write(f"**Rol:** {row_b['rol']}")
        st.write(f"**Base Total:** {int(row_b['base_total'])}")

    st.markdown("### 🏁 Veredicto por stats y tipos")
    winner, margin, reasons, sA, sB = decide_duel(row_a, row_b)

    if winner == "A":
        st.success(
            f"🏆 **{row_a['name'].capitalize()}** gana el duelo (margen {margin:.2f})."
        )
    elif winner == "B":
        st.success(
            f"🏆 **{row_b['name'].capitalize()}** gana el duelo (margen {margin:.2f})."
        )
    else:
        st.info("🤝 **Empate estimado** (puntuaciones muy cercanas).")

    with st.expander("Ver explicación"):
        st.write(f"**Puntuación A** ({row_a['name'].capitalize()}): {sA:.2f}")
        st.write(f"**Puntuación B** ({row_b['name'].capitalize()}): {sB:.2f}")
        if reasons:
            st.write("**Motivos de ventaja/desventaja de tipos:**")
            for r in reasons:
                st.write("- " + r)
        st.caption(
            "Nota: cálculo simplificado; no considera habilidades activas, objetos, clima o prioridad de movimientos."
        )

# -----------------------------
# 5) Análisis competitivo
# -----------------------------
elif menu == "📊 Análisis competitivo":
    st.title("📊 Análisis competitivo")

    tipo = st.selectbox(
        "Selecciona un tipo de Pokémon:", sorted(df["type1"].dropna().unique())
    )
    subset = df[df["type1"] == tipo]

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(
            subset[
                ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
            ].mean()
        )
    with col2:
        st.write(
            subset[["name", "base_total"]]
            .sort_values("base_total", ascending=False)
            .head(10)
        )

# -----------------------------
# 6) Conclusión estadística — Test Chi-cuadrado (por permutación) + Heatmap
# -----------------------------
elif menu == "📐 Conclusión estadística":
    import matplotlib.pyplot as plt

    st.title("📐 Conclusión del contraste de hipótesis")

    st.markdown("""
    ### 📊 Explicación del contraste estadístico utilizado

    En esta sección se contrasta la **hipótesis nula (H₀)** de que el **tipo principal (`type1`) de los Pokémon y su rol competitivo (`rol`) son independientes**,
    es decir, que el tipo no influye en el rol que tiende a ocupar en combate.

    ---

    #### ⚔️ Test aplicado: Chi-cuadrado de independencia (versión por permutación)
    Se construye una **tabla de contingencia** con las frecuencias observadas de cada combinación `type1`–`rol`.
    Después, se calcula el **estadístico χ² (chi-cuadrado)** que mide la diferencia entre las frecuencias observadas y las esperadas bajo independencia.

    Para evitar depender de supuestos teóricos (como la normalidad o grandes muestras),
    se estima el *p-valor* mediante un **método Monte Carlo**, permutando aleatoriamente los roles y recalculando el χ² miles de veces.

    ---

    #### 📐 Hipótesis
    - **H₀:** El tipo (`type1`) y el rol competitivo (`rol`) son independientes.
    - **H₁:** Existe asociación entre el tipo y el rol competitivo.

    ---

    #### 📊 Interpretación
    - Si el **p-valor < 0.05**, se **rechaza H₀** → el tipo influye significativamente en el rol competitivo.
    - Si el **p-valor ≥ 0.05**, **no se rechaza H₀** → no hay evidencia estadística de asociación.

    ---

    #### 🧠 Beneficio del enfoque
    El test por permutación es **no paramétrico**, **robusto** y **no depende de distribuciones teóricas**.
    Esto lo hace especialmente adecuado para datos categóricos o desbalanceados, como los roles o tipos de Pokémon,
    donde las frecuencias no siempre son uniformes ni suficientes para aplicar un ANOVA tradicional.
    """)

    st.divider()

    # --- Ejecución del test sobre el dataframe df ---
    df_test = df.loc[df["type1"].notna() & df["rol"].notna(), ["type1", "rol"]].copy()
    obs = pd.crosstab(df_test["type1"], df_test["rol"]).astype(float)
    row_sum = obs.sum(axis=1).values.reshape(-1, 1)
    col_sum = obs.sum(axis=0).values.reshape(1, -1)
    total = obs.values.sum()

    exp = (row_sum @ col_sum) / total
    mask = exp > 0
    chi2_obs = ((obs.values - exp) ** 2 / np.where(mask, exp, np.nan)).sum()

    n_perm = 2000
    greater = 0
    roles = df_test["rol"].to_numpy()
    types = df_test["type1"].to_numpy()
    type_levels = obs.index.tolist()
    rol_levels = obs.columns.tolist()

    for _ in range(n_perm):
        perm_roles = np.random.permutation(roles)
        perm_df = pd.DataFrame({"type1": types, "rol": perm_roles})
        perm_tab = (
            pd.crosstab(perm_df["type1"], perm_df["rol"])
            .reindex(index=type_levels, columns=rol_levels, fill_value=0)
            .astype(float)
        )
        rs = perm_tab.sum(axis=1).values.reshape(-1, 1)
        cs = perm_tab.sum(axis=0).values.reshape(1, -1)
        tt = perm_tab.values.sum()
        exp_p = (rs @ cs) / tt
        msk_p = exp_p > 0
        chi2_p = ((perm_tab.values - exp_p) ** 2 / np.where(msk_p, exp_p, np.nan)).sum()
        if chi2_p >= chi2_obs:
            greater += 1

    p_val = (greater + 1) / (n_perm + 1)
    alpha = 0.05

    st.subheader("📈 Resultados del test")
    st.write(f"**Estadístico χ² observado:** {chi2_obs:.2f}")
    st.write(f"**Permutaciones:** {n_perm}")
    st.write(f"**p-valor (Monte Carlo):** {p_val:.4f}")

    if p_val < alpha:
        st.success(
            "✅ *p* < 0.05 → **Se rechaza H₀**. Existe asociación significativa entre tipo y rol competitivo."
        )
    else:
        st.info(
            "ℹ️ *p* ≥ 0.05 → **No se rechaza H₀**. No hay evidencia estadística de asociación."
        )

    # --- Top contribuciones ---
    std_resid = (obs.values - exp) / np.sqrt(np.where(mask, exp, np.nan))
    sr_df = pd.DataFrame(std_resid, index=obs.index, columns=obs.columns)

    top = (
        sr_df.abs()
        .stack()
        .sort_values(ascending=False)
        .head(10)
        .rename("Residuo_Estandarizado")
        .reset_index()
        .rename(columns={"level_0": "type1", "level_1": "rol"})
    )

    st.subheader("🔎 Principales contribuciones al χ² (|residuo| más alto)")
    st.dataframe(top, use_container_width=True)

    # --- Heatmap de residuos estandarizados ---
    st.subheader("🌡️ Heatmap de residuos estandarizados (type1 × rol)")
    fig_w = max(6, 0.6 * len(rol_levels))
    fig_h = max(6, 0.5 * len(type_levels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sr_df.values, aspect="auto", cmap="coolwarm")

    # Etiquetas
    ax.set_xticks(range(len(rol_levels)))
    ax.set_yticks(range(len(type_levels)))
    ax.set_xticklabels(rol_levels, rotation=45, ha="right")
    ax.set_yticklabels(type_levels)

    # Anotar valores (opcional: redondeado a 1 decimal)
    for i in range(sr_df.shape[0]):
        for j in range(sr_df.shape[1]):
            val = sr_df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", va="center", ha="center", fontsize=8)

    ax.set_xlabel("Rol competitivo")
    ax.set_ylabel("Tipo principal (type1)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Residuo estandarizado")

    st.pyplot(fig, clear_figure=True)

    st.caption(
        "Método: Chi² de independencia por permutación — α = 0.05 | Fuente: PokeAPI + Kaggle"
    )
    st.markdown("---")
    st.markdown("Carlos Ojea Sánchez, ***The Bridge***")
    st.markdown(
        "Hecho usando [Streamlit](https://streamlit.io) y [PokeAPI](https://pokeapi.co)."
    )
