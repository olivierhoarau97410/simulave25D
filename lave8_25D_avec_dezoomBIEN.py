import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LightSource
from scipy.ndimage import gaussian_filter, uniform_filter
import streamlit as st
import time

st.set_page_config(page_title="SIMUL_LAVE_2.5D", layout="wide")

# --- Titre ---
st.markdown("<h1 style='text-align:center; color:#cc3300;'>SIMUL_LAVE_2.5D</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Simulation d'ecoulement de lave — Navier-Stokes + Bingham</p>",
            unsafe_allow_html=True)

# --- Configuration PHYSIQUE ---
GRID = 200  # réduit pour performance Streamlit
DT = 0.012
DX = 200.0 / GRID
RHO, G = 2600, 9.81


@st.cache_data
def build_terrain():
    x = np.linspace(0, 200, GRID)
    y = np.linspace(0, 200, GRID)
    X, Y = np.meshgrid(x, y)
    Z = 100 - 0.13 * X

    np.random.seed(42)
    bruit_fin = gaussian_filter(np.random.randn(GRID, GRID) * 1.5, sigma=3)
    bruit_large = gaussian_filter(np.random.randn(GRID, GRID) * 3.0, sigma=8)
    Z += bruit_fin + bruit_large

    Z += 4.0 * np.exp(-((Y - (80 + 0.3 * X)) ** 2) / 50)
    Z += 3.5 * np.exp(-((Y - (40 - 0.1 * X)) ** 2) / 40)
    Z -= 3.0 * np.exp(-((Y - (60 + 0.1 * X)) ** 2) / 60)

    cx, cy = 100, 100
    dist_cone = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    Z += np.maximum(0, 25 * (1 - dist_cone / 25))

    ls = LightSource(azdeg=315, altdeg=35)
    hillshade = ls.hillshade(Z, vert_exag=3)
    return Z, hillshade


Z, hillshade = build_terrain()

# --- Fissure ---
fissure_ix = int(15.0 / 200.0 * GRID)
fissure_iy_min = int(133.0 / 200.0 * GRID)
fissure_iy_max = int(200.0 / 200.0 * GRID)
n_fissure = fissure_iy_max - fissure_iy_min
cell_area = DX * DX
debit_base = max(50.0 * DT / (n_fissure * cell_area), 0.5)

# --- Session state ---
if 'h' not in st.session_state:
    st.session_state.h = np.zeros((GRID, GRID))
    st.session_state.T = np.full((GRID, GRID), 1200.0)
    st.session_state.running = False
    st.session_state.step = 0


def compute_flow(h, Z, T):
    H_tot = Z + h
    grad_y, grad_x = np.gradient(H_tot, DX)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    tau_0 = 110 * np.exp(0.011 * (1200 - T))
    eta = 160 * np.exp(0.009 * (1200 - T))
    poussee = RHO * G * h * slope
    v = np.where(poussee > tau_0, (h ** 2 / (3 * eta)) * (poussee - tau_0), 0)
    v = np.clip(v, 0, 0.6 * DX / DT)
    safe_slope = np.where(slope == 0, 1, slope)
    return -(grad_x / safe_slope) * v * h, -(grad_y / safe_slope) * v * h


def step_simulation(h, T, debit, injection_on):
    if injection_on:
        ratio = debit / 50.0
        h[fissure_iy_min:fissure_iy_max, fissure_ix] += debit_base * ratio
        T[fissure_iy_min:fissure_iy_max, fissure_ix] = 1200.0

    if np.max(h) < 0.01:
        return h, T

    fx, fy = compute_flow(h, Z, T)
    h_new = h - DT * (np.gradient(fx, axis=1) + np.gradient(fy, axis=0))
    h_new[0, :] = 0
    h_new[-1, :] = 0
    h_new[:, 0] = 0
    h_new[:, -1] = 0
    h_new = np.clip(h_new, 0, 15)

    mask_lave = h_new > 0.05
    T[mask_lave] -= 0.04 / (h_new[mask_lave] + 0.3)
    bords = (h_new > 0.05) & (h_new < 0.8)
    T[bords] -= 0.15 / (h_new[bords] + 0.1)
    frac = uniform_filter((h_new > 0.1).astype(float), size=5)
    contact = (frac < 0.7) & (h_new > 0.05)
    T[contact] -= 0.08
    T = np.clip(T, 300, 1200)

    return h_new, T


def render(h, T, mode, Z, hillshade):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('#111111')

    ax.imshow(hillshade, cmap='gray', origin='lower', alpha=0.4, zorder=0)
    contour_lines = ax.contour(Z, levels=20, colors='white', alpha=0.7,
                                linewidths=0.8, zorder=3)
    ax.clabel(contour_lines, inline=True, fontsize=6, fmt='%.0f m', colors='white')

    if mode == 'Epaisseur':
        cmap = plt.cm.magma.copy()
        norm = Normalize(vmin=0, vmax=5)
        rgba = cmap(norm(h))
        label = 'Epaisseur (m)'
    else:
        cmap = plt.cm.hot.copy()
        norm = Normalize(vmin=300, vmax=1200)
        rgba = cmap(norm(T))
        label = 'Temperature (C)'

    rgba[..., 3] = np.clip(h / 0.3, 0, 1)
    ax.imshow(rgba, origin='lower', zorder=2, interpolation='gaussian')

    ax.plot([fissure_ix] * 2, [fissure_iy_min, fissure_iy_max], 'r-',
            linewidth=1, alpha=0.7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=10)

    active = np.argwhere(h > 0.1)
    if active.size > 0:
        ymin, xmin = active.min(axis=0)
        ymax, xmax = active.max(axis=0)
        margin = max(15, max(xmax - xmin, ymax - ymin) * 0.3)
        ax.set_xlim(max(0, xmin - margin), min(GRID, xmax + margin))
        ax.set_ylim(max(0, ymin - margin), min(GRID, ymax + margin))
    else:
        ax.set_xlim(0, GRID)
        ax.set_ylim(0, GRID)

    # Etat
    mask_active = h > 0.1
    if np.any(mask_active):
        T_moy = np.mean(T[mask_active])
        H_tot = Z + h
        gy, gx = np.gradient(H_tot, DX)
        sl = np.sqrt(gx**2 + gy**2)
        tau = 110 * np.exp(0.011 * (1200 - T))
        p = RHO * G * h * sl
        mobile = np.sum((p > tau) & mask_active)
        pct = 100 * mobile / np.sum(mask_active)
        if pct < 2:
            etat, col = "FIGEE", "cyan"
        elif pct < 30:
            etat, col = f"Ralentit ({pct:.0f}%)", "orange"
        else:
            etat, col = f"Active ({pct:.0f}%)", "red"
        ax.set_title(f"T moy: {T_moy:.0f} C | {etat}", fontsize=12,
                     fontweight='bold', color=col)

    plt.tight_layout()
    return fig


# --- Interface Streamlit ---
col_ctrl, col_view = st.columns([1, 3])

with col_ctrl:
    st.subheader("Controles")
    debit = st.slider("Debit (m3/s)", 10, 500, 50, step=10)
    mode = st.radio("Visualisation", ["Epaisseur", "Temperature"])
    injection_on = st.toggle("Injection fissure", value=True)

    st.markdown("---")
    st.subheader("Eruption manuelle")
    col_x, col_y = st.columns(2)
    with col_x:
        click_x = st.number_input("Position X", 0, GRID - 1, GRID // 2)
    with col_y:
        click_y = st.number_input("Position Y", 0, GRID - 1, GRID // 2)
    if st.button("Injecter lave ici"):
        y_idx, x_idx = np.ogrid[:GRID, :GRID]
        dist = np.sqrt((x_idx - click_x) ** 2 + (y_idx - click_y) ** 2)
        st.session_state.h[dist < 4] += 4.5 * np.exp(-dist[dist < 4] ** 2 / 6)
        st.session_state.T[dist < 4] = 1200.0

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        run = st.button("Simuler 50 pas")
    with col_b2:
        if st.button("RESET"):
            st.session_state.h = np.zeros((GRID, GRID))
            st.session_state.T = np.full((GRID, GRID), 1200.0)
            st.session_state.step = 0
            st.rerun()

    st.markdown("---")
    with st.expander("--- PRINCIPE PHYSIQUE ---"):
        st.markdown("""
**Navier-Stokes simplifie + rheologie de Bingham :**
- La lave ne coule QUE si contrainte > seuil (tau_0)
- Vitesse = (h^2 / 3.eta) x (contrainte - tau_0)
- Poussee = rho x g x h x pente_locale

**Parametres :** viscosite, seuil de Bingham, temperature (1200 C),
pente, debit. Tous dependent de T !

Refroidissement : fronts minces figent vite, coeur reste chaud.
        """)
    with st.expander("--- IDEES DE MANIPULATION ---"):
        st.markdown("""
1. **Injectez de la lave** ou vous voulez (coordonnees X/Y + bouton)
2. **Curseur debit** : 10 (fine) a 500 m3/s (nappe massive)
3. **Coupez l'injection** et observez le figeage progressif
4. **Epaisseur / Temperature** : comparez les deux modes
5. **Contournement du cone** : faible debit = contourne, fort = submerge
6. **Creez un barrage** : injectez, laissez refroidir, relancez
7. **Heterogeneites de temperature** : mode Temperature pour voir
   zones chaudes (coeur) vs froides (fronts, bords)
        """)

with col_view:
    plot_placeholder = st.empty()

    if run:
        for i in range(50):
            st.session_state.h, st.session_state.T = step_simulation(
                st.session_state.h, st.session_state.T, debit, injection_on)
            st.session_state.step += 1

            if i % 5 == 0:
                fig = render(st.session_state.h, st.session_state.T,
                            mode, Z, hillshade)
                plot_placeholder.pyplot(fig)
                plt.close(fig)

    fig = render(st.session_state.h, st.session_state.T, mode, Z, hillshade)
    plot_placeholder.pyplot(fig)
    plt.close(fig)

    st.caption(f"Pas de simulation : {st.session_state.step}")
