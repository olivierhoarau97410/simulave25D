import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# --- Configuration PHYSIQUE ---
grid_size = 600  # Grille large pour permettre un dézoom massif
dt = 0.012
dx = 200.0 / grid_size  # taille physique d'une cellule en mètres
rho, g = 2600, 9.81

# --- Relief : Pente 13% vers la droite + bruit + crêtes/ravines ---
x = np.linspace(0, 200, grid_size)
y = np.linspace(0, 200, grid_size)
X, Y = np.meshgrid(x, y)

# Pente générale de 13% vers la droite (descente en X)
Z = 100 - 0.13 * X * (100 / 200 * grid_size / grid_size)
# Simplifié : pente 13% = 13m de dénivelé pour 100m horizontal
Z = 100 - 0.13 * X

# Bruit topographique (micro-relief réaliste)
np.random.seed(42)
from scipy.ndimage import gaussian_filter
bruit_fin = np.random.randn(grid_size, grid_size) * 1.5
bruit_fin = gaussian_filter(bruit_fin, sigma=8)  # lissé pour être réaliste
bruit_large = np.random.randn(grid_size, grid_size) * 3.0
bruit_large = gaussian_filter(bruit_large, sigma=25)  # ondulations plus larges
Z += bruit_fin + bruit_large

# Crêtes et ravines pour canaliser la lave
# Crête 1 : diagonale du haut-gauche vers le centre
Z += 4.0 * np.exp(-((Y - (80 + 0.3 * X)) ** 2) / 50)
# Crête 2 : parallèle plus bas
Z += 3.5 * np.exp(-((Y - (40 - 0.1 * X)) ** 2) / 40)
# Ravine centrale : un sillon entre les deux crêtes
Z -= 3.0 * np.exp(-((Y - (60 + 0.1 * X)) ** 2) / 60)

# Cône volcanique
cx, cy = 100, 100
cone_radius = 25
cone_height = 25
dist_cone = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
Z += np.maximum(0, cone_height * (1 - dist_cone / cone_radius))

h = np.zeros((grid_size, grid_size))
T = np.full((grid_size, grid_size), 1200.0)

# --- Paramètres d'injection automatique ---
debit_total = 50.0  # m3/s
fissure_x_m = 15.0
fissure_y_min_m = 133.0
fissure_y_max_m = 200.0
fissure_ix = int(fissure_x_m / 200.0 * grid_size)
fissure_iy_min = int(fissure_y_min_m / 200.0 * grid_size)
fissure_iy_max = int(fissure_y_max_m / 200.0 * grid_size)
n_fissure_cells = fissure_iy_max - fissure_iy_min
fissure_width = 1
cell_area = dx * dx
debit_par_cellule = debit_total * dt / (n_fissure_cells * fissure_width * cell_area)
debit_par_cellule = max(debit_par_cellule, 0.5)


def compute_flow_2d(h, Z, T):
    H_tot = Z + h
    grad_y, grad_x = np.gradient(H_tot, dx)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    tau_0 = 110 * np.exp(0.011 * (1200 - T))
    eta = 160 * np.exp(0.009 * (1200 - T))
    poussee = rho * g * h * slope
    v = np.where(poussee > tau_0, (h ** 2 / (3 * eta)) * (poussee - tau_0), 0)
    v = np.clip(v, 0, 0.6 * dx / dt)
    safe_slope = np.where(slope == 0, 1, slope)
    return -(grad_x / safe_slope) * v * h, -(grad_y / safe_slope) * v * h


# --- Fenêtre IDEES (pure matplotlib) ---
def show_idees(e):
    fig_idee = plt.figure("SIMUL_LAVE_2.5D - Idees", figsize=(10, 8), facecolor='#1a1a2e')
    ax_idee = fig_idee.add_axes([0.05, 0.02, 0.9, 0.96])
    ax_idee.set_facecolor('#1a1a2e')
    ax_idee.axis('off')

    # Titre principal
    ax_idee.text(0.5, 1.0, "SIMUL_LAVE_2.5D",
                 transform=ax_idee.transAxes, fontsize=18, fontweight='bold',
                 color='#ff4400', ha='center', va='top')

    # --- PRINCIPE PHYSIQUE ---
    ax_idee.text(0.5, 0.94, "--- PRINCIPE PHYSIQUE ---",
                 transform=ax_idee.transAxes, fontsize=13, fontweight='bold',
                 color='#00ccff', ha='center', va='top')

    physique = (
        "Navier-Stokes simplifie (couche mince) + rheologie de Bingham :\n"
        "  > La lave ne coule QUE si contrainte > seuil (tau_0)\n"
        "  > Vitesse = (h^2 / 3.eta) x (contrainte - tau_0)\n"
        "  > Poussee = rho x g x h x pente_locale\n"
        "Parametres : viscosite, seuil de Bingham, temperature (1200 C),\n"
        "pente du terrain, debit. Tous dependent de T !\n"
        "Refroidissement : fronts minces figent vite, coeur reste chaud."
    )
    ax_idee.text(0.5, 0.90, physique, transform=ax_idee.transAxes,
                 fontsize=8.5, fontfamily='monospace', color='#e0e0e0',
                 ha='center', va='top', linespacing=1.3)

    # --- IDEES DE MANIPULATION ---
    ax_idee.text(0.5, 0.58, "--- IDEES DE MANIPULATION ---",
                 transform=ax_idee.transAxes, fontsize=13, fontweight='bold',
                 color='#ffaa00', ha='center', va='top')

    manips = (
        "1. CLIQUEZ n'importe ou sur la carte pour creer une eruption !\n"
        "   Sur le cone -> devale | Ravine -> canalisee | Sommet -> radial\n"
        "2. Curseur DEBIT : 10 (fine) a 500 m3/s (nappe massive)\n"
        "3. Bouton INJ OFF : coupez la source, observez le figeage\n"
        "   progressif : front -> coeur -> FIGEE\n"
        "4. EPAISS / TEMP : comparez accumulation vs temperature\n"
        "5. Contournement du cone : faible debit = contourne,\n"
        "   fort debit = submerge !\n"
        "6. Creez un barrage de lave froide : cliquez plusieurs fois,\n"
        "   laissez refroidir, relancez -> observez la deviation\n"
        "7. Appreciez les heterogeneites de temperature le long de la\n"
        "   coulee : mode TEMP -> zones chaudes (coeur) vs froides (bords)"
    )
    ax_idee.text(0.5, 0.535, manips, transform=ax_idee.transAxes,
                 fontsize=8.5, fontfamily='monospace', color='#e0e0e0',
                 ha='center', va='top', linespacing=1.3)

    # --- LEGENDE ---
    ax_idee.text(0.5, 0.06, "--- LEGENDE ---",
                 transform=ax_idee.transAxes, fontsize=11, fontweight='bold',
                 color='#88cc88', ha='center', va='top')
    legende = (
        "Ligne rouge = fissure eruptive | Lignes blanches = courbes de niveau\n"
        "Titre : rouge = active | orange = ralentit | cyan = figee"
    )
    ax_idee.text(0.5, 0.025, legende, transform=ax_idee.transAxes,
                 fontsize=8.5, fontfamily='monospace', color='#e0e0e0',
                 ha='center', va='top', linespacing=1.3)

    fig_idee.show()


# --- Interface ---
fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
plt.subplots_adjust(bottom=0.18, top=0.90, left=0.06, right=0.92)
ax.set_facecolor('#111111')

# Titre principal
fig.suptitle('SIMUL_LAVE_2.5D', fontsize=18, fontweight='bold',
             color='#cc3300', y=0.97)

# Fond topographique en niveaux de gris (relief ombré)
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=35)
hillshade = ls.hillshade(Z, vert_exag=3)
ax.imshow(hillshade, cmap='gray', origin='lower', alpha=0.4, zorder=0)

# Courbes de niveau bien visibles avec labels d'altitude
contour_lines = ax.contour(Z, levels=30, colors='white', alpha=0.7, linewidths=1.0, zorder=3)
ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.0f m', colors='white')

# Image de la lave avec transparence
from matplotlib.colors import Normalize
import matplotlib.cm as cm
cmap_epaisseur = plt.cm.magma.copy()
cmap_temperature = plt.cm.hot.copy()
mode_visu = 'epaisseur'

h_rgba = np.zeros((grid_size, grid_size, 4))
im = ax.imshow(h_rgba, origin='lower', zorder=2, interpolation='gaussian')

# Colorbar
sm = cm.ScalarMappable(cmap='magma', norm=Normalize(vmin=0, vmax=5))
cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Épaisseur de la coulée (m)', fontsize=11)

# Marqueur de la fissure éruptive
ax.plot([fissure_ix] * 2, [fissure_iy_min, fissure_iy_max], 'r-', linewidth=1,
        alpha=0.7, zorder=3, label='Fissure éruptive')
ax.legend(loc='upper right', fontsize=9)

injection_active = True


def reset(e):
    global injection_active
    h.fill(0)
    T.fill(1200)
    injection_active = True
    btn_inj.label.set_text('INJ: ON')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)


def toggle_injection(e):
    global injection_active
    injection_active = not injection_active
    btn_inj.label.set_text('INJ: ON' if injection_active else 'INJ: OFF')


def toggle_visu(e):
    global mode_visu
    if mode_visu == 'epaisseur':
        mode_visu = 'temperature'
        btn_visu.label.set_text('TEMP / EPAISS')
        sm.set_cmap('hot')
        sm.set_norm(Normalize(vmin=300, vmax=1200))
        cb.update_normal(sm)
        cb.set_label('Température (°C)', fontsize=11)
    else:
        mode_visu = 'epaisseur'
        btn_visu.label.set_text('EPAISS / TEMP')
        sm.set_cmap('magma')
        sm.set_norm(Normalize(vmin=0, vmax=5))
        cb.update_normal(sm)
        cb.set_label('Épaisseur de la coulée (m)', fontsize=11)


# --- Widgets : disposition centrée sous le graphique ---
# Slider débit bien centré
ax_slider = plt.axes([0.20, 0.10, 0.50, 0.025])
slider_debit = Slider(ax_slider, 'Débit (m³/s)', 10, 500, valinit=50, valstep=10,
                       color='orangered')

# Boutons en ligne, centrés sous le slider
bw = 0.09  # largeur bouton
gap = 0.015
total = 4 * bw + 3 * gap
x0 = (1 - total) / 2  # centrage
by = 0.04  # position verticale
bh = 0.035

ax_res = plt.axes([x0, by, bw, bh])
btn_res = Button(ax_res, 'RESET', color='lightgray', hovercolor='silver')
btn_res.on_clicked(reset)

ax_inj = plt.axes([x0 + bw + gap, by, bw, bh])
btn_inj = Button(ax_inj, 'INJ: ON', color='lightyellow', hovercolor='yellow')
btn_inj.on_clicked(toggle_injection)

ax_visu = plt.axes([x0 + 2*(bw + gap), by, bw + 0.04, bh])
btn_visu = Button(ax_visu, 'EPAISS / TEMP', color='lightyellow', hovercolor='gold')
btn_visu.on_clicked(toggle_visu)

ax_idee = plt.axes([x0 + 2*(bw + gap) + bw + 0.04 + gap, by, bw, bh])
btn_idee = Button(ax_idee, 'IDEES', color='#ffe0cc', hovercolor='#ffaa66')
btn_idee.on_clicked(show_idees)


def on_click(event):
    if event.inaxes != ax:
        return
    ix, iy = int(event.xdata), int(event.ydata)
    y_idx, x_idx = np.ogrid[:grid_size, :grid_size]
    dist = np.sqrt((x_idx - ix) ** 2 + (y_idx - iy) ** 2)
    if 0 <= ix < grid_size and 0 <= iy < grid_size:
        h[dist < 8] += 4.5 * np.exp(-dist[dist < 8] ** 2 / 12)
        T[dist < 8] = 1200.0


fig.canvas.mpl_connect('button_press_event', on_click)


def update(frame):
    global h, T

    # --- Injection automatique le long de la fissure ---
    if injection_active:
        ratio = slider_debit.val / 50.0
        injection = debit_par_cellule * ratio
        h[fissure_iy_min:fissure_iy_max,
          fissure_ix:fissure_ix + fissure_width] += injection
        T[fissure_iy_min:fissure_iy_max,
          fissure_ix:fissure_ix + fissure_width] = 1200.0

    if np.max(h) < 0.01:
        return [im]

    fx, fy = compute_flow_2d(h, Z, T)
    h_new = h - dt * (np.gradient(fx, axis=1) + np.gradient(fy, axis=0))

    # Drainage aux bords
    h_new[0, :] = 0
    h_new[-1, :] = 0
    h_new[:, 0] = 0
    h_new[:, -1] = 0
    h_new = np.clip(h_new, 0, 15)

    # --- Refroidissement ---
    mask_lave = h_new > 0.05
    T[mask_lave] -= 0.04 / (h_new[mask_lave] + 0.3)

    bords_minces = (h_new > 0.05) & (h_new < 0.8)
    T[bords_minces] -= 0.15 / (h_new[bords_minces] + 0.1)

    from scipy.ndimage import uniform_filter
    fraction_lave = uniform_filter((h_new > 0.1).astype(float), size=5)
    contact_air = (fraction_lave < 0.7) & (h_new > 0.05)
    T[contact_air] -= 0.08

    T = np.clip(T, 300, 1200)
    h[:] = h_new

    # --- ZOOM DYNAMIQUE ---
    active = np.argwhere(h > 0.1)
    if active.size > 0:
        ymin, xmin = active.min(axis=0)
        ymax, xmax = active.max(axis=0)
        dx_fronts = xmax - xmin
        dy_fronts = ymax - ymin
        margin = max(50, max(dx_fronts, dy_fronts) * 0.3)
        ax.set_xlim(max(0, xmin - margin), min(grid_size, xmax + margin))
        ax.set_ylim(max(0, ymin - margin), min(grid_size, ymax + margin))

    # --- Détection coulée figée ---
    mask_active = h > 0.1
    if np.any(mask_active):
        T_moy = np.mean(T[mask_active])
        H_tot = Z + h
        grad_y_check, grad_x_check = np.gradient(H_tot, dx)
        slope_check = np.sqrt(grad_x_check**2 + grad_y_check**2)
        tau_0_check = 110 * np.exp(0.011 * (1200 - T))
        poussee_check = rho * g * h * slope_check
        en_mouvement = np.sum((poussee_check > tau_0_check) & mask_active)
        pct_mobile = 100 * en_mouvement / np.sum(mask_active)
        if pct_mobile < 2:
            etat = "FIGEE"
            couleur_etat = "cyan"
        elif pct_mobile < 30:
            etat = f"Ralentit ({pct_mobile:.0f}% mobile)"
            couleur_etat = "orange"
        else:
            etat = f"Active ({pct_mobile:.0f}% mobile)"
            couleur_etat = "red"
        ax.set_title(f"Débit: {int(slider_debit.val)} m³/s | T moy: {T_moy:.0f}°C | {etat}",
                     fontsize=12, fontweight='bold', color=couleur_etat)
    else:
        ax.set_title(f"Débit: {int(slider_debit.val)} m³/s | En attente... (cliquez ou activez la fissure)",
                     fontsize=12, fontweight='bold', color='gray')

    # Mise à jour image RGBA selon le mode de visualisation
    if mode_visu == 'epaisseur':
        norm = Normalize(vmin=0, vmax=5)
        rgba = cmap_epaisseur(norm(h))
    else:
        norm = Normalize(vmin=300, vmax=1200)
        rgba = cmap_temperature(norm(T))
    rgba[..., 3] = np.clip(h / 0.3, 0, 1)
    im.set_array(rgba)
    return [im]


ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
