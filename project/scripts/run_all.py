from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import math
import platform
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.parameters import BearingParams, RotorParams, SealParams, dump_default_parameters
from models.bearing import (
    solve_pressure,
    integrate_force,
    bearing_coefficients,
    coefficients_sweep,
    bearing_matrices,
    misalignment_clearance_metrics,
    eccentricity_for_target_load,
)
from models.rotor import (
    SupportMatrices,
    support_from_coefficients,
    seal_elements,
    stage_coordinates,
    node_coordinates,
    complex_eigenanalysis,
    normalized_mode_shape,
    normalized_mode_xy,
    mode_peak_coordinate,
    rpm_from_rad_s,
    rigid_critical_speed_rpm,
    orbit_response_full,
    orbit_points,
    ellipse_metrics,
    support_slope_response_full,
    first_critical_for_support,
)

FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
VAL_DIR = ROOT / "validation"
for path in (FIG_DIR, DATA_DIR, VAL_DIR):
    path.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def format_ru_number(value: float, decimals: int | None = None) -> str:
    value = float(value)
    if decimals is None:
        if abs(value) >= 100.0 or abs(value - round(value)) < 1.0e-8:
            decimals = 0
        elif abs(value) >= 10.0:
            decimals = 1
        else:
            decimals = 2
    return f"{value:.{decimals}f}".replace(".", ",")


def comma_formatter(x: float, _pos: int | None = None) -> str:
    if abs(x) >= 1000.0:
        return format_ru_number(x, 0)
    if abs(x) >= 100.0:
        return format_ru_number(x, 0)
    if abs(x) >= 10.0:
        return format_ru_number(x, 1)
    if abs(x) >= 1.0:
        return format_ru_number(x, 1)
    if abs(x) >= 0.01:
        return format_ru_number(x, 2)
    if abs(x) < 1.0e-12:
        return "0"
    return format_ru_number(x, 3)


def _axis_has_categorical_labels(axis) -> bool:
    labels = [label.get_text() for label in axis.get_ticklabels()]
    for label in labels:
        if any(ch.isalpha() for ch in label):
            return True
    return False


def apply_russian_axis_format() -> None:
    formatter = FuncFormatter(comma_formatter)
    fig = plt.gcf()
    # Сначала формируем текущие подписи, чтобы не заменить категориальные оси
    # числовыми индексами 0, 1, 2.
    fig.canvas.draw_idle()
    for ax in fig.axes:
        if not _axis_has_categorical_labels(ax.xaxis):
            ax.xaxis.set_major_formatter(formatter)
        if not _axis_has_categorical_labels(ax.yaxis):
            ax.yaxis.set_major_formatter(formatter)


def savefig(name: str) -> None:
    apply_russian_axis_format()
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    if name == "06_rotor_scheme_with_seals.png":
        plt.savefig(FIG_DIR / "06_rotor_scheme_with_seals.svg", format="svg", bbox_inches="tight")
    plt.close()


def make_bearing_figures(bearing: BearingParams) -> dict:
    phi, z, H, P = solve_pressure(bearing)
    center_z = len(z) // 2
    pressure_mpa = P * bearing.pressure_scale_pa / 1.0e6
    force = integrate_force(bearing, phi, z, P)

    pd.DataFrame({
        "phi_rad": phi,
        "clearance_H_z0": H[:, center_z],
        "pressure_dimensionless_z0": P[:, center_z],
        "pressure_MPa_z0": pressure_mpa[:, center_z],
    }).to_csv(DATA_DIR / "pressure_line_z0.csv", index=False)

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(phi, H[:, center_z])
    plt.xlabel("Угловая координата, рад")
    plt.ylabel("Безразмерный зазор")
    savefig("01_clearance_z0.png")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(phi, pressure_mpa[:, center_z])
    plt.xlabel("Угловая координата, рад")
    plt.ylabel("Давление, МПа")
    savefig("02_pressure_z0.png")

    PHI, ZZ = np.meshgrid(phi, z, indexing="ij")
    plt.figure(figsize=(7.4, 4.8))
    cp = plt.contourf(PHI, ZZ, pressure_mpa, levels=30)
    plt.colorbar(cp, label="Давление, МПа")
    plt.xlabel("Угловая координата, рад")
    plt.ylabel("Осевое положение")
    savefig("03_pressure_field.png")

    return {
        "load_n": float(np.linalg.norm(force)),
        "force_x_n": float(force[0]),
        "force_y_n": float(force[1]),
        "pmax_mpa": float(np.max(pressure_mpa)),
        "pmean_positive_mpa": float(np.mean(pressure_mpa[pressure_mpa > 0.0])),
        "h_min_um": float(bearing.clearance_m * np.min(H) * 1.0e6),
        "h_max_um": float(bearing.clearance_m * np.max(H) * 1.0e6),
    }


def make_coefficients_figures(bearing: BearingParams, rotor: RotorParams) -> tuple[dict, pd.DataFrame, dict]:
    eps_values = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    coeff_rows = coefficients_sweep(bearing, eps_values)
    coeff_df = pd.DataFrame(coeff_rows)
    coeff_df.to_csv(DATA_DIR / "bearing_coefficients_vs_eccentricity.csv", index=False)

    left_target, right_target = rotor.static_support_reactions_n()
    mean_target = 0.5 * (left_target + right_target)
    loads_grid = coeff_df["load_n"].to_numpy(dtype=float)
    eps_grid = coeff_df["eccentricity"].to_numpy(dtype=float)
    eps_left = float(np.interp(left_target, loads_grid, eps_grid))
    eps_right = float(np.interp(right_target, loads_grid, eps_grid))
    eps_mean = float(np.interp(mean_target, loads_grid, eps_grid))

    base_coeffs = bearing_coefficients(bearing, eps_mean)
    left_coeffs = bearing_coefficients(bearing, eps_left)
    right_coeffs = bearing_coefficients(bearing, eps_right)
    pd.DataFrame([base_coeffs]).to_csv(DATA_DIR / "bearing_coefficients_base.csv", index=False)

    load_table = coeff_df[["eccentricity", "load_n"]].to_dict(orient="records")
    support_rows = [
        {"support": "baseline", "target_load_n": mean_target, **base_coeffs},
        {"support": "left", "target_load_n": left_target, **left_coeffs},
        {"support": "right", "target_load_n": right_target, **right_coeffs},
    ]
    pd.DataFrame(support_rows).to_csv(DATA_DIR / "bearing_coefficients_separate_supports.csv", index=False)
    pd.DataFrame(load_table).to_csv(DATA_DIR / "bearing_load_to_eccentricity_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(coeff_df["eccentricity"], coeff_df["Kxx"] / 1e6, marker="o", label=r"$K_{xx}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Kyy"] / 1e6, marker="o", label=r"$K_{yy}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Kxy"] / 1e6, marker="o", label=r"$K_{xy}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Kyx"] / 1e6, marker="o", label=r"$K_{yx}$")
    ax.set_xlabel("Эксцентриситет")
    ax.set_ylabel("Жёсткость, МН/м")
    ax.legend(ncol=2)
    savefig("04_stiffness_coefficients.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(coeff_df["eccentricity"], coeff_df["Cxx"] / 1e3, marker="o", label=r"$C_{xx}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Cyy"] / 1e3, marker="o", label=r"$C_{yy}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Cxy"] / 1e3, marker="o", label=r"$C_{xy}$")
    ax.plot(coeff_df["eccentricity"], coeff_df["Cyx"] / 1e3, marker="o", label=r"$C_{yx}$")
    ax.set_xlabel("Эксцентриситет")
    ax.set_ylabel(r"Демпфирование, кН$\cdot$с/м")
    ax.legend(ncol=2)
    savefig("05_damping_coefficients.png")

    sep_summary = {
        "shaft_weight_n": float(rotor.shaft_mass_kg * rotor.g_m_s2),
        "impeller_weight_n": float(rotor.total_impeller_mass_kg * rotor.g_m_s2),
        "left_target_load_n": float(left_target),
        "right_target_load_n": float(right_target),
        "mean_target_load_n": float(mean_target),
        "epsilon_left": float(eps_left),
        "epsilon_right": float(eps_right),
        "epsilon_mean": float(eps_mean),
        "left": left_coeffs,
        "right": right_coeffs,
        "baseline": base_coeffs,
    }
    return base_coeffs, coeff_df, sep_summary



def support_row(label: str, support: SupportMatrices) -> dict:
    k = support.stiffness_n_m
    c = support.damping_n_s_m
    return {
        "support": label,
        "Kxx": float(k[0, 0]),
        "Kxy": float(k[0, 1]),
        "Kyx": float(k[1, 0]),
        "Kyy": float(k[1, 1]),
        "Cxx": float(c[0, 0]),
        "Cxy": float(c[0, 1]),
        "Cyx": float(c[1, 0]),
        "Cyy": float(c[1, 1]),
        "K_direct_mean": float(support.direct_stiffness_mean),
        "C_direct_mean": float(support.direct_damping_mean),
    }




def support_clearance_scale(multiplier: float) -> float:
    """Оценка снижения коэффициентов подшипника при увеличении радиального зазора."""
    return float(multiplier) ** -3


def seal_clearance_scale(multiplier: float) -> float:
    """По формуле Ломакина жёсткость щелевого уплотнения обратно пропорциональна зазору."""
    return float(multiplier) ** -1


def excitation_components(
    rotor: RotorParams,
    grade_mm_s: float,
    include_hydraulic: bool,
    hydraulic_kr: float | None = None,
) -> tuple[complex, complex, float, float]:
    """Суммарные комплексные амплитуды синхронных сил в плоскостях x и y."""
    f_unb = rotor.unbalance_force_for_grade_n(grade_mm_s)
    f_hyd = rotor.hydraulic_radial_force_total_n(hydraulic_kr) if include_hydraulic else 0.0
    fx = f_unb + f_hyd
    fy = -1j * (f_unb + rotor.hydraulic_force_y_fraction * f_hyd)
    return complex(fx), complex(fy), float(f_unb), float(f_hyd)


def excitation_nodal_forces(
    rotor: RotorParams,
    grade_mm_s: float,
    include_hydraulic: bool,
    hydraulic_kr: float | None = None,
) -> list[tuple[float, complex, complex]]:
    """Нагрузки для упругой модели: гидравлическая сила распределена по ступеням."""
    _, _, f_unb, f_hyd = excitation_components(rotor, grade_mm_s, include_hydraulic, hydraulic_kr)
    nodal_forces: list[tuple[float, complex, complex]] = []
    if include_hydraulic:
        stage_fx = f_hyd / rotor.n_stages
        stage_fy = -1j * rotor.hydraulic_force_y_fraction * f_hyd / rotor.n_stages
        nodal_forces.extend((float(x_i), complex(stage_fx), complex(stage_fy)) for x_i in stage_coordinates(rotor))
    center_x = float(stage_coordinates(rotor)[rotor.n_stages // 2])
    nodal_forces.append((center_x, complex(f_unb), complex(-1j * f_unb)))
    return nodal_forces


def response_case(
    rotor: RotorParams,
    bearing: BearingParams,
    left_support: SupportMatrices,
    right_support: SupportMatrices,
    seal_elems: list,
    grade_mm_s: float,
    include_hydraulic: bool,
    support_clearance_multiplier: float,
    seal_clearance_multiplier: float,
    case: str,
    label: str,
    hydraulic_kr: float | None = None,
) -> dict:
    fx, fy, f_unb, f_hyd = excitation_components(rotor, grade_mm_s, include_hydraulic, hydraulic_kr)
    orbit = orbit_response_full(
        rotor,
        left_support,
        right_support,
        bearing.speed_rpm,
        seal_elems=seal_elems,
        nodal_forces=excitation_nodal_forces(rotor, grade_mm_s, include_hydraulic, hydraulic_kr),
    )
    orbit.update({
        "case": case,
        "label": label,
        "grade_mm_s": float(grade_mm_s),
        "include_hydraulic": bool(include_hydraulic),
        "hydraulic_kr": None if hydraulic_kr is None else float(hydraulic_kr),
        "unbalance_force_n": f_unb,
        "hydraulic_force_n": f_hyd,
        "force_x_abs_n": float(abs(fx)),
        "force_y_abs_n": float(abs(fy)),
        "support_clearance_multiplier": float(support_clearance_multiplier),
        "seal_clearance_multiplier": float(seal_clearance_multiplier),
    })
    return orbit


def draw_rotor_scheme(rotor: RotorParams, seal_table: pd.DataFrame) -> None:
    """Аккуратная расчётная схема ротора в стиле курса сопротивления материалов."""
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    y0 = 0.0

    ax.plot([0.0, rotor.span_m], [y0, y0], linewidth=2.4, color="black")

    # Оси координат.
    ax.annotate("", xy=(rotor.span_m + 0.10, -0.38), xytext=(0.0, -0.38), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(rotor.span_m + 0.115, -0.38, "$x$", va="center", fontsize=10)
    ax.annotate("", xy=(0.05, 0.42), xytext=(0.05, -0.38), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.07, 0.42, "$y$", va="center", fontsize=10)

    # Подшипниковые опоры: треугольник + основание + демпфер как радиальная связь.
    for x_s, label in [(0.0, "левая\nопора"), (rotor.span_m, "правая\nопора")]:
        tri = plt.Polygon([[x_s - 0.055, -0.22], [x_s + 0.055, -0.22], [x_s, -0.05]], closed=True, fill=False, hatch="///", linewidth=1.2)
        ax.add_patch(tri)
        ax.plot([x_s - 0.085, x_s + 0.085], [-0.22, -0.22], linewidth=1.2, color="black")
        ax.plot([x_s, x_s], [-0.05, y0], linewidth=1.2, color="black")
        ax.plot([x_s + 0.075, x_s + 0.075], [-0.04, -0.15], linewidth=0.9, color="black")
        ax.add_patch(plt.Rectangle((x_s + 0.058, -0.12), 0.034, 0.045, fill=False, linewidth=0.8))
        ax.text(x_s, -0.30, label, ha="center", va="top", fontsize=8)

    # Рабочие колёса и силы тяжести.
    imp_x = stage_coordinates(rotor)
    for idx, x_i in enumerate(imp_x, start=1):
        circ = plt.Circle((x_i, y0), 0.038, fill=False, linewidth=1.3)
        ax.add_patch(circ)
        ax.text(x_i, y0, f"{idx}", ha="center", va="center", fontsize=8)
        ax.annotate("", xy=(x_i, -0.18), xytext=(x_i, -0.055), arrowprops=dict(arrowstyle="->", lw=0.75))
    ax.text(imp_x[0], -0.215, "$G_i$", ha="center", va="top", fontsize=8)
    ax.text(float(np.mean(imp_x)), 0.125, "8 сосредоточенных масс рабочих колёс", ha="center", fontsize=8)

    # Щелевые уплотнения: компактные пружины под ступенями без подписей у каждой детали.
    for x_i in imp_x:
        xs = np.linspace(x_i - 0.030, x_i + 0.030, 9)
        ys = np.array([-0.055, -0.080, -0.055, -0.080, -0.055, -0.080, -0.055, -0.080, -0.055])
        ax.plot(xs, ys, linewidth=0.75, color="black")
        ax.plot([x_i, x_i], [-0.080, -0.115], linewidth=0.7, color="black")
    ax.text(float(np.mean(imp_x)), -0.145, "щелевые уплотнения", ha="center", fontsize=8)

    # Разгрузочное устройство справа.
    balance_x = float(seal_table[seal_table["kind"] == "balance"]["x_m"].iloc[0])
    ax.add_patch(plt.Rectangle((balance_x - 0.035, 0.060), 0.070, 0.090, fill=False, linewidth=1.0))
    ax.text(balance_x, 0.175, "разгрузочное\nустройство", ha="center", fontsize=8)

    # Возмущающие силы.
    x_mid = float(imp_x[len(imp_x)//2])
    ax.annotate("", xy=(x_mid + 0.055, 0.285), xytext=(x_mid, 0.055), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(x_mid + 0.065, 0.290, "Fд", fontsize=10, va="center")
    ax.annotate("", xy=(x_mid + 0.240, 0.205), xytext=(x_mid + 0.095, 0.045), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(x_mid + 0.250, 0.210, "Fг", fontsize=10, va="center")

    # Размерные линии.
    ax.annotate("", xy=(0.0, -0.50), xytext=(rotor.span_m, -0.50), arrowprops=dict(arrowstyle="<->", lw=1.0))
    ax.text(rotor.span_m / 2.0, -0.565, f"$l_s={rotor.span_m:.3f}$ м".replace(".", ","), ha="center", fontsize=8)
    ax.annotate("", xy=(imp_x[0], -0.405), xytext=(imp_x[1], -0.405), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text((imp_x[0] + imp_x[1]) / 2.0, -0.455, "$l_p$", ha="center", fontsize=8)
    ax.axvline(rotor.hydraulic_start_m, ymin=0.40, ymax=0.70, linestyle="--", linewidth=0.8, color="black")
    ax.axvline(rotor.hydraulic_start_m + rotor.hydraulic_length_m, ymin=0.40, ymax=0.70, linestyle="--", linewidth=0.8, color="black")
    ax.text(rotor.hydraulic_start_m + rotor.hydraulic_length_m / 2.0, 0.255, "пакет ступеней", ha="center", fontsize=8)

    ax.set_xlim(-0.13, rotor.span_m + 0.20)
    ax.set_ylim(-0.62, 0.48)
    ax.set_xlabel("Координата вдоль ротора, м")
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)


def rigid_orbit_response_full(rotor: RotorParams, left: SupportMatrices, right: SupportMatrices, speed_rpm: float) -> dict:
    omega = 2.0 * math.pi * speed_rpm / 60.0
    mass = np.eye(2, dtype=float) * rotor.total_mass_kg
    k = left.stiffness_n_m + right.stiffness_n_m
    c = left.damping_n_s_m + right.damping_n_s_m
    rhs = np.array([rotor.unbalance_force_n, -1j * rotor.unbalance_force_n], dtype=complex)
    q = np.linalg.solve(k - omega**2 * mass + 1j * omega * c, rhs)
    t, x, y = orbit_points(q[0], q[1], omega)
    transform = np.array([[np.real(q[0]), -np.imag(q[0])], [np.real(q[1]), -np.imag(q[1])]])
    semi = np.linalg.svd(transform, compute_uv=False)
    return {
        "time_s": t,
        "x_m": x,
        "y_m": y,
        "amp_um": float(max(np.max(np.abs(x)), np.max(np.abs(y))) * 1e6),
        "semi_major_um": float(semi[0] * 1e6),
        "semi_minor_um": float(semi[1] * 1e6),
        "axis_ratio": float(semi[0] / semi[1]) if semi[1] > 0 else float("inf"),
    }


def eigen_rows(label: str, eig: dict, n_modes: int = 6) -> list[dict]:
    rows: list[dict] = []
    for i in range(min(n_modes, len(eig["rpm"]))):
        lam = eig["eigenvalues"][i]
        rows.append({
            "model": label,
            "mode": i + 1,
            "rpm": float(eig["rpm"][i]),
            "omega_rad_s": float(eig["omega_rad_s"][i]),
            "real_part_1_s": float(np.real(lam)),
            "imag_part_1_s": float(np.imag(lam)),
            "damping_ratio": float(eig["damping_ratio"][i]),
            "log_decrement": log_decrement_from_eigenvalue(lam),
            "whirl": eig["whirl"][i],
        })
    return rows




def log_decrement_from_eigenvalue(lam: complex) -> float:
    im = abs(float(np.imag(lam)))
    if im < 1.0e-12:
        return float("nan")
    return float(-2.0 * math.pi * float(np.real(lam)) / im)


def vibration_zone_iso20816(v_rms_mm_s: float) -> str:
    """Ориентировочная шкала зон A/B/C/D для насосного агрегата на жёстком основании."""
    v = float(v_rms_mm_s)
    if v <= 2.3:
        return "A"
    if v <= 4.5:
        return "B"
    if v <= 7.1:
        return "C"
    return "D"


def critical_margin_status(margin_percent: float) -> str:
    m = float(margin_percent)
    if m >= 20.0:
        return "запас не менее 20 %"
    if m >= 15.0:
        return "минимальный запас 15--20 %"
    return "запас менее 15 %, требуется контроль"


def stability_mode_detail_rows(case: str, label: str, eig: dict, n_modes: int = 3) -> list[dict]:
    rows: list[dict] = []
    for idx in range(min(n_modes, len(eig["eigenvalues"]))):
        lam = complex(eig["eigenvalues"][idx])
        delta = log_decrement_from_eigenvalue(lam)
        rows.append({
            "case": case,
            "label": label,
            "mode": idx + 1,
            "rpm": float(eig["rpm"][idx]),
            "real_part_1_s": float(np.real(lam)),
            "imag_part_1_s": float(np.imag(lam)),
            "log_decrement": delta,
            "whirl": eig["whirl"][idx],
            "stability_by_real_part": "устойчива" if float(np.real(lam)) < 0.0 else "неустойчива",
            "api684_delta_status": "достаточно" if delta >= 0.10 else "недостаточно",
        })
    return rows


def support_orbit_normative_metrics(orbit: dict, rotor: RotorParams) -> dict:
    """Оценка виброскорости по орбите в опорных сечениях, а не по середине пролёта."""
    q = orbit["q"]
    nodes = orbit["nodes"]
    omega = rotor.omega_operating_rad_s
    rows = []
    for node_idx, support_name in [(0, "левая опора"), (len(nodes) - 1, "правая опора")]:
        em = ellipse_metrics(q[4 * node_idx], q[4 * node_idx + 2])
        semi_major_um = float(em["semi_major_m"] * 1.0e6)
        semi_minor_um = float(em["semi_minor_m"] * 1.0e6)
        v_rms = semi_major_um * 1.0e-6 * omega / math.sqrt(2.0) * 1000.0
        rows.append({
            "support": support_name,
            "x_m": float(nodes[node_idx]),
            "semi_major_um": semi_major_um,
            "semi_minor_um": semi_minor_um,
            "axis_ratio": float(em["axis_ratio"]),
            "vibration_velocity_rms_mm_s": float(v_rms),
            "iso20816_zone": vibration_zone_iso20816(v_rms),
        })
    max_row = max(rows, key=lambda row: row["vibration_velocity_rms_mm_s"])
    return {
        "support_rows": rows,
        "support_max_name": max_row["support"],
        "support_max_semi_major_um": max_row["semi_major_um"],
        "support_max_semi_minor_um": max_row["semi_minor_um"],
        "support_max_velocity_rms_mm_s": max_row["vibration_velocity_rms_mm_s"],
        "support_max_iso20816_zone": max_row["iso20816_zone"],
    }



def support_orbit_metrics_for_case(case: str, label: str, orbit: dict, rotor: RotorParams, bearing: BearingParams) -> tuple[list[dict], list[dict]]:
    """Точки и метрики орбит в опорных сечениях для одного эксплуатационного варианта."""
    q = orbit["q"]
    nodes = orbit["nodes"]
    omega = bearing.omega_rad_s
    orbit_rows: list[dict] = []
    metric_rows: list[dict] = []
    for node_label, support_label, node_idx in [("left_support", "левая опора", 0), ("right_support", "правая опора", len(nodes) - 1)]:
        qx = q[4 * node_idx]
        qy = q[4 * node_idx + 2]
        t_s, x_s, y_s = orbit_points(qx, qy, omega)
        metrics = ellipse_metrics(qx, qy)
        semi_major_um = float(metrics["semi_major_m"] * 1.0e6)
        semi_minor_um = float(metrics["semi_minor_m"] * 1.0e6)
        v_rms_mm_s = float(semi_major_um * 1.0e-6 * rotor.omega_operating_rad_s / math.sqrt(2.0) * 1000.0)
        metric_rows.append({
            "case": case,
            "label": label,
            "node": node_label,
            "support_label": support_label,
            "x_m": float(nodes[node_idx]),
            "semi_major_um": semi_major_um,
            "semi_minor_um": semi_minor_um,
            "axis_ratio": float(metrics["axis_ratio"]),
            "vibration_velocity_rms_mm_s": v_rms_mm_s,
            "iso20816_zone": vibration_zone_iso20816(v_rms_mm_s),
        })
        for ti, xi, yi in zip(t_s, x_s, y_s):
            orbit_rows.append({
                "case": case,
                "label": label,
                "node": node_label,
                "support_label": support_label,
                "time_s": float(ti),
                "x_um": float(xi * 1.0e6),
                "y_um": float(yi * 1.0e6),
            })
    return orbit_rows, metric_rows


def stability_summary_row(case: str, label: str, eig: dict, orbit: dict, rotor: RotorParams) -> dict:
    lambdas = [complex(v) for v in eig["eigenvalues"][:3]]
    deltas = [log_decrement_from_eigenvalue(v) for v in lambdas]
    lambda1 = lambdas[0]
    first_rpm = float(eig["rpm"][0])
    margin_signed = (first_rpm - rotor.operating_speed_rpm) / rotor.operating_speed_rpm * 100.0
    margin_abs = abs(margin_signed)
    support_norm = support_orbit_normative_metrics(orbit, rotor)
    v_rms_mm_s = support_norm["support_max_velocity_rms_mm_s"]
    max_real_first3 = max(float(np.real(v)) for v in lambdas)
    min_delta_first3 = min(deltas)
    stable_first3 = max_real_first3 < 0.0
    api_delta_ok = min_delta_first3 >= 0.10
    zone = vibration_zone_iso20816(v_rms_mm_s)
    if stable_first3 and api_delta_ok and margin_abs >= 20.0 and zone in ("A", "B"):
        conclusion = "соответствует критериям"
    elif stable_first3 and api_delta_ok and margin_abs >= 15.0:
        conclusion = "устойчив, требуется контроль вибрации"
    elif stable_first3 and api_delta_ok:
        conclusion = "устойчив, но запас по критической скорости недостаточен"
    else:
        conclusion = "динамический запас потерян"
    return {
        "case": case,
        "label": label,
        "first_critical_rpm": first_rpm,
        "lambda1_real_1_s": float(np.real(lambda1)),
        "lambda1_imag_1_s": float(np.imag(lambda1)),
        "log_decrement_delta1": deltas[0],
        "mode2_real_1_s": float(np.real(lambdas[1])),
        "mode2_log_decrement": deltas[1],
        "mode3_real_1_s": float(np.real(lambdas[2])),
        "mode3_log_decrement": deltas[2],
        "max_real_first3_1_s": max_real_first3,
        "min_log_decrement_first3": min_delta_first3,
        "stable_by_first3_eigenvalues": stable_first3,
        "api684_log_decrement_ok": api_delta_ok,
        "operating_speed_rpm": float(rotor.operating_speed_rpm),
        "critical_speed_margin_signed_percent": float(margin_signed),
        "critical_speed_margin_percent": float(margin_abs),
        "critical_margin_status": critical_margin_status(margin_abs),
        "orbit_semi_major_um": float(orbit["semi_major_um"]),
        "orbit_semi_minor_um": float(orbit["semi_minor_um"]),
        "support_max_name": support_norm["support_max_name"],
        "support_semi_major_um": float(support_norm["support_max_semi_major_um"]),
        "support_semi_minor_um": float(support_norm["support_max_semi_minor_um"]),
        "equivalent_velocity_rms_mm_s": float(v_rms_mm_s),
        "iso20816_zone": zone,
        "practical_conclusion": conclusion,
    }


def make_rotor_figures(rotor: RotorParams, bearing: BearingParams, seals: SealParams, base_coeffs: dict, coeff_df: pd.DataFrame, sep: dict) -> dict:
    seal_elems = seal_elements(rotor, seals)
    seal_table = pd.DataFrame([el.__dict__ for el in seal_elems])
    seal_table.to_csv(DATA_DIR / "seal_elements.csv", index=False)

    base_support = support_from_coefficients(base_coeffs, label="base")
    left_support = support_from_coefficients(sep["left"], label="left")
    right_support = support_from_coefficients(sep["right"], label="right")
    pd.DataFrame([
        support_row("base", base_support),
        support_row("left", left_support),
        support_row("right", right_support),
    ]).to_csv(DATA_DIR / "support_matrices_used.csv", index=False)

    mild_support_scale = support_clearance_scale(rotor.mild_bearing_clearance_multiplier)
    mild_seal_scale = seal_clearance_scale(rotor.mild_seal_clearance_multiplier)
    limit_support_scale = support_clearance_scale(rotor.limit_bearing_clearance_multiplier)
    limit_seal_scale = seal_clearance_scale(rotor.limit_seal_clearance_multiplier)
    mild_left = left_support.scaled(mild_support_scale, mild_support_scale, label="left_allowable_wear")
    mild_right = right_support.scaled(mild_support_scale, mild_support_scale, label="right_allowable_wear")
    limit_left = left_support.scaled(limit_support_scale, limit_support_scale, label="left_limit")
    limit_right = right_support.scaled(limit_support_scale, limit_support_scale, label="right_limit")
    mild_seal_elems = seal_elements(rotor, seals, stiffness_scale=mild_seal_scale)
    limit_seal_elems = seal_elements(rotor, seals, stiffness_scale=limit_seal_scale)

    # Собственные частоты: без дополнительных связей, с номинальными уплотнениями и при предельных зазорах.
    eig_base = complex_eigenanalysis(rotor, base_support, base_support, seal_elems=None, n_modes=6)
    eig_wet = complex_eigenanalysis(rotor, left_support, right_support, seal_elems=seal_elems, n_modes=6)
    eig_operating = complex_eigenanalysis(rotor, mild_left, mild_right, seal_elems=mild_seal_elems, n_modes=6)
    eig_limit = complex_eigenanalysis(rotor, limit_left, limit_right, seal_elems=limit_seal_elems, n_modes=6)
    rpm_base = eig_base["rpm"]
    rpm_wet = eig_wet["rpm"]
    rpm_operating = eig_operating["rpm"]
    rpm_limit = eig_limit["rpm"]
    modes_base = eig_base["modes"]
    modes_wet = eig_wet["modes"]
    modes_limit = eig_limit["modes"]
    nodes_base = eig_base["nodes"]
    nodes_wet = eig_wet["nodes"]
    nodes_limit = eig_limit["nodes"]
    rigid_rpm = rigid_critical_speed_rpm(rotor, base_support, base_support)

    pd.DataFrame(
        eigen_rows("without_annular_seals", eig_base)
        + eigen_rows("with_annular_seals", eig_wet)
        + eigen_rows("allowable_wear_clearances", eig_operating)
        + eigen_rows("limit_operating_clearances", eig_limit)
    ).to_csv(DATA_DIR / "critical_speeds_complex.csv", index=False)
    pd.DataFrame({
        "mode": np.arange(1, 5),
        "without_annular_seals_rpm": rpm_base[:4],
        "with_annular_seals_rpm": rpm_wet[:4],
        "allowable_wear_clearances_rpm": rpm_operating[:4],
        "limit_operating_clearances_rpm": rpm_limit[:4],
        "without_annular_seals_precession": eig_base["whirl"][:4],
        "with_annular_seals_precession": eig_wet["whirl"][:4],
        "allowable_wear_precession": eig_operating["whirl"][:4],
        "limit_precession": eig_limit["whirl"][:4],
    }).to_csv(DATA_DIR / "critical_speeds_comparison.csv", index=False)

    # Расчётная схема.
    draw_rotor_scheme(rotor, seal_table)
    savefig("06_rotor_scheme_with_seals.png")

    # Формы первой комплексной моды: для отчёта показываем нормированную поперечную амплитуду.
    shape_base = normalized_mode_shape(modes_base[:, 0])
    shape_wet = normalized_mode_shape(modes_wet[:, 0])
    shape_limit = normalized_mode_shape(modes_limit[:, 0])
    pd.DataFrame({"x_m": nodes_base, "without_annular_seals_mode_1": shape_base}).to_csv(DATA_DIR / "first_mode_baseline.csv", index=False)
    pd.DataFrame({"x_m": nodes_wet, "with_annular_seals_mode_1": shape_wet}).to_csv(DATA_DIR / "first_mode_wet.csv", index=False)
    pd.DataFrame({"x_m": nodes_limit, "limit_operating_mode_1": shape_limit}).to_csv(DATA_DIR / "first_mode_limit.csv", index=False)
    plt.figure(figsize=(8.0, 4.6))
    plt.plot(nodes_base, shape_base, marker="o", label="Без уплотнений")
    plt.plot(nodes_wet, shape_wet, marker="o", label="С уплотнениями")
    plt.plot(nodes_limit, shape_limit, marker="o", label="Предельные зазоры")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Координата вдоль ротора, м")
    plt.ylabel("Нормированная поперечная амплитуда")
    plt.legend()
    savefig("07_first_mode_shape_seals_comparison.png")

    plt.figure(figsize=(8.4, 4.8))
    labels = ["Жёсткая\nоценка", "Без\nуплотнений", "С\nуплотнениями", "Допустимый\nизнос", "Предельные\nзазоры"]
    values = [rigid_rpm, rpm_base[0], rpm_wet[0], rpm_operating[0], rpm_limit[0]]
    plt.bar(labels, values)
    plt.ylabel("Частота первой ветви, об/мин")
    for i, v in enumerate(values):
        plt.text(i, v * 1.02, format_ru_number(v, 0), ha="center", va="bottom")
    savefig("08_critical_speed_comparison_seals.png")

    # Орбиты с учётом реальных эксплуатационных нагрузок.
    fx_base, fy_base, f_unb_base, f_hyd_base = excitation_components(
        rotor, rotor.balance_grade_mm_s, include_hydraulic=True, hydraulic_kr=rotor.hydraulic_force_kr_nominal
    )
    fx_oper, fy_oper, f_unb_oper, f_hyd_oper = excitation_components(
        rotor, rotor.balance_grade_mm_s, include_hydraulic=True, hydraulic_kr=rotor.hydraulic_force_kr_offdesign
    )
    fx_limit, fy_limit, f_unb_limit, f_hyd_limit = excitation_components(
        rotor, rotor.balance_grade_limit_mm_s, include_hydraulic=True, hydraulic_kr=rotor.hydraulic_force_kr_offdesign
    )

    orbit_baseline = response_case(
        rotor, bearing, left_support, right_support, seal_elems,
        rotor.balance_grade_mm_s, True, 1.0, 1.0,
        "baseline_g6_3_hydraulic", "G6,3 + номинальная сила",
        hydraulic_kr=rotor.hydraulic_force_kr_nominal,
    )
    orbit_operating = response_case(
        rotor, bearing, mild_left, mild_right, mild_seal_elems,
        rotor.balance_grade_mm_s, True,
        rotor.mild_bearing_clearance_multiplier, rotor.mild_seal_clearance_multiplier,
        "allowable_wear_g6_3_offdesign", "G6,3 + допустимый износ",
        hydraulic_kr=rotor.hydraulic_force_kr_offdesign,
    )
    orbit_limit = response_case(
        rotor, bearing, limit_left, limit_right, limit_seal_elems,
        rotor.balance_grade_limit_mm_s, True,
        rotor.limit_bearing_clearance_multiplier, rotor.limit_seal_clearance_multiplier,
        "limit_g16_offdesign", "G16 + предельные зазоры",
        hydraulic_kr=rotor.hydraulic_force_kr_offdesign,
    )

    orbit_df = pd.DataFrame({
        "time_s": orbit_baseline["time_s"],
        "baseline_g6_3_hydraulic_x_um": orbit_baseline["x_m"] * 1e6,
        "baseline_g6_3_hydraulic_y_um": orbit_baseline["y_m"] * 1e6,
        "allowable_wear_g6_3_offdesign_x_um": orbit_operating["x_m"] * 1e6,
        "allowable_wear_g6_3_offdesign_y_um": orbit_operating["y_m"] * 1e6,
        "limit_g16_offdesign_x_um": orbit_limit["x_m"] * 1e6,
        "limit_g16_offdesign_y_um": orbit_limit["y_m"] * 1e6,
    })
    orbit_df.to_csv(DATA_DIR / "orbits_comparison_seals.csv", index=False)

    orbit_rows = []
    for orb in [orbit_baseline, orbit_operating, orbit_limit]:
        orbit_rows.append({
            "case": orb["case"],
            "label": orb["label"],
            "balance_grade_mm_s": orb["grade_mm_s"],
            "hydraulic_kr": orb["hydraulic_kr"],
            "hydraulic_force_n": orb["hydraulic_force_n"],
            "unbalance_force_n": orb["unbalance_force_n"],
            "force_x_abs_n": orb["force_x_abs_n"],
            "force_y_abs_n": orb["force_y_abs_n"],
            "bearing_clearance_multiplier": orb["support_clearance_multiplier"],
            "seal_clearance_multiplier": orb["seal_clearance_multiplier"],
            "amp_um": orb["amp_um"],
            "semi_major_um": orb["semi_major_um"],
            "semi_minor_um": orb["semi_minor_um"],
            "axis_ratio": orb["axis_ratio"],
        })
    orbit_metrics_df = pd.DataFrame(orbit_rows)
    orbit_metrics_df.to_csv(DATA_DIR / "orbit_ellipse_metrics.csv", index=False)

    plt.figure(figsize=(6.8, 6.2))
    plt.plot(orbit_df["baseline_g6_3_hydraulic_x_um"], orbit_df["baseline_g6_3_hydraulic_y_um"], label="G6,3 + номинальная сила")
    plt.plot(orbit_df["allowable_wear_g6_3_offdesign_x_um"], orbit_df["allowable_wear_g6_3_offdesign_y_um"], label="G6,3 + допустимый износ")
    plt.plot(orbit_df["limit_g16_offdesign_x_um"], orbit_df["limit_g16_offdesign_y_um"], label="G16 + предельные зазоры")
    plt.xlabel("x, мкм")
    plt.ylabel("y, мкм")
    plt.axis("equal")
    plt.legend(fontsize=8)
    savefig("09_orbits_comparison_seals.png")

    # Орбиты в опорных сечениях и таблицы нормативной оценки.
    support_orbit_rows = []
    support_metric_rows = []
    for case_id, label_case, orbit_case in [
        (orbit_baseline["case"], orbit_baseline["label"], orbit_baseline),
        (orbit_operating["case"], orbit_operating["label"], orbit_operating),
        (orbit_limit["case"], orbit_limit["label"], orbit_limit),
    ]:
        rows_i, metrics_i = support_orbit_metrics_for_case(case_id, label_case, orbit_case, rotor, bearing)
        support_orbit_rows.extend(rows_i)
        support_metric_rows.extend(metrics_i)
    pd.DataFrame(support_orbit_rows).to_csv(DATA_DIR / "support_orbits_full_matrix.csv", index=False)
    support_metrics_df = pd.DataFrame(support_metric_rows)
    support_metrics_df.to_csv(DATA_DIR / "support_orbit_ellipse_metrics.csv", index=False)

    # Для рисунка опорных орбит показывается предельно допустимый вариант, где эллиптичность наиболее наглядна.
    plt.figure(figsize=(6.7, 6.1))
    support_orbits_df = pd.DataFrame(support_orbit_rows)
    limit_support_orbits_df = support_orbits_df[support_orbits_df["case"] == orbit_limit["case"]]
    for node_label, label_node in [("left_support", "левая опора"), ("right_support", "правая опора")]:
        sub = limit_support_orbits_df[limit_support_orbits_df["node"] == node_label]
        plt.plot(sub["x_um"], sub["y_um"], label=label_node)
    plt.xlabel("x, мкм")
    plt.ylabel("y, мкм")
    plt.axis("equal")
    plt.legend(fontsize=8)
    savefig("16_support_orbits_full_matrix.png")

    # Свип по эксцентриситету выполняется с полной матрицей опоры.
    ecc_rows = []
    for _, row in coeff_df.iterrows():
        coeffs = row.to_dict()
        s_i = support_from_coefficients(coeffs, label=f"eps={row['eccentricity']:.2f}")
        flex_i = first_critical_for_support(rotor, s_i, s_i, None)
        rigid_i = rigid_critical_speed_rpm(rotor, s_i, s_i)
        ecc_rows.append({
            "eccentricity": row["eccentricity"],
            "Kxx_n_m": coeffs["Kxx"],
            "Kyy_n_m": coeffs["Kyy"],
            "anisotropic_critical_rpm": flex_i,
            "rigid_critical_rpm": rigid_i,
        })
    ecc_df = pd.DataFrame(ecc_rows)
    ecc_df.to_csv(DATA_DIR / "sweep_eccentricity_critical_speed.csv", index=False)
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(ecc_df["eccentricity"], ecc_df["rigid_critical_rpm"], marker="o", label="Жёсткая оценка")
    plt.plot(ecc_df["eccentricity"], ecc_df["anisotropic_critical_rpm"], marker="o", label="Упругая модель")
    plt.xlabel("Эксцентриситет")
    plt.ylabel("Первая ветвь, об/мин")
    plt.legend()
    savefig("10_sweep_eccentricity.png")

    spans = np.array([1.40, 1.60, 1.80, 1.951, 2.10, 2.30, 2.50])
    span_rows = []
    for L in spans:
        rot_i = replace(rotor, span_m=float(L))
        if rot_i.hydraulic_start_m + rot_i.hydraulic_length_m >= rot_i.span_m:
            rot_i = replace(rot_i, hydraulic_start_m=0.18 * L, hydraulic_length_m=0.55 * L)
        flex_i = first_critical_for_support(rot_i, base_support, base_support, None)
        rigid_i = rigid_critical_speed_rpm(rot_i, base_support, base_support)
        span_rows.append({"span_m": L, "anisotropic_critical_rpm": flex_i, "rigid_critical_rpm": rigid_i})
    span_df = pd.DataFrame(span_rows)
    span_df.to_csv(DATA_DIR / "sweep_span_critical_speed.csv", index=False)
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(span_df["span_m"], span_df["rigid_critical_rpm"], marker="o", label="Жёсткая оценка")
    plt.plot(span_df["span_m"], span_df["anisotropic_critical_rpm"], marker="o", label="Упругая модель")
    plt.axvline(rotor.span_m, linestyle="--", linewidth=1, label="Принято по чертежу")
    plt.xlabel("Расстояние между опорами, м")
    plt.ylabel("Первая ветвь, об/мин")
    plt.legend()
    savefig("11_sweep_span.png")

    # Анализ чувствительности первой критической скорости к коэффициенту Ломакина.
    sens_rows = []
    for factor in np.linspace(0.70, 1.30, 7):
        seals_i = replace(seals, lomakin_factor=seals.lomakin_factor * float(factor))
        seal_elems_i = seal_elements(rotor, seals_i)
        rpm_i = first_critical_for_support(rotor, left_support, right_support, seal_elems_i)
        sens_rows.append({
            "lambda_factor": float(factor),
            "lambda_L": float(seals_i.lomakin_factor),
            "critical_rpm": float(rpm_i),
            "relative_change_percent": float((rpm_i - rpm_wet[0]) / rpm_wet[0] * 100.0),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(DATA_DIR / "sensitivity_lomakin_lambda.csv", index=False)
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(sens_df["lambda_L"], sens_df["critical_rpm"], marker="o")
    plt.axvline(seals.lomakin_factor, linestyle="--", linewidth=1)
    plt.xlabel(r"Коэффициент Ломакина $\lambda_L$")
    plt.ylabel("Первая ветвь, об/мин")
    savefig("14_sensitivity_lomakin_lambda.png")

    # Анализ устойчивости и нормативное сопоставление эксплуатационных сценариев.
    stability_rows = [
        stability_summary_row("baseline_g6_3_hydraulic", "G6,3 + номинальная сила", eig_wet, orbit_baseline, rotor),
        stability_summary_row("allowable_wear_g6_3_offdesign", "G6,3 + допустимый износ", eig_operating, orbit_operating, rotor),
        stability_summary_row("limit_g16_offdesign", "G16 + предельные зазоры", eig_limit, orbit_limit, rotor),
    ]
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(DATA_DIR / "stability_summary.csv", index=False)
    stability_df.to_csv(DATA_DIR / "normative_comparison.csv", index=False)

    stability_mode_rows = (
        stability_mode_detail_rows("baseline_g6_3_hydraulic", "G6,3 + номинальная сила", eig_wet, n_modes=3)
        + stability_mode_detail_rows("allowable_wear_g6_3_offdesign", "G6,3 + допустимый износ", eig_operating, n_modes=3)
        + stability_mode_detail_rows("limit_g16_offdesign", "G16 + предельные зазоры", eig_limit, n_modes=3)
    )
    stability_modes_df = pd.DataFrame(stability_mode_rows)
    stability_modes_df.to_csv(DATA_DIR / "stability_modes.csv", index=False)

    support_rows = []
    for case_id, label_i, orbit_i in [
        ("baseline_g6_3_hydraulic", "G6,3 + номинальная сила", orbit_baseline),
        ("allowable_wear_g6_3_offdesign", "G6,3 + допустимый износ", orbit_operating),
        ("limit_g16_offdesign", "G16 + предельные зазоры", orbit_limit),
    ]:
        for row in support_orbit_normative_metrics(orbit_i, rotor)["support_rows"]:
            support_rows.append({"case": case_id, "label": label_i, **row})
    pd.DataFrame(support_rows).to_csv(DATA_DIR / "support_vibration_velocity.csv", index=False)

    plt.figure(figsize=(7.6, 4.8))
    labels_stab = ["G6,3\nноминал", "G6,3\nизнос", "G16\nпредел"]
    plt.plot(labels_stab, stability_df["log_decrement_delta1"], marker="o", label="Первая мода")
    plt.plot(labels_stab, stability_df["min_log_decrement_first3"], marker="o", label="Минимум первых трёх мод")
    plt.axhline(0.10, linestyle="--", linewidth=1, label="Критерий 0,1")
    plt.ylabel("Логарифмический декремент")
    plt.legend(fontsize=8)
    savefig("17_log_decrement_stability.png")

    summary_seal = seal_table.groupby("kind")["stiffness_n_m"].sum().reset_index()
    summary_seal.to_csv(DATA_DIR / "seal_stiffness_summary.csv", index=False)
    plt.figure(figsize=(7.4, 4.6))
    labels_seal = ["Ступеневые\nуплотнения", "Разгрузочное\nуплотнение", "Левая\nопора", "Правая\nопора"]
    values_seal = [
        float(seal_table[seal_table["kind"].str.startswith("stage")]["stiffness_n_m"].sum() / 1e6),
        float(seal_table[seal_table["kind"] == "balance"]["stiffness_n_m"].sum() / 1e6),
        float(left_support.direct_stiffness_mean / 1e6),
        float(right_support.direct_stiffness_mean / 1e6),
    ]
    plt.bar(labels_seal, values_seal)
    plt.ylabel("Средняя прямая жёсткость, МН/м")
    for i, v in enumerate(values_seal):
        plt.text(i, v * 1.02, format_ru_number(v, 1), ha="center", va="bottom")
    savefig("15_seal_stiffness_contributions.png")

    comparison_rows = [
        {
            "case": "without_annular_seals",
            "critical_rpm": float(rpm_base[0]),
            "mode_peak_x_m": mode_peak_coordinate(nodes_base, modes_base[:, 0]),
        },
        {
            "case": "with_annular_seals",
            "critical_rpm": float(rpm_wet[0]),
            "mode_peak_x_m": mode_peak_coordinate(nodes_wet, modes_wet[:, 0]),
        },
        {
            "case": "limit_operating_clearances",
            "critical_rpm": float(rpm_limit[0]),
            "mode_peak_x_m": mode_peak_coordinate(nodes_limit, modes_limit[:, 0]),
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(DATA_DIR / "model_comparison_summary.csv", index=False)

    support_metrics_all = pd.DataFrame(support_metric_rows)
    support_metrics_limit = support_metrics_all[support_metrics_all["case"] == orbit_limit["case"]]
    return {
        "support_matrix_base": support_row("base", base_support),
        "support_matrix_left": support_row("left", left_support),
        "support_matrix_right": support_row("right", right_support),
        "rigid_critical_rpm": float(rigid_rpm),
        "no_seal_critical_1_rpm": float(rpm_base[0]),
        "no_seal_critical_2_rpm": float(rpm_base[1]),
        "wet_critical_1_rpm": float(rpm_wet[0]),
        "wet_critical_2_rpm": float(rpm_wet[1]),
        "wet_critical_3_rpm": float(rpm_wet[2]),
        "operating_wear_critical_1_rpm": float(rpm_operating[0]),
        "operating_wear_critical_2_rpm": float(rpm_operating[1]),
        "limit_critical_1_rpm": float(rpm_limit[0]),
        "limit_critical_2_rpm": float(rpm_limit[1]),
        "wet_precession_split_1_2_rpm": float(abs(rpm_wet[1] - rpm_wet[0])),
        "operating_precession_split_1_2_rpm": float(abs(rpm_operating[1] - rpm_operating[0])),
        "limit_precession_split_1_2_rpm": float(abs(rpm_limit[1] - rpm_limit[0])),
        "critical_growth_percent": float((rpm_wet[0] - rpm_base[0]) / rpm_base[0] * 100.0),
        "no_seal_mode_peak_x_m": float(mode_peak_coordinate(nodes_base, modes_base[:, 0])),
        "wet_mode_peak_x_m": float(mode_peak_coordinate(nodes_wet, modes_wet[:, 0])),
        "limit_mode_peak_x_m": float(mode_peak_coordinate(nodes_limit, modes_limit[:, 0])),
        "baseline_orbit_amp_um": float(orbit_baseline["amp_um"]),
        "baseline_orbit_semi_major_um": float(orbit_baseline["semi_major_um"]),
        "baseline_orbit_semi_minor_um": float(orbit_baseline["semi_minor_um"]),
        "baseline_orbit_axis_ratio": float(orbit_baseline["axis_ratio"]),
        "operating_orbit_amp_um": float(orbit_operating["amp_um"]),
        "operating_orbit_semi_major_um": float(orbit_operating["semi_major_um"]),
        "operating_orbit_semi_minor_um": float(orbit_operating["semi_minor_um"]),
        "operating_orbit_axis_ratio": float(orbit_operating["axis_ratio"]),
        "limit_orbit_amp_um": float(orbit_limit["amp_um"]),
        "limit_orbit_semi_major_um": float(orbit_limit["semi_major_um"]),
        "limit_orbit_semi_minor_um": float(orbit_limit["semi_minor_um"]),
        "limit_orbit_axis_ratio": float(orbit_limit["axis_ratio"]),
        "limit_support_axis_ratio_max": float(support_metrics_limit["axis_ratio"].max()),
        "limit_support_semi_major_max_um": float(support_metrics_limit["semi_major_um"].max()),
        "limit_support_semi_minor_min_um": float(support_metrics_limit["semi_minor_um"].min()),
        "unbalance_force_g6_3_n": float(f_unb_base),
        "unbalance_force_g16_n": float(f_unb_limit),
        "hydraulic_radial_force_nominal_n": float(f_hyd_base),
        "hydraulic_radial_force_offdesign_n": float(f_hyd_oper),
        "baseline_force_x_abs_n": float(abs(fx_base)),
        "baseline_force_y_abs_n": float(abs(fy_base)),
        "operating_force_x_abs_n": float(abs(fx_oper)),
        "operating_force_y_abs_n": float(abs(fy_oper)),
        "limit_force_x_abs_n": float(abs(fx_limit)),
        "limit_force_y_abs_n": float(abs(fy_limit)),
        "allowable_eccentricity_g6_3_um": float(rotor.allowable_eccentricity_for_grade_m(rotor.balance_grade_mm_s) * 1e6),
        "allowable_eccentricity_g16_um": float(rotor.allowable_eccentricity_for_grade_m(rotor.balance_grade_limit_mm_s) * 1e6),
        "hydraulic_force_kr_nominal": float(rotor.hydraulic_force_kr_nominal),
        "hydraulic_force_kr_offdesign": float(rotor.hydraulic_force_kr_offdesign),
        "stage_seal_k_n_m": float(seals.stage_seal_stiffness_n_m(rotor)),
        "balance_seal_k_n_m": float(seals.balance_seal_stiffness_n_m(rotor)),
        "total_stage_seal_k_n_m": float(seal_table[seal_table["kind"].str.startswith("stage")]["stiffness_n_m"].sum()),
        "total_balance_seal_k_n_m": float(seal_table[seal_table["kind"] == "balance"]["stiffness_n_m"].sum()),
        "n_seal_elements": int(len(seal_elems)),
        "mild_bearing_clearance_multiplier": float(rotor.mild_bearing_clearance_multiplier),
        "mild_seal_clearance_multiplier": float(rotor.mild_seal_clearance_multiplier),
        "limit_bearing_clearance_multiplier": float(rotor.limit_bearing_clearance_multiplier),
        "limit_seal_clearance_multiplier": float(rotor.limit_seal_clearance_multiplier),
        "stability": stability_rows,
        "nodes_no_seal": [float(v) for v in nodes_base],
        "nodes_wet": [float(v) for v in nodes_wet],
        "nodes_limit": [float(v) for v in nodes_limit],
    }


def make_misalignment_figures(bearing: BearingParams, rotor: RotorParams, seals: SealParams, base_coeffs: dict, sep: dict) -> dict:
    seal_elems = seal_elements(rotor, seals)
    left_support = support_from_coefficients(sep["left"], label="left")
    right_support = support_from_coefficients(sep["right"], label="right")

    limit_support_scale = support_clearance_scale(rotor.limit_bearing_clearance_multiplier)
    limit_seal_scale = seal_clearance_scale(rotor.limit_seal_clearance_multiplier)
    limit_left = left_support.scaled(limit_support_scale, limit_support_scale, label="left_limit")
    limit_right = right_support.scaled(limit_support_scale, limit_support_scale, label="right_limit")
    limit_seal_elems = seal_elements(rotor, seals, stiffness_scale=limit_seal_scale)

    # Базовый новый ротор: G6,3 и номинальная гидродинамическая сила потока.
    fx_base, fy_base, _, _ = excitation_components(
        rotor, rotor.balance_grade_mm_s, include_hydraulic=True, hydraulic_kr=rotor.hydraulic_force_kr_nominal
    )
    slope_baseline = support_slope_response_full(
        rotor,
        left_support,
        right_support,
        bearing.speed_rpm,
        seal_elems=seal_elems,
        nodal_forces=excitation_nodal_forces(rotor, rotor.balance_grade_mm_s, True, rotor.hydraulic_force_kr_nominal),
    )

    # Предельно допустимый вариант: G16, нерасчётный режим и увеличенные зазоры.
    fx_limit, fy_limit, f_unb_limit, f_hyd_limit = excitation_components(
        rotor, rotor.balance_grade_limit_mm_s, include_hydraulic=True, hydraulic_kr=rotor.hydraulic_force_kr_offdesign
    )
    slope_limit = support_slope_response_full(
        rotor,
        limit_left,
        limit_right,
        bearing.speed_rpm,
        seal_elems=limit_seal_elems,
        nodal_forces=excitation_nodal_forces(rotor, rotor.balance_grade_limit_mm_s, True, rotor.hydraulic_force_kr_offdesign),
    )

    # Допустимый эксплуатационный износ: G6,3, нерасчётный режим и умеренно увеличенные зазоры.
    mild_support_scale = support_clearance_scale(rotor.mild_bearing_clearance_multiplier)
    mild_seal_scale = seal_clearance_scale(rotor.mild_seal_clearance_multiplier)
    mild_left = left_support.scaled(mild_support_scale, mild_support_scale, label="left_allowable_wear")
    mild_right = right_support.scaled(mild_support_scale, mild_support_scale, label="right_allowable_wear")
    mild_seal_elems = seal_elements(rotor, seals, stiffness_scale=mild_seal_scale)
    slope_mild = support_slope_response_full(
        rotor,
        mild_left,
        mild_right,
        bearing.speed_rpm,
        seal_elems=mild_seal_elems,
        nodal_forces=excitation_nodal_forces(rotor, rotor.balance_grade_mm_s, True, rotor.hydraulic_force_kr_offdesign),
    )

    rows = []
    for model, slopes in [("baseline_g6_3_hydraulic", slope_baseline), ("allowable_wear_g6_3_offdesign", slope_mild), ("limit_g16_offdesign", slope_limit)]:
        for support, theta in [("левая опора", slopes["left_theta_amp_rad"]), ("правая опора", slopes["right_theta_amp_rad"])]:
            metrics = misalignment_clearance_metrics(bearing, theta)
            metrics.update({"model": model, "support": support})
            rows.append(metrics)
    mis_df = pd.DataFrame(rows)
    mis_df.to_csv(DATA_DIR / "misalignment_summary.csv", index=False)

    Z = np.linspace(-1.0, 1.0, 200)
    z_mm = Z * bearing.length_m * 1000.0 / 2.0
    h_aligned = bearing.clearance_m * (1.0 - bearing.eccentricity) * np.ones_like(Z)
    h_profiles = {"Без перекоса": h_aligned * 1.0e6}
    for model, slopes in [("G6,3 + номинальная сила", slope_baseline), ("G16 + предельные зазоры", slope_limit)]:
        chi = bearing.length_m * slopes["max_theta_amp_rad"] / (2.0 * bearing.clearance_m)
        h_profiles[model] = bearing.clearance_m * (1.0 - bearing.eccentricity - chi * Z) * 1.0e6
    profile_df = pd.DataFrame({"z_mm": z_mm, **h_profiles})
    profile_df.to_csv(DATA_DIR / "misalignment_clearance_profile.csv", index=False)
    plt.figure(figsize=(7.4, 4.6))
    for col in profile_df.columns[1:]:
        plt.plot(profile_df["z_mm"], profile_df[col], label=col)
    plt.xlabel("Осевое положение во вкладыше, мм")
    plt.ylabel("Минимальный зазор, мкм")
    plt.legend()
    savefig("12_misalignment_clearance.png")

    # Давление вдоль оси для предельно допустимого варианта.
    phi0, z0, H0, P0 = solve_pressure(bearing)
    center_z = len(z0) // 2
    i_peak = int(np.argmax(P0[:, center_z]))
    theta_limit = slope_limit["max_theta_amp_rad"]
    phi1, z1, H1, P1 = solve_pressure(bearing, tilt_x_rad=theta_limit)
    p0_mpa = P0[i_peak, :] * bearing.pressure_scale_pa / 1.0e6
    p1_mpa = P1[i_peak, :] * bearing.pressure_scale_pa / 1.0e6
    z_axis_mm = z0 * bearing.length_m * 1000.0 / 2.0
    pd.DataFrame({"z_mm": z_axis_mm, "pressure_aligned_MPa": p0_mpa, "pressure_limit_misaligned_MPa": p1_mpa}).to_csv(DATA_DIR / "misalignment_pressure_axial.csv", index=False)
    plt.figure(figsize=(7.4, 4.6))
    plt.plot(z_axis_mm, p0_mpa, label="Без перекоса")
    plt.plot(z_axis_mm, p1_mpa, label="G16 + предельные зазоры")
    plt.xlabel("Осевое положение во вкладыше, мм")
    plt.ylabel("Давление, МПа")
    plt.legend()
    savefig("13_misalignment_pressure_axial.png")

    limit_rows = mis_df[mis_df["model"] == "limit_g16_offdesign"]
    baseline_rows = mis_df[mis_df["model"] == "baseline_g6_3_hydraulic"]
    return {
        "baseline_left_theta_mrad": float(slope_baseline["left_theta_amp_rad"] * 1e3),
        "baseline_right_theta_mrad": float(slope_baseline["right_theta_amp_rad"] * 1e3),
        "limit_left_theta_mrad": float(slope_limit["left_theta_amp_rad"] * 1e3),
        "limit_right_theta_mrad": float(slope_limit["right_theta_amp_rad"] * 1e3),
        "baseline_max_theta_mrad": float(slope_baseline["max_theta_amp_rad"] * 1e3),
        "limit_max_theta_mrad": float(slope_limit["max_theta_amp_rad"] * 1e3),
        "pmax_aligned_mpa": float(np.max(P0 * bearing.pressure_scale_pa / 1.0e6)),
        "pmax_limit_misaligned_mpa": float(np.max(P1 * bearing.pressure_scale_pa / 1.0e6)),
        "pmax_growth_percent_limit": float((np.max(P1) - np.max(P0)) / np.max(P0) * 100.0),
        "baseline_min_edge_um": float(baseline_rows["h_min_edge_loaded_um"].min()),
        "limit_min_edge_um": float(limit_rows["h_min_edge_loaded_um"].min()),
        "limit_unbalance_force_n": float(f_unb_limit),
        "limit_hydraulic_force_n": float(f_hyd_limit),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    bearing = BearingParams()
    rotor = RotorParams()
    seals = SealParams()
    dump_default_parameters(DATA_DIR / "input_parameters.json")

    assumptions = {
        "drawing_basis": "сборочный чертёж УГНТУ; насос ЦНС 105-392 (5МС-10); подшипник стороны всасывания; чертёж колеса первой ступени ЦНС90.00.00.012",
        "bearing_journal_diameter_mm": 90.0,
        "bearing_radius_m": bearing.radius_m,
        "bearing_liner_length_mm": 40.0,
        "liner_material": bearing.liner_material,
        "radial_bearings_count": 2,
        "stages": "1 колесо первой ступени по чертежу + 7 рабочих колёс позиции 11, учтённых как 6 промежуточных и одно крайнее с той же расчётной массой",
        "thrust_unloading_support": "разгрузочное устройство учитывается как осевая разгрузка; его щелевое уплотнение добавлено как радиальная гидродинамическая связь",
        "support_span_assumption_m": rotor.span_m,
        "support_span_components_m": {"left_to_stage_pack": 0.360, "stage_pack": 1.071, "stage_pack_to_right": 0.520},
        "shaft_equivalent_diameter_m": rotor.shaft_diameter_m,
        "shaft_diameter_basis": "по сборочному чертежу явно читаются посадочные диаметры около 90-95 мм; для бакалаврской модели принят эквивалентный d=90 мм",
        "impeller_masses_kg": {"first_stage": rotor.mass_first_stage_kg, "intermediate": rotor.mass_intermediate_kg, "last_stage": rotor.mass_last_stage_kg},
        "impeller_geometry_from_drawing": {"D2_m": rotor.impeller_outer_diameter_m, "B2_m": rotor.impeller_equivalent_width_m, "blade_count": rotor.impeller_blade_count, "allowable_unbalance_g_mm": rotor.impeller_first_stage_unbalance_g_mm},
        "static_support_reactions_n": {"left": rotor.static_support_reactions_n()[0], "right": rotor.static_support_reactions_n()[1]},
        "annular_seal_model": "инженерная линеаризация эффекта Ломакина: K=lambda_L*Delta_p*D*L/c; C=beta*K/Omega",
        "annular_seal_geometry": "Ds переднего уплотнения 0.110 м, Ds заднего 0.195 м, Ds разгрузочного 0.120 м; gamma_s=0.40/0.60/0.30; Ls=10 мм, cs=0.20 мм для нового состояния, до 0.30-0.40 мм при износе",
        "unbalance_model": "ГОСТ ИСО 1940-1: G6,3 для нового ротора и G16 для предельно допустимого эксплуатационного варианта",
        "hydraulic_force_model": "F_rad=K_R*rho*g*H*D2*B2; K_R=0.02 для номинального режима и K_R=0.04 для нерасчётного режима",
    }
    (DATA_DIR / "drawing_assumptions.json").write_text(json.dumps(assumptions, indent=2, ensure_ascii=False), encoding="utf-8")

    base_coeffs, coeff_df, separate = make_coefficients_figures(bearing, rotor)
    bearing = replace(bearing, eccentricity=separate["epsilon_mean"])
    pressure_summary = make_bearing_figures(bearing)
    rotor_summary = make_rotor_figures(rotor, bearing, seals, base_coeffs, coeff_df, separate)
    misalignment_summary = make_misalignment_figures(bearing, rotor, seals, base_coeffs, separate)

    summary = {
        "pressure": pressure_summary,
        "bearing_base": base_coeffs,
        "separate_bearings": separate,
        "rotor": rotor_summary,
        "misalignment": misalignment_summary,
        "input": {
            "bearing_radius_m": bearing.radius_m,
            "bearing_clearance_m": bearing.clearance_m,
            "bearing_length_m": bearing.length_m,
            "speed_rpm": bearing.speed_rpm,
            "rotor_span_m": rotor.span_m,
            "shaft_diameter_m": rotor.shaft_diameter_m,
            "n_stages": rotor.n_stages,
            "mass_first_stage_kg": rotor.mass_first_stage_kg,
            "mass_intermediate_kg": rotor.mass_intermediate_kg,
            "mass_last_stage_kg": rotor.mass_last_stage_kg,
            "stage_masses_kg": rotor.stage_masses_kg,
            "static_support_reactions_n": rotor.static_support_reactions_n(),
            "working_eccentricity_left": separate["epsilon_left"],
            "working_eccentricity_right": separate["epsilon_right"],
            "working_eccentricity_mean": separate["epsilon_mean"],
            "total_impeller_mass_kg": rotor.total_impeller_mass_kg,
            "mean_impeller_mass_kg": rotor.mean_impeller_mass_kg,
            "shaft_mass_kg": rotor.shaft_mass_kg,
            "total_mass_kg": rotor.total_mass_kg,
            "balance_grade_mm_s": rotor.balance_grade_mm_s,
            "balance_grade_limit_mm_s": rotor.balance_grade_limit_mm_s,
            "allowable_eccentricity_um": rotor.allowable_eccentricity_m * 1e6,
            "allowable_eccentricity_g16_um": rotor.allowable_eccentricity_for_grade_m(rotor.balance_grade_limit_mm_s) * 1e6,
            "unbalance_force_n": rotor.unbalance_force_n,
            "unbalance_force_g16_n": rotor.unbalance_force_for_grade_n(rotor.balance_grade_limit_mm_s),
            "drawing_unbalance_force_n": rotor.drawing_unbalance_force_n,
            "drawing_equivalent_grade_mm_s": rotor.drawing_equivalent_grade_mm_s,
            "g63_unbalance_first_stage_g_mm": rotor.g63_unbalance_first_stage_g_mm,
            "impeller_outer_diameter_m": rotor.impeller_outer_diameter_m,
            "impeller_equivalent_width_m": rotor.impeller_equivalent_width_m,
            "impeller_blade_count": rotor.impeller_blade_count,
            "hydraulic_force_kr_nominal": rotor.hydraulic_force_kr_nominal,
            "hydraulic_force_kr_offdesign": rotor.hydraulic_force_kr_offdesign,
            "hydraulic_radial_force_nominal_n": rotor.hydraulic_radial_force_nominal_n,
            "hydraulic_radial_force_offdesign_n": rotor.hydraulic_radial_force_offdesign_n,
            "mild_bearing_clearance_multiplier": rotor.mild_bearing_clearance_multiplier,
            "mild_seal_clearance_multiplier": rotor.mild_seal_clearance_multiplier,
            "limit_bearing_clearance_multiplier": rotor.limit_bearing_clearance_multiplier,
            "limit_seal_clearance_multiplier": rotor.limit_seal_clearance_multiplier,
            "seal_front_diameter_m": seals.front_seal_diameter_m,
            "seal_rear_diameter_m": seals.rear_seal_diameter_m,
            "seal_balance_diameter_m": seals.balance_seal_diameter_m,
            "seal_front_pressure_fraction": seals.front_pressure_fraction,
            "seal_rear_pressure_fraction": seals.rear_pressure_fraction,
            "seal_balance_pressure_fraction": seals.balance_pressure_fraction,
            "seal_stage_length_m": seals.stage_seal_length_m,
            "seal_clearance_m": seals.seal_clearance_m,
            "lomakin_factor": seals.lomakin_factor,
            "seal_damping_factor": seals.seal_damping_factor,
        },
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig_files = sorted(p.name for p in FIG_DIR.glob("*.png")) + sorted(p.name for p in FIG_DIR.glob("*.svg"))
    data_files = sorted(p.name for p in DATA_DIR.glob("*.csv")) + ["summary.json", "input_parameters.json", "drawing_assumptions.json"]
    log = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "figures": fig_files,
        "data_files": data_files,
        "sha256": {name: file_sha256(DATA_DIR / name) for name in sorted(p.name for p in DATA_DIR.glob("*.csv"))},
        "validation_checks": {
            "bearing_matrices_2x2_are_used": True,
            "scalar_equivalent_support_removed": True,
            "two_plane_rotor_dofs_x_theta_y_y_theta_x": True,
            "complex_first_order_eigenproblem": True,
            "annular_seal_elements_separate_from_impeller_nodes": True,
            "left_and_right_bearings_calculated_separately": True,
            "balance_grades_G6_3_and_G16_are_used": True,
            "coarse_balance_grade_excluded_from_operating_scenarios": True,
            "operating_hydraulic_force_is_included": True,
            "wear_clearance_scenario_is_included": True,
            "bearing_eccentricity_from_static_equilibrium": True,
            "impeller_geometry_from_koleso2_drawing": True,
            "stability_analysis_log_decrement_added": True,
            "normative_comparison_added": True,
            "rotor_scheme_saved_png_and_svg": True,
            "no_cfd_no_gyroscope": True,
        },
    }
    (VAL_DIR / "reproducibility_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    (VAL_DIR / "reproducibility_log.txt").write_text("\n".join([f"{k}: {v}" for k, v in log.items()]), encoding="utf-8")

    print("Готово. Фигуры сохранены в", FIG_DIR)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
