from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .parameters import BearingParams


def clearance(
    phi: np.ndarray,
    eps_x: float,
    eps_y: float = 0.0,
    z_grid: np.ndarray | None = None,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    params: BearingParams | None = None,
) -> np.ndarray:
    """Безразмерный зазор гладкого цилиндрического подшипника.

    Без перекоса используется H = 1 + eps_x*cos(phi) + eps_y*sin(phi).
    При перекосе шейки к эксцентриситету добавляется линейное по оси смещение:
    s*theta/c, где s = Z*L/2. Возвращается массив размера (n_phi, n_z).
    """
    phi = np.asarray(phi, dtype=float)
    if z_grid is None:
        h = 1.0 + eps_x * np.cos(phi) + eps_y * np.sin(phi)
        if np.min(h) <= 0.0:
            raise ValueError("Получен неположительный зазор H. Уменьшите эксцентриситет или перекос.")
        return h
    if params is None:
        raise ValueError("Для расчёта перекоса требуется params.")
    z_grid = np.asarray(z_grid, dtype=float)
    s_over_c = (params.length_m / (2.0 * params.clearance_m)) * z_grid
    eps_x_z = eps_x + tilt_x_rad * s_over_c
    eps_y_z = eps_y + tilt_y_rad * s_over_c
    H = 1.0 + np.cos(phi)[:, None] * eps_x_z[None, :] + np.sin(phi)[:, None] * eps_y_z[None, :]
    if np.min(H) <= 0.0:
        raise ValueError("Получен неположительный зазор H при перекосе. Уменьшите угол перекоса.")
    return H


def solve_pressure(
    params: BearingParams,
    eps_x: float | None = None,
    eps_y: float = 0.0,
    vel_x: float = 0.0,
    vel_y: float = 0.0,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    cavitation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Решение безразмерного уравнения Рейнольдса для гладкой опоры.

    При tilt_x_rad / tilt_y_rad зазор зависит от осевой координаты Z, что
    используется для оценки misalignment. Для скорости расчётов дискретная
    система собирается в разреженную матрицу и решается spsolve.
    """
    if eps_x is None:
        eps_x = params.eccentricity

    n_phi = params.n_phi
    n_z = params.n_z
    alpha = params.alpha

    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    z_grid = np.linspace(-1.0, 1.0, n_z)
    d_phi = 2.0 * math.pi / n_phi
    d_z = 2.0 / (n_z - 1)

    H = clearance(phi, eps_x, eps_y, z_grid=z_grid, tilt_x_rad=tilt_x_rad, tilt_y_rad=tilt_y_rad, params=params)

    # Производная H по phi. Для перекоса компоненты эксцентриситета зависят от Z.
    s_over_c = (params.length_m / (2.0 * params.clearance_m)) * z_grid
    eps_x_z = eps_x + tilt_x_rad * s_over_c
    eps_y_z = eps_y + tilt_y_rad * s_over_c
    dH_dphi = -np.sin(phi)[:, None] * eps_x_z[None, :] + np.cos(phi)[:, None] * eps_y_z[None, :]

    # Динамическая добавка оставлена только окружной функцией; для коэффициентов
    # демпфирования перекос не используется.
    dynamic_part = 2.0 * (vel_x * np.sin(phi) + vel_y * np.cos(phi))[:, None]
    RHS = 3.0 * (dH_dphi + dynamic_part)

    H_ip = 0.5 * (H + np.roll(H, -1, axis=0))
    H_im = 0.5 * (H + np.roll(H, 1, axis=0))

    H_jp = np.empty_like(H)
    H_jm = np.empty_like(H)
    H_jp[:, :-1] = 0.5 * (H[:, :-1] + H[:, 1:])
    H_jp[:, -1] = H[:, -1]
    H_jm[:, 1:] = 0.5 * (H[:, 1:] + H[:, :-1])
    H_jm[:, 0] = H[:, 0]

    n_unknown = n_phi * (n_z - 2)
    matrix = lil_matrix((n_unknown, n_unknown), dtype=float)
    vector = np.zeros(n_unknown, dtype=float)

    def idx(i: int, j: int) -> int:
        return (j - 1) * n_phi + (i % n_phi)

    for j in range(1, n_z - 1):
        for i in range(n_phi):
            row = idx(i, j)
            A = H_ip[i, j] ** 3 / d_phi**2
            B = H_im[i, j] ** 3 / d_phi**2
            C = alpha**2 * H_jp[i, j] ** 3 / d_z**2
            D = alpha**2 * H_jm[i, j] ** 3 / d_z**2
            E = A + B + C + D

            matrix[row, idx(i, j)] = E
            matrix[row, idx(i + 1, j)] = -A
            matrix[row, idx(i - 1, j)] = -B
            if j + 1 <= n_z - 2:
                matrix[row, idx(i, j + 1)] = -C
            if j - 1 >= 1:
                matrix[row, idx(i, j - 1)] = -D
            vector[row] = -RHS[i, j]

    solution = spsolve(matrix.tocsr(), vector)
    pressure = np.zeros((n_phi, n_z), dtype=float)
    for j in range(1, n_z - 1):
        pressure[:, j] = solution[(j - 1) * n_phi : j * n_phi]

    if cavitation:
        pressure[pressure < 0.0] = 0.0

    return phi, z_grid, H, pressure


def integrate_force(params: BearingParams, phi: np.ndarray, z: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Размерные компоненты гидродинамической силы, Н."""
    fx_dim = np.trapezoid(np.trapezoid(pressure * np.cos(phi)[:, None], z, axis=1), phi)
    fy_dim = np.trapezoid(np.trapezoid(pressure * np.sin(phi)[:, None], z, axis=1), phi)
    return params.force_scale_n * np.array([fx_dim, fy_dim], dtype=float)


def bearing_force(params: BearingParams, eps_x: float, eps_y: float = 0.0, vel_x: float = 0.0, vel_y: float = 0.0) -> np.ndarray:
    phi, z, _, pressure = solve_pressure(params, eps_x=eps_x, eps_y=eps_y, vel_x=vel_x, vel_y=vel_y)
    return integrate_force(params, phi, z, pressure)


def bearing_coefficients(params: BearingParams, eccentricity: float | None = None) -> dict:
    """Коэффициенты жёсткости и демпфирования гладкой опоры."""
    eps0 = params.eccentricity if eccentricity is None else eccentricity
    de = params.perturb_eps
    dv = params.perturb_vel

    base_force = bearing_force(params, eps0, 0.0)

    stiffness = np.zeros((2, 2), dtype=float)
    for col, (dx, dy) in enumerate(((de, 0.0), (0.0, de))):
        f_plus = bearing_force(params, eps0 + dx, dy)
        f_minus = bearing_force(params, eps0 - dx, -dy)
        stiffness[:, col] = -(f_plus - f_minus) / (2.0 * de * params.clearance_m)

    damping = np.zeros((2, 2), dtype=float)
    for col, (vx, vy) in enumerate(((dv, 0.0), (0.0, dv))):
        f_plus = bearing_force(params, eps0, 0.0, vx, vy)
        f_minus = bearing_force(params, eps0, 0.0, -vx, -vy)
        damping[:, col] = (f_plus - f_minus) / (2.0 * dv * params.clearance_m * params.omega_rad_s)

    return {
        "eccentricity": eps0,
        "force_x_n": float(base_force[0]),
        "force_y_n": float(base_force[1]),
        "load_n": float(np.linalg.norm(base_force)),
        "Kxx": float(stiffness[0, 0]),
        "Kxy": float(stiffness[0, 1]),
        "Kyx": float(stiffness[1, 0]),
        "Kyy": float(stiffness[1, 1]),
        "Cxx": float(damping[0, 0]),
        "Cxy": float(damping[0, 1]),
        "Cyx": float(damping[1, 0]),
        "Cyy": float(damping[1, 1]),
    }


def coefficients_sweep(params: BearingParams, eps_values: np.ndarray) -> list[dict]:
    return [bearing_coefficients(params, eccentricity=float(eps)) for eps in eps_values]


def bearing_matrices(coeffs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Полные матрицы жёсткости и демпфирования подшипника.

    Матрицы возвращаются в порядке поступательных координат (x, y) и
    передаются в роторную модель без скалярного усреднения и без удаления
    кросс-компонент.
    """
    stiffness = np.array(
        [[coeffs["Kxx"], coeffs["Kxy"]], [coeffs["Kyx"], coeffs["Kyy"]]],
        dtype=float,
    )
    damping = np.array(
        [[coeffs["Cxx"], coeffs["Cxy"]], [coeffs["Cyx"], coeffs["Cyy"]]],
        dtype=float,
    )
    return stiffness, damping


def misalignment_clearance_metrics(params: BearingParams, theta_rad: float, eps: float | None = None) -> dict:
    """Консервативная оценка краевого зазора при перекосе шейки."""
    if eps is None:
        eps = params.eccentricity
    chi = params.length_m * theta_rad / (2.0 * params.clearance_m)
    h_min_center = params.clearance_m * (1.0 - eps)
    h_min_edge_safe = params.clearance_m * (1.0 - eps - abs(chi))
    h_min_edge_far = params.clearance_m * (1.0 - eps + abs(chi))
    return {
        "theta_rad": float(theta_rad),
        "theta_mrad": float(theta_rad * 1.0e3),
        "chi": float(chi),
        "h_min_center_um": float(h_min_center * 1.0e6),
        "h_min_edge_loaded_um": float(h_min_edge_safe * 1.0e6),
        "h_min_edge_unloaded_um": float(h_min_edge_far * 1.0e6),
        "relative_reduction_percent": float(abs(chi) / max(1.0 - eps, 1e-12) * 100.0),
    }


def eccentricity_for_target_load(
    params: BearingParams,
    target_load_n: float,
    eps_min: float = 0.20,
    eps_max: float = 0.82,
    n_points: int = 17,
) -> tuple[float, list[dict]]:
    """Подбор эксцентриситета по заданной статической нагрузке.

    Используется монотонная интерполяция по предварительно рассчитанной
    зависимости несущей силы гладкого подшипника от эксцентриситета. Такой
    способ удобен для раздельного расчёта левой и правой опоры, когда статические
    реакции отличаются.
    """
    eps_grid = np.linspace(eps_min, eps_max, n_points)
    rows = []
    for eps in eps_grid:
        coeff = bearing_coefficients(params, eccentricity=float(eps))
        rows.append({"eccentricity": float(eps), "load_n": coeff["load_n"]})
    loads = np.array([r["load_n"] for r in rows], dtype=float)
    eps_values = np.array([r["eccentricity"] for r in rows], dtype=float)
    target = float(np.clip(target_load_n, loads.min(), loads.max()))
    eps_interp = float(np.interp(target, loads, eps_values))
    return eps_interp, rows
