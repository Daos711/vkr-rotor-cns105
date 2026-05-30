from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable

import numpy as np
from scipy.linalg import eig

from .parameters import RotorParams, SealParams


@dataclass
class SealElement:
    x_m: float
    stiffness_n_m: float
    damping_n_s_m: float
    label: str
    kind: str


@dataclass
class SupportMatrices:
    """Линейная модель радиальной опоры с полной матрицей коэффициентов.

    В поступательных степенях свободы узла используется порядок (x, y):

        [Fx]   [Kxx Kxy] [x]   [Cxx Cxy] [xdot]
        [Fy] = [Kyx Kyy] [y] + [Cyx Cyy] [ydot]

    Матрицы не симметризуются: кросс-компоненты гидродинамической плёнки
    передаются в роторную модель в том виде, в котором получены из расчёта
    Рейнольдса.
    """

    stiffness_n_m: np.ndarray
    damping_n_s_m: np.ndarray
    label: str = "support"

    def __post_init__(self) -> None:
        self.stiffness_n_m = np.asarray(self.stiffness_n_m, dtype=float).reshape(2, 2)
        self.damping_n_s_m = np.asarray(self.damping_n_s_m, dtype=float).reshape(2, 2)

    @classmethod
    def from_coefficients(cls, coeffs: dict, label: str = "support") -> "SupportMatrices":
        k = np.array(
            [[coeffs["Kxx"], coeffs["Kxy"]], [coeffs["Kyx"], coeffs["Kyy"]]],
            dtype=float,
        )
        c = np.array(
            [[coeffs["Cxx"], coeffs["Cxy"]], [coeffs["Cyx"], coeffs["Cyy"]]],
            dtype=float,
        )
        return cls(k, c, label=label)

    @classmethod
    def isotropic(cls, stiffness_n_m: float, damping_n_s_m: float = 0.0, label: str = "isotropic") -> "SupportMatrices":
        return cls(
            np.eye(2, dtype=float) * float(stiffness_n_m),
            np.eye(2, dtype=float) * float(damping_n_s_m),
            label=label,
        )

    def scaled(self, stiffness_factor: float = 1.0, damping_factor: float = 1.0, label: str | None = None) -> "SupportMatrices":
        return SupportMatrices(
            self.stiffness_n_m * float(stiffness_factor),
            self.damping_n_s_m * float(damping_factor),
            label=self.label if label is None else label,
        )

    @property
    def kxx(self) -> float:
        return float(self.stiffness_n_m[0, 0])

    @property
    def kyy(self) -> float:
        return float(self.stiffness_n_m[1, 1])

    @property
    def cxx(self) -> float:
        return float(self.damping_n_s_m[0, 0])

    @property
    def cyy(self) -> float:
        return float(self.damping_n_s_m[1, 1])

    @property
    def direct_stiffness_mean(self) -> float:
        return float(0.5 * (self.kxx + self.kyy))

    @property
    def direct_damping_mean(self) -> float:
        return float(0.5 * (self.cxx + self.cyy))


def support_from_coefficients(coeffs: dict, label: str = "support") -> SupportMatrices:
    return SupportMatrices.from_coefficients(coeffs, label=label)


def rpm_from_rad_s(omega: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(omega) * 60.0 / (2.0 * math.pi)


def rad_s_from_rpm(rpm: float) -> float:
    return 2.0 * math.pi * rpm / 60.0


def stage_coordinates(params: RotorParams) -> np.ndarray:
    return np.array(params.stage_positions_m, dtype=float)


def seal_elements(params: RotorParams, seals: SealParams, stiffness_scale: float = 1.0) -> list[SealElement]:
    """Передние и задние щелевые уплотнения ступеней и разгрузочного устройства.

    Для бакалаврской модели щелевые уплотнения оставлены изотропными:
    каждый элемент добавляет одинаковые K и C в направлениях x и y.
    """
    pitch = params.stage_pitch_m
    offset = seals.seal_offset_pitch_fraction * pitch
    front_k = seals.front_stage_seal_stiffness_n_m(params) * stiffness_scale
    rear_k = seals.rear_stage_seal_stiffness_n_m(params) * stiffness_scale
    front_c = seals.damping_from_stiffness(front_k, params.omega_operating_rad_s)
    rear_c = seals.damping_from_stiffness(rear_k, params.omega_operating_rad_s)
    balance_k = seals.balance_seal_stiffness_n_m(params) * stiffness_scale
    balance_c = seals.damping_from_stiffness(balance_k, params.omega_operating_rad_s)

    elements: list[SealElement] = []
    for idx, x_c in enumerate(stage_coordinates(params), start=1):
        x_front = max(params.hydraulic_start_m + 0.01 * pitch, x_c - offset)
        x_rear = min(params.hydraulic_start_m + params.hydraulic_length_m - 0.01 * pitch, x_c + offset)
        elements.append(SealElement(float(x_front), float(front_k), float(front_c), f"Уплотнение {idx}п", "stage_front"))
        elements.append(SealElement(float(x_rear), float(rear_k), float(rear_c), f"Уплотнение {idx}з", "stage_rear"))

    balance_x = min(params.span_m - 0.18, params.hydraulic_start_m + params.hydraulic_length_m + 0.09)
    elements.append(SealElement(float(balance_x), float(balance_k), float(balance_c), "Разгрузочное уплотнение", "balance"))
    return elements


def node_coordinates(params: RotorParams, seal_elems: Iterable[SealElement] | None = None) -> np.ndarray:
    x_list = [0.0, params.span_m]
    x_list.extend(stage_coordinates(params).tolist())
    if seal_elems is not None:
        x_list.extend([el.x_m for el in seal_elems])
    return np.unique(np.round(np.array(x_list, dtype=float), 12))


def element_matrices(EI: float, rhoA: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    le = length
    ke = EI / le**3 * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le**2, -6.0 * le, 2.0 * le**2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le**2, -6.0 * le, 4.0 * le**2],
        ],
        dtype=float,
    )
    me = rhoA * le / 420.0 * np.array(
        [
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * le**2, 13.0 * le, -3.0 * le**2],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * le**2, -22.0 * le, 4.0 * le**2],
        ],
        dtype=float,
    )
    return ke, me


def _node_index(nodes: np.ndarray, x_m: float) -> int:
    idx = int(np.argmin(np.abs(nodes - x_m)))
    if abs(nodes[idx] - x_m) > 1.0e-8:
        raise ValueError(f"Не найден узел для координаты {x_m}")
    return idx


def _as_support(support: SupportMatrices | float, damping_n_s_m: float = 0.0, label: str = "support") -> SupportMatrices:
    if isinstance(support, SupportMatrices):
        return support
    return SupportMatrices.isotropic(float(support), float(damping_n_s_m), label=label)


def _plane_element_dofs(node_i: int, node_j: int, plane: str) -> list[int]:
    if plane == "x":
        # x_i, theta_y_i, x_j, theta_y_j
        return [4 * node_i, 4 * node_i + 1, 4 * node_j, 4 * node_j + 1]
    if plane == "y":
        # y_i, theta_x_i, y_j, theta_x_j
        return [4 * node_i + 2, 4 * node_i + 3, 4 * node_j + 2, 4 * node_j + 3]
    raise ValueError("plane must be 'x' or 'y'")


def _translation_dofs(node: int) -> list[int]:
    return [4 * node, 4 * node + 2]


def _rotation_dofs(node: int) -> list[int]:
    return [4 * node + 1, 4 * node + 3]


def assemble_beam_matrices_2d(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float | None = None,
    seal_elems: Iterable[SealElement] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сборка двумерной балочной модели с полной матрицей опоры.

    На каждый узел приходится четыре степени свободы:
    (x, theta_y, y, theta_x). Балочная часть дублируется для двух плоскостей,
    а концевые подшипники связывают плоскости через полные матрицы опор 2x2.
    """
    left = _as_support(left_support, label="left")
    if right_support is None:
        right = left
    else:
        right = _as_support(right_support, label="right")

    seal_elems = list(seal_elems or [])
    x = node_coordinates(params, seal_elems)
    n_nodes = len(x)
    n_dof = 4 * n_nodes
    M = np.zeros((n_dof, n_dof), dtype=float)
    K = np.zeros((n_dof, n_dof), dtype=float)
    C = np.zeros((n_dof, n_dof), dtype=float)

    EI = params.young_modulus_pa * params.shaft_inertia_m4
    rhoA = params.density_kg_m3 * params.shaft_area_m2

    for e in range(n_nodes - 1):
        le = x[e + 1] - x[e]
        ke, me = element_matrices(EI, rhoA, le)
        for plane in ("x", "y"):
            ids = _plane_element_dofs(e, e + 1, plane)
            for a in range(4):
                for b in range(4):
                    K[ids[a], ids[b]] += ke[a, b]
                    M[ids[a], ids[b]] += me[a, b]

    # Рабочие колёса как сосредоточенные массы в обеих поступательных степенях свободы.
    # Для ЦНС 105-392 учитываются три типа колёс: первая ступень, промежуточные
    # ступени и последняя ступень перед разгрузочным устройством.
    for x_imp, m_imp in zip(stage_coordinates(params), params.stage_masses_kg):
        i = _node_index(x, float(x_imp))
        dofs = _translation_dofs(i)
        M[dofs[0], dofs[0]] += float(m_imp)
        M[dofs[1], dofs[1]] += float(m_imp)

    # Концевые подшипники: полные матрицы опор 2x2 в поступательном блоке (x, y).
    for node, support in ((0, left), (n_nodes - 1, right)):
        dofs = _translation_dofs(node)
        K[np.ix_(dofs, dofs)] += support.stiffness_n_m
        C[np.ix_(dofs, dofs)] += support.damping_n_s_m * params.damping_scale

    # Щелевые уплотнения: изотропные промежуточные связи с корпусом.
    for el in seal_elems:
        i = _node_index(x, el.x_m)
        ix, iy = _translation_dofs(i)
        K[ix, ix] += el.stiffness_n_m
        K[iy, iy] += el.stiffness_n_m
        C[ix, ix] += el.damping_n_s_m
        C[iy, iy] += el.damping_n_s_m

    return x, M, K, C


# Совместимый псевдоним: теперь функция собирает двумерную модель.
def assemble_beam_matrices(
    params: RotorParams,
    left_stiffness_n_m: SupportMatrices | float,
    right_stiffness_n_m: SupportMatrices | float | None = None,
    left_damping_n_s_m: float = 0.0,
    right_damping_n_s_m: float | None = None,
    seal_elems: Iterable[SealElement] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = _as_support(left_stiffness_n_m, left_damping_n_s_m, label="left")
    if right_stiffness_n_m is None:
        right = left
    else:
        right = _as_support(right_stiffness_n_m, 0.0 if right_damping_n_s_m is None else right_damping_n_s_m, label="right")
    return assemble_beam_matrices_2d(params, left, right, seal_elems=seal_elems)


def state_matrix(M: np.ndarray, K: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    zero = np.zeros_like(M)
    eye = np.eye(n, dtype=float)
    minv_k = np.linalg.solve(M, K)
    minv_c = np.linalg.solve(M, C)
    return np.block([[zero, eye], [-minv_k, -minv_c]])


def complex_eigenanalysis(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float | None = None,
    seal_elems: Iterable[SealElement] | None = None,
    n_modes: int = 6,
) -> dict:
    """Комплексная задача первого порядка для M qdd + C qd + K q = 0."""
    nodes, M, K, C = assemble_beam_matrices_2d(params, left_support, right_support, seal_elems=seal_elems)
    A = state_matrix(M, K, C)
    values, vectors = eig(A)

    # Берём только половину спектра с положительной мнимой частью.
    mask = np.imag(values) > 1.0e-3
    values_pos = values[mask]
    vectors_pos = vectors[: M.shape[0], mask]
    order = np.argsort(np.abs(np.imag(values_pos)))
    values_pos = values_pos[order]
    vectors_pos = vectors_pos[:, order]

    if len(values_pos) == 0:
        # Запасной вариант для сильно демпфированных реальных корней.
        order = np.argsort(np.abs(values))
        values_pos = values[order]
        vectors_pos = vectors[: M.shape[0], order]

    values_sel = values_pos[:n_modes]
    vectors_sel = vectors_pos[:, :n_modes]
    omega = np.abs(np.imag(values_sel))
    damping_ratio = -np.real(values_sel) / np.maximum(np.abs(values_sel), 1.0e-12)
    whirl = [whirl_direction(vectors_sel[:, i], nodes) for i in range(vectors_sel.shape[1])]

    return {
        "omega_rad_s": omega,
        "rpm": rpm_from_rad_s(omega),
        "eigenvalues": values_sel,
        "modes": vectors_sel,
        "nodes": nodes,
        "damping_ratio": damping_ratio,
        "whirl": whirl,
    }


def critical_speeds(
    params: RotorParams,
    left_stiffness_n_m: SupportMatrices | float,
    right_stiffness_n_m: SupportMatrices | float | None = None,
    seal_elems: Iterable[SealElement] | None = None,
    n_modes: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = complex_eigenanalysis(params, left_stiffness_n_m, right_stiffness_n_m, seal_elems=seal_elems, n_modes=n_modes)
    return result["omega_rad_s"], result["modes"], result["nodes"]


def _is_two_plane_mode(mode_vector: np.ndarray) -> bool:
    return mode_vector.size % 4 == 0


def mode_translations(mode_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _is_two_plane_mode(mode_vector):
        return mode_vector[0::4], mode_vector[2::4]
    # Совместимость с прежней одной плоскостью.
    w = mode_vector[0::2]
    return w, np.zeros_like(w)


def normalized_mode_shape(mode_vector: np.ndarray) -> np.ndarray:
    x, y = mode_translations(mode_vector)
    amp = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2).astype(float)
    max_abs = np.max(amp)
    if max_abs == 0.0:
        return amp
    return amp / max_abs


def normalized_mode_xy(mode_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = mode_translations(mode_vector)
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0e-30)
    return np.real(x / scale), np.real(y / scale)


def mode_peak_coordinate(nodes: np.ndarray, mode_vector: np.ndarray) -> float:
    w = normalized_mode_shape(mode_vector)
    return float(nodes[int(np.argmax(np.abs(w)))])


def whirl_direction(mode_vector: np.ndarray, nodes: np.ndarray) -> str:
    x, y = mode_translations(mode_vector)
    if len(x) == 0:
        return "не определено"
    idx = int(np.argmax(np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)))
    indicator = float(np.imag(np.conj(x[idx]) * y[idx]))
    if abs(indicator) < 1.0e-12:
        return "плоская/слабая связь"
    return "прямая прецессия" if indicator > 0.0 else "обратная прецессия"


def rigid_critical_speed_rpm(params: RotorParams, left_support: SupportMatrices | float, right_support: SupportMatrices | float | None = None) -> float:
    left = _as_support(left_support, label="left")
    right = left if right_support is None else _as_support(right_support, label="right")
    k_total = left.stiffness_n_m + right.stiffness_n_m
    c_total = left.damping_n_s_m + right.damping_n_s_m
    M = np.eye(2) * params.total_mass_kg
    A = state_matrix(M, k_total, c_total)
    values, _ = eig(A)
    pos = values[np.imag(values) > 1.0e-6]
    if len(pos) == 0:
        # Для полной несимметричной матрицы жёсткости жёсткая 2DOF-оценка
        # может давать вещественные корни в форме первого порядка. В этом
        # справочном случае используем модуль корней undamped-задачи Kq=lambda Mq.
        lambdas = eig(k_total, M, right=False, check_finite=False)
        omega = np.sort(np.abs(np.sqrt(lambdas.astype(complex))))[0]
    else:
        omega = np.sort(np.abs(np.imag(pos)))[0]
    return float(rpm_from_rad_s(omega))


def frequency_response_full(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float,
    omega_exc_rad_s: float,
    seal_elems: Iterable[SealElement] | None = None,
    force_x: complex | float | None = None,
    force_y: complex | float | None = None,
    force_node_x_m: float | None = None,
    nodal_forces: Iterable[tuple[float, complex | float, complex | float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, M, K, C = assemble_beam_matrices_2d(params, left_support, right_support, seal_elems=seal_elems)
    rhs = np.zeros(len(nodes) * 4, dtype=complex)
    if nodal_forces is None:
        center_x = stage_coordinates(params)[params.n_stages // 2] if force_node_x_m is None else force_node_x_m
        center_node = _node_index(nodes, float(center_x))
        fx = params.unbalance_force_n if force_x is None else force_x
        fy = -1j * params.unbalance_force_n if force_y is None else force_y
        rhs[4 * center_node] += fx
        rhs[4 * center_node + 2] += fy
    else:
        for x_m, fx, fy in nodal_forces:
            node = _node_index(nodes, float(x_m))
            rhs[4 * node] += fx
            rhs[4 * node + 2] += fy
    dynamic_matrix = K - omega_exc_rad_s**2 * M + 1j * omega_exc_rad_s * C
    q = np.linalg.solve(dynamic_matrix.astype(complex), rhs)
    return q, nodes


# Устаревшая одноосевая функция оставлена только для совместимости сторонних проверок.
def frequency_response(
    params: RotorParams,
    left_stiffness_n_m: SupportMatrices | float,
    right_stiffness_n_m: SupportMatrices | float,
    left_damping_n_s_m: float,
    right_damping_n_s_m: float,
    omega_exc_rad_s: float,
    force_phase: complex = 1.0 + 0.0j,
    seal_elems: Iterable[SealElement] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left = _as_support(left_stiffness_n_m, left_damping_n_s_m, label="left")
    right = _as_support(right_stiffness_n_m, right_damping_n_s_m, label="right")
    q, nodes = frequency_response_full(params, left, right, omega_exc_rad_s, seal_elems=seal_elems, force_x=params.unbalance_force_n * force_phase, force_y=0.0)
    return q, nodes



def orbit_points(qx: complex, qy: complex, omega_rad_s: float, n_points: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi / omega_rad_s, n_points)
    exp_term = np.exp(1j * omega_rad_s * t)
    x = np.real(qx * exp_term)
    y = np.real(qy * exp_term)
    return t, x, y


def ellipse_metrics(qx: complex, qy: complex) -> dict:
    transform = np.array(
        [[np.real(qx), -np.imag(qx)], [np.real(qy), -np.imag(qy)]],
        dtype=float,
    )
    semi_axes = np.linalg.svd(transform, compute_uv=False)
    a = float(semi_axes[0])
    b = float(semi_axes[1])
    return {
        "semi_major_m": a,
        "semi_minor_m": b,
        "axis_ratio": float(a / b) if b > 0.0 else float("inf"),
        "area_m2": float(math.pi * a * b),
    }


def orbit_response_full(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float,
    speed_rpm: float,
    seal_elems: Iterable[SealElement] | None = None,
    force_x: complex | float | None = None,
    force_y: complex | float | None = None,
    force_node_x_m: float | None = None,
    nodal_forces: Iterable[tuple[float, complex | float, complex | float]] | None = None,
) -> Dict[str, np.ndarray | float]:
    omega = rad_s_from_rpm(speed_rpm)
    q, nodes = frequency_response_full(
        params,
        left_support,
        right_support,
        omega,
        seal_elems=seal_elems,
        force_x=force_x,
        force_y=force_y,
        force_node_x_m=force_node_x_m,
        nodal_forces=nodal_forces,
    )
    center_x = stage_coordinates(params)[params.n_stages // 2]
    center = _node_index(nodes, float(center_x))
    qx = q[4 * center]
    qy = q[4 * center + 2]
    t, xf, yf = orbit_points(qx, qy, omega)
    metrics = ellipse_metrics(qx, qy)
    return {
        "time_s": t,
        "x_m": xf,
        "y_m": yf,
        "amp_um": float(max(np.max(np.abs(xf)), np.max(np.abs(yf))) * 1.0e6),
        "semi_major_um": float(metrics["semi_major_m"] * 1.0e6),
        "semi_minor_um": float(metrics["semi_minor_m"] * 1.0e6),
        "axis_ratio": float(metrics["axis_ratio"]),
        "nodes": nodes,
        "q": q,
    }


# Устаревшая сигнатура: преобразуем прямые коэффициенты в диагональные матрицы.
def orbit_response(
    params: RotorParams,
    kx_left: float,
    kx_right: float,
    ky_left: float,
    ky_right: float,
    cx_left: float,
    cx_right: float,
    cy_left: float,
    cy_right: float,
    speed_rpm: float,
    seal_elems: Iterable[SealElement] | None = None,
) -> Dict[str, np.ndarray | float]:
    left = SupportMatrices(np.array([[kx_left, 0.0], [0.0, ky_left]]), np.array([[cx_left, 0.0], [0.0, cy_left]]), label="left_diag")
    right = SupportMatrices(np.array([[kx_right, 0.0], [0.0, ky_right]]), np.array([[cx_right, 0.0], [0.0, cy_right]]), label="right_diag")
    return orbit_response_full(params, left, right, speed_rpm, seal_elems=seal_elems)


def support_slope_response_full(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float,
    speed_rpm: float,
    seal_elems: Iterable[SealElement] | None = None,
    force_x: complex | float | None = None,
    force_y: complex | float | None = None,
    force_node_x_m: float | None = None,
    nodal_forces: Iterable[tuple[float, complex | float, complex | float]] | None = None,
) -> dict:
    omega = rad_s_from_rpm(speed_rpm)
    q, nodes = frequency_response_full(
        params,
        left_support,
        right_support,
        omega,
        seal_elems=seal_elems,
        force_x=force_x,
        force_y=force_y,
        force_node_x_m=force_node_x_m,
        nodal_forces=nodal_forces,
    )
    left_theta_y = q[1]
    left_theta_x = q[3]
    right_theta_y = q[-3]
    right_theta_x = q[-1]

    def rot_amp(a: complex, b: complex) -> float:
        metrics = ellipse_metrics(a, b)
        return float(metrics["semi_major_m"])

    left_amp = rot_amp(left_theta_y, left_theta_x)
    right_amp = rot_amp(right_theta_y, right_theta_x)
    return {
        "left_theta_x_rad_complex_abs": float(abs(left_theta_x)),
        "left_theta_y_rad_complex_abs": float(abs(left_theta_y)),
        "right_theta_x_rad_complex_abs": float(abs(right_theta_x)),
        "right_theta_y_rad_complex_abs": float(abs(right_theta_y)),
        "left_theta_amp_rad": float(left_amp),
        "right_theta_amp_rad": float(right_amp),
        "max_theta_amp_rad": float(max(left_amp, right_amp)),
        "node_coordinates_m": nodes,
        "q": q,
    }


# Устаревшая сигнатура, сохранена как диагональная модель.
def support_slope_response(
    params: RotorParams,
    kx_left: float,
    kx_right: float,
    ky_left: float,
    ky_right: float,
    cx_left: float,
    cx_right: float,
    cy_left: float,
    cy_right: float,
    speed_rpm: float,
    seal_elems: Iterable[SealElement] | None = None,
) -> dict:
    left = SupportMatrices(np.array([[kx_left, 0.0], [0.0, ky_left]]), np.array([[cx_left, 0.0], [0.0, cy_left]]), label="left_diag")
    right = SupportMatrices(np.array([[kx_right, 0.0], [0.0, ky_right]]), np.array([[cx_right, 0.0], [0.0, cy_right]]), label="right_diag")
    return support_slope_response_full(params, left, right, speed_rpm, seal_elems=seal_elems)



def undamped_eigenanalysis(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float | None = None,
    seal_elems: Iterable[SealElement] | None = None,
    n_modes: int = 4,
) -> dict:
    """Быстрая вспомогательная задача K phi = lambda M phi.

    Используется для параметрических свипов и чувствительности, чтобы не
    решать десятки раз полную задачу первого порядка. Основные результаты
    по частотам в отчёте берутся из complex_eigenanalysis().
    """
    nodes, M, K, _ = assemble_beam_matrices_2d(params, left_support, right_support, seal_elems=seal_elems)
    values, vectors = eig(K, M, check_finite=False)
    finite = np.isfinite(values)
    values = values[finite]
    vectors = vectors[:, finite]
    # Для слабонесимметричной K возможна небольшая мнимая часть lambda.
    omega = np.abs(np.sqrt(values.astype(complex)))
    mask = omega > 1.0e-6
    omega = omega[mask]
    vectors = vectors[:, mask]
    order = np.argsort(omega)
    omega = np.real(omega[order])
    vectors = vectors[:, order]
    return {"omega_rad_s": omega[:n_modes], "rpm": rpm_from_rad_s(omega[:n_modes]), "modes": vectors[:, :n_modes], "nodes": nodes}

def first_critical_for_span(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float,
    span_m: float,
    seal_elems: Iterable[SealElement] | None = None,
) -> float:
    p = replace(params, span_m=float(span_m), hydraulic_length_m=min(params.hydraulic_length_m, max(0.4, span_m - params.hydraulic_start_m - 0.30)))
    if p.hydraulic_start_m + p.hydraulic_length_m >= p.span_m:
        p = replace(p, hydraulic_start_m=0.18 * span_m, hydraulic_length_m=0.55 * span_m)
    result = undamped_eigenanalysis(p, left_support, right_support, seal_elems=None, n_modes=1)
    return float(result["rpm"][0])


def first_critical_for_support(
    params: RotorParams,
    left_support: SupportMatrices | float,
    right_support: SupportMatrices | float,
    seal_elems: Iterable[SealElement] | None = None,
) -> float:
    result = undamped_eigenanalysis(params, left_support, right_support, seal_elems=seal_elems, n_modes=1)
    return float(result["rpm"][0])
