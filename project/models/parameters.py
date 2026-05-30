from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math


@dataclass
class BearingParams:
    """Расчётные параметры гладкого гидродинамического подшипника ЦНС 105-392."""
    # По чертежу подшипника серии 5МС-10: отверстие вкладыша 90H7;
    # для шейки принята типовая посадка e7. Средний диаметральный зазор
    # по сочетанию H7/e7 равен около 107 мкм, средний радиальный -- 53,5 мкм.
    radius_m: float = 0.045
    clearance_m: float = 5.35e-5
    length_m: float = 0.040
    viscosity_pa_s: float = 0.01105
    speed_rpm: float = 2950.0
    # Рабочий эксцентриситет далее уточняется из уравнения статического равновесия.
    # Это значение используется только как начальное/среднее для одиночных расчётов.
    eccentricity: float = 0.36
    n_phi: int = 81
    n_z: int = 35
    perturb_eps: float = 1.0e-3
    perturb_vel: float = 1.0e-3
    liner_material: str = "Сталь ШХ15 ГОСТ 801-78"
    left_load_factor: float = 0.92
    right_load_factor: float = 1.08

    @property
    def omega_rad_s(self) -> float:
        return 2.0 * math.pi * self.speed_rpm / 60.0

    @property
    def alpha(self) -> float:
        return 2.0 * self.radius_m / self.length_m

    @property
    def pressure_scale_pa(self) -> float:
        # Масштаб согласован с безразмерным уравнением: RHS = 3*dH/dphi.
        return 2.0 * self.viscosity_pa_s * self.omega_rad_s * self.radius_m**2 / self.clearance_m**2

    @property
    def force_scale_n(self) -> float:
        return self.pressure_scale_pa * self.radius_m * self.length_m / 2.0


@dataclass
class RotorParams:
    """Расчётные параметры упругого ротора ЦНС 105-392 (5МС-10)."""
    pump_model: str = "ЦНС 105-392 (5МС-10)"
    pump_flow_m3_h: float = 105.0
    pump_head_m: float = 392.0
    pump_power_kw: float = 200.0
    pump_mass_kg: float = 836.0
    operating_speed_rpm: float = 2950.0
    span_m: float = 1.951
    hydraulic_start_m: float = 0.360
    hydraulic_length_m: float = 1.071
    shaft_diameter_m: float = 0.090
    n_stages: int = 8
    # Колесо первой ступени -- по чертежу ЦНС90.00.00.012;
    # остальные семь рабочих колёс относятся к позиции "Колесо рабочее" спецификации
    # и приняты расчётно как более лёгкие промежуточные/крайние колёса.
    mass_first_stage_kg: float = 19.60
    mass_intermediate_kg: float = 16.00
    mass_last_stage_kg: float = 16.00
    density_kg_m3: float = 7850.0
    young_modulus_pa: float = 2.10e11
    damping_scale: float = 1.0
    balance_grade_mm_s: float = 6.3  # ГОСТ ИСО 1940-1, класс G6,3 для нового ротора
    balance_grade_limit_mm_s: float = 16.0  # предельно допустимый вариант для промышленного ротора
    hydraulic_force_kr_nominal: float = 0.020
    hydraulic_force_kr_offdesign: float = 0.040
    # Геометрия колеса первой ступени по чертежу ЦНС90.00.00.012.
    impeller_outer_diameter_m: float = 0.300
    impeller_equivalent_width_m: float = 0.0224
    impeller_first_stage_unbalance_g_mm: float = 250.0
    impeller_blade_count: int = 7
    hydraulic_force_y_fraction: float = 0.50
    mild_bearing_clearance_multiplier: float = 1.40
    mild_seal_clearance_multiplier: float = 1.50
    limit_bearing_clearance_multiplier: float = 1.50
    limit_seal_clearance_multiplier: float = 2.00
    g_m_s2: float = 9.81

    @property
    def stage_pitch_m(self) -> float:
        return self.hydraulic_length_m / self.n_stages

    @property
    def stage_positions_m(self) -> list[float]:
        pitch = self.stage_pitch_m
        return [self.hydraulic_start_m + (i + 0.5) * pitch for i in range(self.n_stages)]

    @property
    def stage_masses_kg(self) -> list[float]:
        if self.n_stages < 2:
            return [self.mass_first_stage_kg]
        return [self.mass_first_stage_kg] + [self.mass_intermediate_kg] * (self.n_stages - 2) + [self.mass_last_stage_kg]

    @property
    def total_impeller_mass_kg(self) -> float:
        return float(sum(self.stage_masses_kg))

    @property
    def mean_impeller_mass_kg(self) -> float:
        return self.total_impeller_mass_kg / self.n_stages

    @property
    def shaft_area_m2(self) -> float:
        return math.pi * self.shaft_diameter_m**2 / 4.0

    @property
    def shaft_inertia_m4(self) -> float:
        return math.pi * self.shaft_diameter_m**4 / 64.0

    @property
    def shaft_mass_kg(self) -> float:
        return self.density_kg_m3 * self.shaft_area_m2 * self.span_m

    @property
    def total_mass_kg(self) -> float:
        return self.shaft_mass_kg + self.total_impeller_mass_kg

    def static_support_reactions_n(self) -> tuple[float, float]:
        """Статические реакции в концевых опорах от веса вала и рабочих колёс."""
        span = self.span_m
        shaft_weight = self.shaft_mass_kg * self.g_m_s2
        left = 0.5 * shaft_weight
        right = 0.5 * shaft_weight
        for x_i, m_i in zip(self.stage_positions_m, self.stage_masses_kg):
            w_i = m_i * self.g_m_s2
            left += w_i * (span - x_i) / span
            right += w_i * x_i / span
        return float(left), float(right)

    @property
    def omega_operating_rad_s(self) -> float:
        return 2.0 * math.pi * self.operating_speed_rpm / 60.0

    def allowable_eccentricity_for_grade_m(self, grade_mm_s: float) -> float:
        # G = e*omega. G задаётся в мм/с, переводим в м/с.
        return (float(grade_mm_s) * 1.0e-3) / self.omega_operating_rad_s

    @property
    def allowable_eccentricity_m(self) -> float:
        return self.allowable_eccentricity_for_grade_m(self.balance_grade_mm_s)

    def unbalance_force_for_grade_n(self, grade_mm_s: float, mass_kg: float | None = None) -> float:
        # При e=G/omega сила равна m*G*omega. По умолчанию берётся промежуточное колесо.
        m = self.mass_intermediate_kg if mass_kg is None else float(mass_kg)
        return m * (float(grade_mm_s) * 1.0e-3) * self.omega_operating_rad_s

    @property
    def unbalance_force_n(self) -> float:
        # Расчётная сила от промежуточного рабочего колеса с допустимым остаточным дисбалансом.
        return self.unbalance_force_for_grade_n(self.balance_grade_mm_s)

    def unbalance_force_for_stage_n(self, stage_index: int, grade_mm_s: float | None = None) -> float:
        grade = self.balance_grade_mm_s if grade_mm_s is None else float(grade_mm_s)
        return self.unbalance_force_for_grade_n(grade, self.stage_masses_kg[stage_index])

    @property
    def drawing_unbalance_kg_m(self) -> float:
        return self.impeller_first_stage_unbalance_g_mm * 1.0e-6

    @property
    def drawing_unbalance_force_n(self) -> float:
        return self.drawing_unbalance_kg_m * self.omega_operating_rad_s**2

    @property
    def drawing_equivalent_grade_mm_s(self) -> float:
        ecc_m = self.drawing_unbalance_kg_m / self.mass_first_stage_kg
        return ecc_m * self.omega_operating_rad_s * 1000.0

    @property
    def g63_unbalance_first_stage_g_mm(self) -> float:
        return self.mass_first_stage_kg * self.allowable_eccentricity_for_grade_m(self.balance_grade_mm_s) * 1.0e6

    def hydraulic_radial_force_stage_n(self, kr: float | None = None) -> float:
        stage_dp = 1000.0 * self.g_m_s2 * self.pump_head_m / self.n_stages
        projected_area = self.impeller_outer_diameter_m * self.impeller_equivalent_width_m
        kr_eff = self.hydraulic_force_kr_nominal if kr is None else float(kr)
        return kr_eff * stage_dp * projected_area

    def hydraulic_radial_force_total_n(self, kr: float | None = None) -> float:
        return self.hydraulic_radial_force_stage_n(kr) * self.n_stages

    @property
    def hydraulic_radial_force_nominal_n(self) -> float:
        return self.hydraulic_radial_force_total_n(self.hydraulic_force_kr_nominal)

    @property
    def hydraulic_radial_force_offdesign_n(self) -> float:
        return self.hydraulic_radial_force_total_n(self.hydraulic_force_kr_offdesign)


@dataclass
class SealParams:
    """Инженерные параметры щелевых уплотнений ступеней и разгрузочного устройства."""
    water_density_kg_m3: float = 1000.0
    pump_head_m: float = 392.0
    # Диаметры щелевых уплотнений заданы явно: переднее -- по входной зоне
    # колеса, заднее -- по наружному диску, разгрузочное -- по типовому
    # диаметру разгрузочного устройства.
    front_seal_diameter_m: float = 0.110
    rear_seal_diameter_m: float = 0.195
    balance_seal_diameter_m: float = 0.120
    # Типовые размеры щелевых уплотнений рабочих колёс серии 5МС: Ls=8--12 мм,
    # радиальный зазор нового насоса 0,15--0,25 мм; для расчёта приняты
    # средние значения Ls=10 мм и cs=0,20 мм.
    stage_seal_length_m: float = 0.010
    balance_seal_length_m: float = 0.045
    seal_clearance_m: float = 2.0e-4
    lomakin_factor: float = 0.55
    seal_damping_factor: float = 0.25
    front_pressure_fraction: float = 0.40
    rear_pressure_fraction: float = 0.60
    balance_pressure_fraction: float = 0.30
    seal_offset_pitch_fraction: float = 0.32

    def stage_delta_p_pa(self, rotor: RotorParams) -> float:
        return self.water_density_kg_m3 * rotor.g_m_s2 * self.pump_head_m / rotor.n_stages

    def _seal_stiffness(self, delta_p_pa: float, diameter_m: float, length_m: float, clearance_m: float | None = None) -> float:
        c = self.seal_clearance_m if clearance_m is None else float(clearance_m)
        return self.lomakin_factor * delta_p_pa * diameter_m * length_m / c

    def front_stage_seal_stiffness_n_m(self, rotor: RotorParams) -> float:
        dp = self.front_pressure_fraction * self.stage_delta_p_pa(rotor)
        return self._seal_stiffness(dp, self.front_seal_diameter_m, self.stage_seal_length_m)

    def rear_stage_seal_stiffness_n_m(self, rotor: RotorParams) -> float:
        dp = self.rear_pressure_fraction * self.stage_delta_p_pa(rotor)
        return self._seal_stiffness(dp, self.rear_seal_diameter_m, self.stage_seal_length_m)

    def stage_seal_stiffness_n_m(self, rotor: RotorParams) -> float:
        # Совместимый средний коэффициент для справочных расчётов.
        return 0.5 * (self.front_stage_seal_stiffness_n_m(rotor) + self.rear_stage_seal_stiffness_n_m(rotor))

    def balance_seal_stiffness_n_m(self, rotor: RotorParams) -> float:
        dp = self.balance_pressure_fraction * self.water_density_kg_m3 * rotor.g_m_s2 * self.pump_head_m
        return self._seal_stiffness(dp, self.balance_seal_diameter_m, self.balance_seal_length_m)

    def damping_from_stiffness(self, stiffness_n_m: float, omega_rad_s: float) -> float:
        return self.seal_damping_factor * stiffness_n_m / omega_rad_s


def dump_default_parameters(path: str | Path) -> None:
    path = Path(path)
    bearing = BearingParams()
    rotor = RotorParams()
    seals = SealParams()
    data = {
        "bearing": asdict(bearing),
        "rotor": asdict(rotor),
        "seals": asdict(seals),
    }
    data["bearing"].update({
        "omega_rad_s": bearing.omega_rad_s,
        "alpha": bearing.alpha,
        "pressure_scale_pa": bearing.pressure_scale_pa,
        "force_scale_n": bearing.force_scale_n,
    })
    data["rotor"].update({
        "stage_pitch_m": rotor.stage_pitch_m,
        "stage_positions_m": rotor.stage_positions_m,
        "stage_masses_kg": rotor.stage_masses_kg,
        "static_support_reactions_n": rotor.static_support_reactions_n(),
        "shaft_mass_kg": rotor.shaft_mass_kg,
        "total_impeller_mass_kg": rotor.total_impeller_mass_kg,
        "mean_impeller_mass_kg": rotor.mean_impeller_mass_kg,
        "total_mass_kg": rotor.total_mass_kg,
        "allowable_eccentricity_m": rotor.allowable_eccentricity_m,
        "allowable_eccentricity_limit_m": rotor.allowable_eccentricity_for_grade_m(rotor.balance_grade_limit_mm_s),
        "unbalance_force_n": rotor.unbalance_force_n,
        "unbalance_force_limit_n": rotor.unbalance_force_for_grade_n(rotor.balance_grade_limit_mm_s),
        "drawing_unbalance_force_n": rotor.drawing_unbalance_force_n,
        "drawing_equivalent_grade_mm_s": rotor.drawing_equivalent_grade_mm_s,
        "g63_unbalance_first_stage_g_mm": rotor.g63_unbalance_first_stage_g_mm,
        "hydraulic_radial_force_stage_nominal_n": rotor.hydraulic_radial_force_stage_n(rotor.hydraulic_force_kr_nominal),
        "hydraulic_radial_force_total_nominal_n": rotor.hydraulic_radial_force_nominal_n,
        "hydraulic_radial_force_stage_offdesign_n": rotor.hydraulic_radial_force_stage_n(rotor.hydraulic_force_kr_offdesign),
        "hydraulic_radial_force_total_offdesign_n": rotor.hydraulic_radial_force_offdesign_n,
    })
    data["seals"].update({
        "stage_delta_p_pa": seals.stage_delta_p_pa(rotor),
        "front_stage_seal_stiffness_n_m": seals.front_stage_seal_stiffness_n_m(rotor),
        "rear_stage_seal_stiffness_n_m": seals.rear_stage_seal_stiffness_n_m(rotor),
        "stage_seal_stiffness_n_m": seals.stage_seal_stiffness_n_m(rotor),
        "balance_seal_stiffness_n_m": seals.balance_seal_stiffness_n_m(rotor),
    })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
