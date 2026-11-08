"""Parameters of the tolerance model. Units: mm and rad.

Every value marked ASSUMPTION is not given by the drawing or a datasheet; it is a
stated engineering estimate, to be replaced by measurements (e.g. measured profiles,
measured robot accuracy). They are grouped here so the report can list them in one table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

# ISO 2768-1 class m, linear dimensions: (upper bound of range, permissible deviation)
_ISO2768_M = [(3, 0.1), (6, 0.1), (30, 0.2), (120, 0.3), (400, 0.5), (1000, 0.8)]


def iso2768m(nominal: float) -> float:
    for upper, dev in _ISO2768_M:
        if nominal <= upper:
            return dev
    raise ValueError(nominal)


@dataclass(frozen=True)
class Nominal:
    """Drawing AMZ-SA-26-001 rev. A."""
    L1: float = 320.0      # Pos 1 length
    W1: float = 40.0       # Pos 1 width
    H1: float = 20.0       # Pos 1 height
    L2: float = 280.0      # Pos 2 length
    W2: float = 30.0
    H2: float = 15.0
    d: float = 120.0       # A1, A2 centre distances
    a3: float = 100.0      # A3 Pos 1 end -> Pos 2 axis
    a4: float = 60.0       # A4 Pos 1 outer edge -> Pos 2 end
    tol: float = 0.05      # +/- on A1-A4
    geo: float = 0.05      # A5-A8 tolerance zones / max gap


@dataclass(frozen=True)
class PartTol:
    """Part tolerances (half ranges). Default = ISO 2768-m, as on the drawing."""
    L1: float = iso2768m(320)
    W1: float = iso2768m(40)
    H1: float = iso2768m(20)
    L2: float = iso2768m(280)
    W2: float = iso2768m(30)
    H2: float = iso2768m(15)
    dist: str = "normal"                 # "normal": tolerance = 3 sigma ; "uniform": anywhere in the band
    paired_height_sigma: float | None = None   # sigma of H1a - H1b when Pos 1 are paired (same bar / sorted)


@dataclass(frozen=True)
class FormErrors:
    """Form errors of the cold-formed profiles. ALL ASSUMPTIONS, to be measured."""
    bow_sigma: float = 0.15          # lateral bow over the part length, unclamped
    h2_var_sigma: float = 0.02       # height variation of a Pos 2 along its length
    gap_sigma: float = 0.10          # raw gap at a Pos 2 / Pos 1 contact (twist, flange flatness)


@dataclass(frozen=True)
class Process:
    """Process errors."""
    robot_sigma: float = 0.33                    # ASSUMPTION absolute positioning per axis (+/-1 mm at 3 sigma); repeatability alone is +/-0.1
    robot_yaw_sigma: float = math.radians(0.05)  # ASSUMPTION
    grasp_sigma: float = 0.17                    # ASSUMPTION part offset / slip in the gripper along its length (+/-0.5)
    base_cal_sigma: float = 0.17                 # ASSUMPTION robot B base frame relative to robot A after calibration
    fixture_sigma: float = 0.01 / 3              # machined fixture features +/-0.01
    centring_sigma: float = 0.003                # repeatability of self-centring jaws
    bow_residual_sigma: float = 0.005            # bow left after clamping against a straight reference
    gap_closable: float = 0.5                    # ASSUMPTION gap the clamps close by elastic deformation
    gap_residual_sigma: float = 0.005            # gap left after clamping
    machining_sigma: float = 0.02 / 3            # end machining after joining, +/-0.02


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    placement: str            # "robot" | "datum" | "centring"
    clamped: bool
    machine_ends: bool = False
    parts: PartTol = field(default_factory=PartTol)


PRECISION_CUT = 0.05          # +/- length, precision saw (ASSUMPTION, typical for a cut-to-length saw with stop)
PAIRED_H = 0.01               # sigma of the height difference of two Pos 1 cut consecutively from one bar / sorted


def scenarios() -> list[Scenario]:
    paired = replace(PartTol(), paired_height_sigma=PAIRED_H)
    controlled = replace(paired, L1=PRECISION_CUT, L2=PRECISION_CUT)
    return [
        Scenario("S0", "Robot only", "robot", clamped=False),
        Scenario("S1", "Fixture,\ndatum stops", "datum", clamped=True),
        Scenario("S2", "Fixture,\nself-centring", "centring", clamped=True),
        Scenario("S3", "S2 + controlled\nparts", "centring", clamped=True, parts=controlled),
        Scenario("S4", "S2 + paired heights\n+ end machining", "centring", clamped=True,
                 machine_ends=True, parts=paired),
    ]
