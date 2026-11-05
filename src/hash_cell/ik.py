"""Damped least-squares inverse kinematics on a MuJoCo site (position + orientation)."""
from __future__ import annotations

import mujoco
import numpy as np


def _rot_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector, world frame."""
    q_t, q, q_err = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_t, R_target.flatten())
    mujoco.mju_mat2Quat(q, R.flatten())
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q)
    mujoco.mju_mulQuat(q_err, q_t, q_inv)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


def solve_ik(model, data, site: str, joints: list[str], pos, R=None,
             q_init=None, iters: int = 300, tol: float = 1e-5, damping: float = 1e-3):
    """Return (q, pos_err_m, rot_err_rad, ok). Does not modify `data`."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    qadr = np.array([model.joint(j).qposadr[0] for j in joints])
    dadr = np.array([model.joint(j).dofadr[0] for j in joints])
    lo = np.array([model.joint(j).range[0] for j in joints])
    hi = np.array([model.joint(j).range[1] for j in joints])
    if q_init is not None:
        d.qpos[qadr] = q_init
    sid = model.site(site).id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pos = np.asarray(pos, dtype=float)
    ep, er = np.inf, 0.0
    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        x = d.site_xpos[sid]
        e = pos - x
        ep = np.linalg.norm(e)
        if R is not None:
            erv = _rot_error(R, d.site_xmat[sid].reshape(3, 3))
            er = np.linalg.norm(erv)
            e = np.concatenate([e, erv])
        if ep < tol and er < 10 * tol:
            break
        mujoco.mj_jacSite(model, d, jacp, jacr, sid)
        J = jacp[:, dadr] if R is None else np.vstack([jacp[:, dadr], jacr[:, dadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    q = d.qpos[qadr].copy()
    return q, ep, er, bool(ep < 1e-4 and er < 1e-3)


FLIP_Z = np.diag([-1.0, -1.0, 1.0])   # 180 deg about the tool axis


def solve_grasp_ik(model, data, site, joints, pos, R, q_init=None, **kw):
    """IK for a symmetric parallel gripper: the grasp R and R rotated 180 deg about
    the tool axis are equivalent, so try both and keep the one closest to q_init."""
    best = None
    for Rc in (R, R @ FLIP_Z):
        q, ep, er, ok = solve_ik(model, data, site, joints, pos, Rc, q_init=q_init, **kw)
        cost = (not ok, np.abs(q - (q_init if q_init is not None else q)).sum())
        if best is None or cost < best[0]:
            best = (cost, (q, ep, er, ok))
    return best[1]
