#include <algorithm>

#include <mujoco/mujoco.h>

#include "mjorbit/orbit_cache.h"
#include "mjorbit/orbit_schedule.h"
#include "orbit_instance.h"

namespace {

mjorbit::OrbitInstance* GetInstance(const mjModel* m, mjData* d, int instance) {
  if (!m || !d || instance < 0 || instance >= m->nplugin) {
    return nullptr;
  }
  return reinterpret_cast<mjorbit::OrbitInstance*>(d->plugin_data[instance]);
}

int MjoStateTailSize(const mjorbit::OrbitInstance* inst) {
  return 7 + inst->num_reaction_wheels + 2 * inst->num_cmgs;
}

int MjoControlTailSize(const mjorbit::OrbitInstance* inst) {
  return inst->num_reaction_wheels + inst->num_magnetorquers + inst->num_thrusters +
         inst->num_cmgs;
}

void CopyMjtToDouble3(const mjtNum* src, double* dst) {
  dst[0] = static_cast<double>(src[0]);
  dst[1] = static_cast<double>(src[1]);
  dst[2] = static_cast<double>(src[2]);
}

void CopyDoubleToMjt3(const double* src, mjtNum* dst) {
  dst[0] = static_cast<mjtNum>(src[0]);
  dst[1] = static_cast<mjtNum>(src[1]);
  dst[2] = static_cast<mjtNum>(src[2]);
}

void UpdateReactionWheelMomentum(mjorbit::OrbitInstance* inst) {
  if (!inst->rw_speed || !inst->rw_momentum || !inst->reaction_wheels) {
    return;
  }
  for (int i = 0; i < inst->num_reaction_wheels; ++i) {
    inst->rw_momentum[i] = inst->rw_speed[i] * inst->reaction_wheels[i].inertia;
  }
}

void SetMjoState(
    const mjModel* m,
    mjData* d,
    mjorbit::OrbitInstance* inst,
    const mjtNum* state,
    int full_state_size) {
  mj_setState(m, d, state, mjSTATE_FULLPHYSICS);

  const mjtNum* tail = state + full_state_size;
  CopyMjtToDouble3(tail, inst->R_eci);
  tail += 3;
  CopyMjtToDouble3(tail, inst->V_eci);
  tail += 3;
  inst->t = static_cast<double>(*tail);
  ++tail;

  if (inst->rw_speed) {
    for (int i = 0; i < inst->num_reaction_wheels; ++i) {
      inst->rw_speed[i] = static_cast<double>(*tail++);
    }
  } else {
    tail += inst->num_reaction_wheels;
  }

  if (inst->cmg_gimbal_angle) {
    for (int i = 0; i < inst->num_cmgs; ++i) {
      inst->cmg_gimbal_angle[i] = static_cast<double>(*tail++);
    }
  } else {
    tail += inst->num_cmgs;
  }

  if (inst->cmg_rotor_momentum) {
    for (int i = 0; i < inst->num_cmgs; ++i) {
      inst->cmg_rotor_momentum[i] = static_cast<double>(*tail++);
    }
  }

  UpdateReactionWheelMomentum(inst);
  if (inst->orbit_dt > 0.0 && inst->orbit_dt > m->opt.timestep + 1.0e-12) {
    inst->orbit_schedule_initialized = 0;
    mjorbit::initialize_orbit_schedule(m, inst);
  } else {
    mjorbit::refresh_orbit_caches(inst);
  }
}

void GetMjoState(
    const mjModel* m,
    mjData* d,
    const mjorbit::OrbitInstance* inst,
    mjtNum* state,
    int full_state_size) {
  mj_getState(m, d, state, mjSTATE_FULLPHYSICS);

  mjtNum* tail = state + full_state_size;
  CopyDoubleToMjt3(inst->R_eci, tail);
  tail += 3;
  CopyDoubleToMjt3(inst->V_eci, tail);
  tail += 3;
  *tail++ = static_cast<mjtNum>(inst->t);

  for (int i = 0; i < inst->num_reaction_wheels; ++i) {
    *tail++ = static_cast<mjtNum>(inst->rw_speed ? inst->rw_speed[i] : 0.0);
  }
  for (int i = 0; i < inst->num_cmgs; ++i) {
    *tail++ = static_cast<mjtNum>(
        inst->cmg_gimbal_angle ? inst->cmg_gimbal_angle[i] : 0.0);
  }
  for (int i = 0; i < inst->num_cmgs; ++i) {
    *tail++ = static_cast<mjtNum>(
        inst->cmg_rotor_momentum ? inst->cmg_rotor_momentum[i] : 0.0);
  }
}

void ZeroMjoControls(mjorbit::OrbitInstance* inst) {
  if (inst->rw_torque_cmd) {
    std::fill(inst->rw_torque_cmd, inst->rw_torque_cmd + inst->num_reaction_wheels, 0.0);
  }
  if (inst->mtq_dipole_cmd) {
    std::fill(inst->mtq_dipole_cmd, inst->mtq_dipole_cmd + inst->num_magnetorquers, 0.0);
  }
  if (inst->thr_force_cmd) {
    std::fill(inst->thr_force_cmd, inst->thr_force_cmd + inst->num_thrusters, 0.0);
  }
  if (inst->cmg_gimbal_rate_cmd) {
    std::fill(inst->cmg_gimbal_rate_cmd, inst->cmg_gimbal_rate_cmd + inst->num_cmgs, 0.0);
  }
}

void SetMjoControls(mjorbit::OrbitInstance* inst, const mjtNum* control_tail) {
  if (!control_tail) {
    ZeroMjoControls(inst);
    return;
  }

  if (inst->rw_torque_cmd) {
    for (int i = 0; i < inst->num_reaction_wheels; ++i) {
      inst->rw_torque_cmd[i] = static_cast<double>(*control_tail++);
    }
  } else {
    control_tail += inst->num_reaction_wheels;
  }

  if (inst->mtq_dipole_cmd) {
    for (int i = 0; i < inst->num_magnetorquers; ++i) {
      inst->mtq_dipole_cmd[i] = static_cast<double>(*control_tail++);
    }
  } else {
    control_tail += inst->num_magnetorquers;
  }

  if (inst->thr_force_cmd) {
    for (int i = 0; i < inst->num_thrusters; ++i) {
      inst->thr_force_cmd[i] = static_cast<double>(*control_tail++);
    }
  } else {
    control_tail += inst->num_thrusters;
  }

  if (inst->cmg_gimbal_rate_cmd) {
    for (int i = 0; i < inst->num_cmgs; ++i) {
      inst->cmg_gimbal_rate_cmd[i] = static_cast<double>(*control_tail++);
    }
  }
}

void ResetDefaultUserInputs(const mjModel* m, mjData* d, unsigned int control_spec) {
  if (!(control_spec & mjSTATE_CTRL)) {
    mju_zero(d->ctrl, m->nu);
  }
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
    mju_zero(d->qfrc_applied, m->nv);
  }
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
    mju_zero(d->xfrc_applied, 6 * m->nbody);
  }
  if (!(control_spec & mjSTATE_MOCAP_POS)) {
    for (int i = 0; i < m->nbody; ++i) {
      const int id = m->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos + 3 * id, m->body_pos + 3 * i);
      }
    }
  }
  if (!(control_spec & mjSTATE_MOCAP_QUAT)) {
    for (int i = 0; i < m->nbody; ++i) {
      const int id = m->body_mocapid[i];
      if (id >= 0) {
        mju_copy4(d->mocap_quat + 4 * id, m->body_quat + 4 * i);
      }
    }
  }
  if (!(control_spec & mjSTATE_EQ_ACTIVE)) {
    for (int i = 0; i < m->neq; ++i) {
      d->eq_active[i] = m->eq_active0[i];
    }
  }
}

void ResetAllUserInputs(const mjModel* m, mjData* d) {
  mju_zero(d->ctrl, m->nu);
  mju_zero(d->qfrc_applied, m->nv);
  mju_zero(d->xfrc_applied, 6 * m->nbody);
  ResetDefaultUserInputs(m, d, 0);
}

void ClearWrenchSnapshot(
    const mjModel* m,
    mjData* d,
    mjorbit::OrbitInstance* inst,
    bool clear_xfrc_applied) {
  if (inst->wrench_buffer) {
    const int nbody = std::min(static_cast<int>(m->nbody), inst->wrench_body_count);
    std::fill(inst->wrench_buffer, inst->wrench_buffer + 6 * nbody, 0.0);
  }
  if (clear_xfrc_applied) {
    mju_zero(d->xfrc_applied, 6 * m->nbody);
  }
}

void CopyWrenchSnapshot(const mjModel* m, mjData* d, const mjorbit::OrbitInstance* inst) {
  if (!inst->wrench_buffer) {
    return;
  }
  const int nbody = std::min(static_cast<int>(m->nbody), inst->wrench_body_count);
  mju_zero(d->xfrc_applied, 6 * m->nbody);
  mju_copy(d->xfrc_applied, inst->wrench_buffer, 6 * nbody);
}

bool HasWarnings(const mjData* d) {
  for (int i = 0; i < mjNWARNING; ++i) {
    if (d->warning[i].number) {
      return true;
    }
  }
  return false;
}

void ClearWarnings(mjData* d) {
  for (int i = 0; i < mjNWARNING; ++i) {
    d->warning[i].number = 0;
  }
}

}  // namespace

extern "C" int mjo_rollout(
    const mjModel* m,
    mjData* d,
    int orbit_plugin_instance,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int mjo_state_size,
    int mjo_control_size,
    const mjtNum* initial_state,
    const mjtNum* initial_warmstart,
    const mjtNum* control,
    mjtNum* state,
    mjtNum* sensordata) {
  if (!m || !d || !initial_state) {
    return -1;
  }
  if (nbatch < 0 || nstep < 0) {
    return -5;
  }

  auto* inst = GetInstance(m, d, orbit_plugin_instance);
  if (!inst) {
    return -2;
  }

  const int full_state_size = mj_stateSize(m, mjSTATE_FULLPHYSICS);
  const int control_prefix_size = mj_stateSize(m, control_spec);
  if (mjo_state_size != full_state_size + MjoStateTailSize(inst)) {
    return -3;
  }
  if (mjo_control_size != control_prefix_size + MjoControlTailSize(inst)) {
    return -4;
  }
  if (nstep < 1) {
    return 0;
  }

  const int nsensordata = m->nsensordata;
  const int nv = m->nv;

  for (int r = 0; r < nbatch; ++r) {
    SetMjoState(m, d, inst, initial_state + r * mjo_state_size, full_state_size);
    if (initial_warmstart) {
      mju_copy(d->qacc_warmstart, initial_warmstart + r * nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }

    if (control) {
      ResetDefaultUserInputs(m, d, control_spec);
    } else {
      ResetAllUserInputs(m, d);
    }
    ZeroMjoControls(inst);
    ClearWarnings(d);

    for (int t = 0; t < nstep; ++t) {
      const int step = r * nstep + t;
      if (HasWarnings(d)) {
        for (int remaining = t; remaining < nstep; ++remaining) {
          const int out_step = r * nstep + remaining;
          if (state) {
            GetMjoState(
                m,
                d,
                inst,
                state + out_step * mjo_state_size,
                full_state_size);
          }
          if (sensordata) {
            mju_copy(sensordata + out_step * nsensordata, d->sensordata, nsensordata);
          }
        }
        break;
      }

      if (control) {
        const mjtNum* step_control = control + step * mjo_control_size;
        if (control_prefix_size > 0) {
          mj_setState(m, d, step_control, control_spec);
        }
        SetMjoControls(inst, step_control + control_prefix_size);
        if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
          mju_zero(d->qfrc_applied, m->nv);
        }
        if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
          mju_zero(d->xfrc_applied, 6 * m->nbody);
        }
      } else {
        ResetAllUserInputs(m, d);
        ZeroMjoControls(inst);
      }

      ClearWrenchSnapshot(m, d, inst, !(control && (control_spec & mjSTATE_XFRC_APPLIED)));
      mj_step(m, d);
      CopyWrenchSnapshot(m, d, inst);

      if (state) {
        GetMjoState(m, d, inst, state + step * mjo_state_size, full_state_size);
      }
      if (sensordata) {
        mju_copy(sensordata + step * nsensordata, d->sensordata, nsensordata);
      }
    }
  }
  return 0;
}
