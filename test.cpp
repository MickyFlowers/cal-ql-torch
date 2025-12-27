void AdmittanceSE3Full6D::step(
  const Eigen::Vector3d& desired_p,
  const Eigen::Quaterniond& desired_q,
  const Eigen::Vector3d& current_p,
  const Eigen::Quaterniond& current_q,
  const Eigen::Matrix<double,6,1>& wrench_W_meas,
  Eigen::Vector3d& cmd_p,
  Eigen::Quaterniond& cmd_q,
  double dt)
{
  /* 1. pose error e = adm - desired */
  Eigen::Matrix<double,6,1> e;
  e.head<3>() = current_p - desired_p;

  const Eigen::Matrix3d R  = current_q.normalized().toRotationMatrix();
  const Eigen::Matrix3d Rd = desired_q.normalized().toRotationMatrix();
  e.tail<3>() = so3Log(R * Rd.transpose());

  /* 2. external wrench (world) */
  Eigen::Matrix<double,6,1> w = preprocessWrench(wrench_W_meas);

  // Use gate_state_ directly as interpolation weight a in [gate_alpha_min, 1].
  // Larger contact force -> gate_state_ closer to 1 -> K approaches K_contact (and D approaches D_contact).
  Eigen::Matrix<double,6,6> K_eff = cfg_.K;
  Eigen::Matrix<double,6,6> D_eff = cfg_.D;
  if (cfg_.gate_enable) {
    constexpr double K_contact = 10.0; // alpha=1 target (translation)
    constexpr double D_contact = 20.0; // alpha=1 target (translation)
    for (int i = 0; i < 3; ++i) {
      const double a_i = gate_state_(i);  // per-axis gating
      // Nonlinear K mapping to reduce sensitivity at small contacts.
      // gate_state in [gate_alpha_min, 1] -> weight in [0, 1]
      const double denom = (1.0 - cfg_.gate_alpha_min);
      const double w01 = (denom > 1e-12) ? (a_i - cfg_.gate_alpha_min) / denom : 1.0;
      const double wK_raw = std::pow(w01, cfg_.k_gate_gamma);

      // Add dynamics (enter slow, exit fast) on K weight to avoid oscillation near contact.
      double wK = wK_raw;
      if (cfg_.k_tau_in > 0.0 && cfg_.k_tau_out > 0.0) {
        const bool entering = (wK_raw > k_weight_state_(i)); // entering contact => K decreasing
        const double tau = entering ? cfg_.k_tau_in : cfg_.k_tau_out;
        const double beta = (tau > 1e-12) ? (dt / (tau + dt)) : 1.0;
        k_weight_state_(i) = (1.0 - beta) * k_weight_state_(i) + beta * wK_raw;
        wK = k_weight_state_(i);
      } else {
        k_weight_state_(i) = wK_raw;
      }

      K_eff(i, i) = (1.0 - wK) * cfg_.K(i, i) + wK * K_contact;
      D_eff(i, i) = (1.0 - a_i) * cfg_.D(i, i) + a_i * D_contact;
    }
  }

  // Publish debug values actually used in computation
  if (dbg_pub_enabled_) {
    dbg_gate_pub_.publish(toMsg6(gate_state_));
    dbg_K_pub_.publish(toMsg6(K_eff.diagonal()));
    dbg_D_pub_.publish(toMsg6(D_eff.diagonal()));
  }

  Eigen::Matrix<double,6,1> xd_des = -K_eff * e;

  /* 3. M ë + D ė + K e = w */
  Eigen::Matrix<double,6,1> e_ddot = cfg_.M_inv * (w - D_eff * (admVel_ - xd_des));

  // e_ddot = clampNorm(e_ddot, cfg_.vdot_max);

  /* 4. integrate admittance velocity (ė) */
  admVel_ += e_ddot * dt;
  // admVel_ = clampNorm(admVel_, cfg_.v_max);

  /* 5. integrate admittance pose (world Δp / ΔR) */
  cmd_p = current_p;
  cmd_q = current_q;
  // integrateWorldPoseRate(cmd_p, cmd_q, admVel_ * 0.2, dt);
  integrateWorldPoseRate(cmd_p, cmd_q, cfg_.gain * admVel_, dt);
}