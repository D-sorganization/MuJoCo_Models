---
issue: 453
summary: "Contact-aware inverse dynamics: inverse_dynamics_with_contacts reports the contact generalized force next to the free-root residual; gait and sit-to-stand feet seated at the contact equilibrium like the lifts; grip.dense_efc_jacobian picks the Jacobian layout from mj_isSparse instead of the unset sparse row metadata (MuJoCo 3.15 segfault and zero contact load)"
dl_state: "in_review"
next_step: "Merge the PR."
branch: "fix/453-id-contacts"
---
