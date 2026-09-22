_base_ = ["../unimmv2x_covlm_cmd_coop_e2e.py"]

# High pose noise: σ = 0.5m translation, 0.05 rad yaw
data = dict(
    val=dict(
        pose_noise_std=0.5,
    ),
    test=dict(
        pose_noise_std=0.5,
    ),
)

method_metadata = dict(
    experiment="Pose noise robustness σ=0.5",
    pose_noise_std=0.5,
    note="High localization error: 0.5m translation noise, 0.05 rad yaw noise",
)
