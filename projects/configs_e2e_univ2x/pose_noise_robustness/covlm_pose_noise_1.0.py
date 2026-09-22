_base_ = ["../unimmv2x_covlm_cmd_coop_e2e.py"]

# Very high pose noise: σ = 1.0m translation, 0.1 rad yaw
data = dict(
    val=dict(
        pose_noise_std=1.0,
    ),
    test=dict(
        pose_noise_std=1.0,
    ),
)

method_metadata = dict(
    experiment="Pose noise robustness σ=1.0",
    pose_noise_std=1.0,
    note="Very high localization error: 1.0m translation noise, 0.1 rad yaw noise",
)
