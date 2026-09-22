_base_ = ["../unimmv2x_covlm_cmd_coop_e2e.py"]

# Low pose noise: σ = 0.1m translation, 0.01 rad yaw
data = dict(
    val=dict(
        pose_noise_std=0.1,
    ),
    test=dict(
        pose_noise_std=0.1,
    ),
)

method_metadata = dict(
    experiment="Pose noise robustness σ=0.1",
    pose_noise_std=0.1,
    note="Low localization error: 0.1m translation noise, 0.01 rad yaw noise",
)
