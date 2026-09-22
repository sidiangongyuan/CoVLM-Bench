_base_ = ["../unimmv2x_covlm_cmd_coop_e2e.py"]

# Medium pose noise: σ = 0.2m translation, 0.02 rad yaw
data = dict(
    val=dict(
        pose_noise_std=0.2,
    ),
    test=dict(
        pose_noise_std=0.2,
    ),
)

method_metadata = dict(
    experiment="Pose noise robustness σ=0.2",
    pose_noise_std=0.2,
    note="Medium localization error: 0.2m translation noise, 0.02 rad yaw noise",
)
