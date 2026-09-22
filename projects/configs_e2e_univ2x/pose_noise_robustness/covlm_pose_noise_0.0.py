_base_ = ["../unimmv2x_covlm_cmd_coop_e2e.py"]

# Baseline: no pose noise
data = dict(
    val=dict(
        pose_noise_std=0.0,
    ),
    test=dict(
        pose_noise_std=0.0,
    ),
)

method_metadata = dict(
    experiment="Pose noise robustness baseline",
    pose_noise_std=0.0,
    note="Baseline evaluation without pose perturbation",
)
