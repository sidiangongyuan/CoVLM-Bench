_base_ = ["./univ2x_coop_e2e.py"]

seed = 20260524
planning_steps = 6
total_epochs = 6
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
load_from = "ckpts/univ2x_coop_e2e_stg2.pth"
fp16 = dict(loss_scale=512.0)

covlm_index_file = (
    "output/covla_baseline/"
    "qwen3_vl_8b_cot_supervised_reference_seed20260524_20260603_212658/"
    "index_v3_l4_usable_2129.jsonl"
)

moe_transformer_layer = dict(
    is_moe=True,
    num_experts=8,
    top_k=2,
    load_balance_loss_weight=0.05,
)

model_other_agent_inf = dict(
    load_from=None,
    planning_only=True,
    planning_only_freeze_perception=True,
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(
                transformerlayers=moe_transformer_layer,
            ),
        ),
    ),
)

model_ego_agent = dict(
    load_from=None,
    planning_only=True,
    planning_only_freeze_perception=True,
    planning_only_num_modes=1,
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(
                transformerlayers=moe_transformer_layer,
            ),
        ),
    ),
    planning_head=dict(
        planning_steps=planning_steps,
        num_commands=7,
        predict_command=True,
        use_col_optim=False,
        loss_command=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
)

data = dict(
    workers_per_gpu=4,
    train=dict(
        planning_steps=planning_steps,
        covlm_index_file=covlm_index_file,
        covlm_required_split="train",
    ),
    val=dict(
        planning_steps=planning_steps,
        covlm_index_file=covlm_index_file,
        covlm_required_split="val",
    ),
    test=dict(
        planning_steps=planning_steps,
        covlm_index_file=covlm_index_file,
        covlm_required_split="val",
    ),
)

method_metadata = dict(
    method="UniMM-V2X CoVLM-Bench protocol port/adaptation",
    seed=seed,
    initialization="ckpts/univ2x_coop_e2e_stg2.pth",
    command_protocol="CoVLM 7-way predicted command logits, no GT command at inference",
    planning_supervision="CoVLM 6-step structured waypoints in [x_lateral, y_forward]",
    planning_export="CoVLM 654-row normal eval, 6x2 CoVLM-frame structured predictions",
    unimm_compatible_subset=[
        "BEVFormer encoder MoE FFN",
        "existing UniV2X agent query fusion",
        "planning-only training path using existing UniV2X agent query fusion",
    ],
    limitations=[
        "full UniMM decoder/motion MoE is not enabled in this first old-stack port",
        "map, motion, and occupancy heads are skipped for this planning-only retry",
        "normal-only external baseline does not support causal V2X-gain claims",
    ],
)
