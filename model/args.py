"""Command-line switches for the InstrAct release."""

import argparse


def get_args(description="InstrAct pretraining"):
    parser = argparse.ArgumentParser(description=description)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--amp", choices=("off", "fp16", "bf16"), default="fp16")
    runtime.add_argument("--workers", type=int, default=16)
    runtime.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="launch one process per visible GPU with torch.multiprocessing.spawn",
    )
    runtime.add_argument("--dist-backend", default="nccl")
    runtime.add_argument("--dist-url", default="env://")
    runtime.add_argument(
        "--find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable DDP unused-parameter graph traversal if a custom branch needs it",
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--model-config", default="models/config.yaml")
    paths.add_argument("--train-annotations")
    paths.add_argument("--val-annotations")
    paths.add_argument("--video-root", required=True)
    paths.add_argument("--output-dir", default="outputs")
    paths.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="checkpoint path, or omit the value to resume the latest epoch",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=72)
    training.add_argument("--val-batch-size", type=int, default=128)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=0.05)
    training.add_argument("--warmup-steps", type=int, default=1000)
    training.add_argument(
        "--scheduler-total-epochs",
        type=int,
        help="cosine schedule horizon; defaults to --epochs",
    )
    training.add_argument("--gradient-clip-norm", type=float, default=1.0)
    training.add_argument("--log-every", type=int, default=20)
    training.add_argument("--eval-every", type=int, default=1)
    training.add_argument("--keep-checkpoints", type=int, default=5)
    training.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="retain every epoch checkpoint (used only by speculative runs)",
    )
    training.add_argument(
        "--early-stop-training-loss",
        action="store_true",
        help="stop when epoch training loss reaches the configured plateau",
    )
    training.add_argument("--plateau-patience", type=int, default=3)
    training.add_argument(
        "--plateau-min-relative-improvement", type=float, default=0.005
    )
    training.add_argument("--plateau-min-epochs", type=int, default=5)
    training.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="optional batch limit for debugging and smoke tests",
    )
    training.add_argument(
        "--max-eval-samples",
        type=int,
        help="optional evaluation sample limit for debugging",
    )
    training.add_argument("--eval-only", action="store_true")
    training.add_argument(
        "--eval-benchmark",
        choices=("both", "hard_negative", "order"),
        default="both",
        help="evaluate both candidate groups or one separated benchmark JSON",
    )
    training.add_argument(
        "--eval-max-negatives",
        type=int,
        help="override the number of hard negatives per evaluation sample",
    )

    features = parser.add_argument_group("model switches")
    features.add_argument(
        "--use-instract",
        action="store_true",
        help=(
            "enable full InstrAct: hard negatives, Action Perceiver, DTW, and MAM"
        ),
    )
    features.add_argument(
        "--use-hard-negatives",
        action="store_true",
        help="enable action-centric hard-negative contrastive learning",
    )
    features.add_argument(
        "--hard-negative-type",
        choices=("both", "verb", "order"),
        default="both",
        help="select verb-altered, order-swapped, or both HN sources",
    )
    features.add_argument(
        "--use-action-perceiver",
        action="store_true",
        help="enable only the Action Perceiver (for ablations)",
    )
    features.add_argument(
        "--use-dtw",
        action="store_true",
        help="enable DTW-Align and its required Action Perceiver",
    )
    features.add_argument(
        "--use-mam",
        action="store_true",
        help="enable masked action modeling and its required Action Perceiver",
    )
    features.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="freeze both InternVideo encoders and train only InstrAct modules",
    )

    logging = parser.add_argument_group("logging")
    logging.add_argument("--run-name", default="instract")
    logging.add_argument("--wandb", action="store_true")
    logging.add_argument("--wandb-project", default="instract")

    return parser.parse_args()
