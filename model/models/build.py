"""Model construction utilities."""

from pathlib import Path

from .instract import InstrAct


def build_instract(config, args):
    """Build InstrAct with the configured external backbone."""
    backbone_name = config.backbone.name.lower()
    if backbone_name != "internvideo_b16":
        raise ValueError(f"unsupported backbone: {config.backbone.name}")

    try:
        from .adapters.internvideo import InternVideoAdapter
        from .third_party.internvideo_compat.viclip_text import clip_text_b16
        from .third_party.internvideo_compat.viclip_vision import clip_joint_b16
    except ImportError as error:
        raise ImportError(
            "The InternVideo compatibility subset could not be imported. "
            "See models/third_party/internvideo_compat/README.md."
        ) from error

    pretrained = config.backbone.pretrained
    vision_encoder = clip_joint_b16(
        pretrained=pretrained,
        input_resolution=config.backbone.input_resolution,
        kernel_size=config.backbone.kernel_size,
        center=config.backbone.center,
        num_frames=config.data.num_frames,
        drop_path=config.backbone.drop_path_rate,
        checkpoint_num=config.backbone.checkpoint_num,
        dropout=config.backbone.dropout,
    )
    text_encoder = clip_text_b16(
        embed_dim=config.model.text_feature_dim,
        context_length=config.data.max_text_length,
        vocab_size=config.model.multimodal_decoder.num_tokens,
        checkpoint_num=config.backbone.checkpoint_num,
        pretrained=pretrained,
    )
    model = InstrAct(config, args, vision_encoder, text_encoder)
    InternVideoAdapter.resize_model_temporal_embedding(
        model.vision_encoder, config.data.num_frames
    )

    checkpoint = config.backbone.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file() and not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
        model.load_report = InternVideoAdapter.load_checkpoint(
            model,
            checkpoint_path,
            config.data.num_frames,
            strict=bool(config.backbone.strict_checkpoint),
        )

    InternVideoAdapter.configure_finetuning(
        model,
        config,
        freeze_backbone=bool(getattr(args, "freeze_backbone", False)),
    )
    return model
