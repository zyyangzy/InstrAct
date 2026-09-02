# InternVideo compatibility subset

This directory is a minimal, modified subset of
[OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo), based on its
ViCLIP-B implementation. It is distributed under the accompanying Apache-2.0
license.

InstrAct-specific changes are limited to exposing global and per-frame patch
features, normalizing the text encoder return interface, handling temporal
embedding resizing, and removing the runtime dependency on `timm`. The
unmodified reference repository belongs in `models/upstream/internvideo/`.
