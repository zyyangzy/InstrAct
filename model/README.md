# InstrAct

Official code release for **“InstrAct: Towards Action-Centric Understanding in Instructional Videos”**, an ECCV 2026 poster paper.

This compact release contains the full paper implementation (action-centric hard negatives, Action Perceiver with verb-guided distillation, DTW-Align, and Masked Action Modeling) and evaluation code for all three InstrAct Bench tasks. Benchmark annotations are distributed separately. The release intentionally excludes scripts used only to export or select similarity maps.

## Setup

```bash
conda env create -f environment.yml
conda activate instract
```

### Upstream backbone source

InstrAct keeps its model-specific interface code in `models/adapters/`. The
unmodified upstream repositories should be cloned into the following exact
directories. Pinning the commits prevents later upstream changes from breaking
the adapters.

```bash
# InternVideo / ViCLIP reference source
git clone https://github.com/OpenGVLab/InternVideo.git \
  models/upstream/internvideo
git -C models/upstream/internvideo checkout 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3

# CLIP-ViP (inside Microsoft's XPretrain repository)
git clone https://github.com/microsoft/XPretrain.git \
  models/upstream/clip_vip
git -C models/upstream/clip_vip checkout 2d2580ffe7844fe40249e42df2dbffe38056627b

# Clip4Clip
git clone https://github.com/ArrowLuo/CLIP4Clip.git \
  models/upstream/clip4clip
git -C models/upstream/clip4clip checkout 508ffa3de39ba0563a03199c440ab602a72e9b6f
```

The ViCLIP branch requires a small compatibility implementation to expose
frame-level patch tokens to the Action Perceiver. That minimal, modified code
is included in `models/third_party/internvideo_compat/`; its changes are kept
separate from both the upstream checkout and the InstrAct adapter. CLIP-ViP
and Clip4Clip use their unmodified checkouts through `models/adapters/clip_vip.py`
and `models/adapters/clip4clip.py`.

Download the official ViCLIP-B InternVideo checkpoint and place it at:

```text
checkpoints/ViCLIP-B_InternVid-200M.pth
```

Place the released InstrAct checkpoint anywhere convenient. Videos are expected as `<video-root>/<youtube_id>.mp4`.

Download the benchmark annotations from the [InstrAct Google Drive folder](https://drive.google.com/drive/folders/1bQzpNbi7BbhkBJBzFm4OLtwg9BdlqJSF), then extract them into the following paths:

```text
benchmarks/
├── InstrAct-Semantic/annotations.json
├── InstrAct-Logic/annotations.json
└── InstrAct-Dynamics/*.json
```

These are the default paths used by `eval.py`. The annotation files are not included in this source release.

## Annotation JSON format

All timestamps are floating-point seconds and `start`/`end` delimit the
annotated clip. A record with `"video_id": "abc123"` is loaded from
`<video-root>/abc123.mp4`. By default, decoding adds five seconds of temporal
context on each side and uniformly samples the configured number of frames.
Missing video files and malformed records are skipped.

### Training

The training JSON must be an object keyed by the positive caption. The caption
key must exactly match the text used by every `mam_mask_candidates` character
span.

```json
{
  "crack the egg and whisk it in the bowl": {
    "video_id": "abc123",
    "start": 12.4,
    "end": 19.8,
    "verb_phrases": [
      "crack egg",
      "whisk in bowl"
    ],
    "hard_negatives": [
      "peel the egg and whisk it in the bowl",
      "crack the egg and pour it in the bowl"
    ],
    "order_swapped_hn": [
      "whisk the egg in the bowl and then crack it"
    ],
    "mam_mask_candidates": [
      {
        "word": "crack",
        "char_start": 0,
        "char_end": 5,
        "phrase": "crack egg",
        "method": "verb"
      },
      {
        "word": "whisk",
        "char_start": 18,
        "char_end": 23,
        "phrase": "whisk in bowl",
        "method": "verb"
      }
    ]
  }
}
```

`char_start` is zero-based and `char_end` is exclusive, following normal
Python slicing: `caption[char_start:char_end]` must equal `word`. Empty
negative lists are accepted, although the full InstrAct training recipe uses
both negative types. The configured maximums are six verb phrases and twelve
hard negatives per sample; extra entries are truncated.

### InstrAct-Semantic

Semantic annotations are a JSON list. Each example contains one positive
caption and nine verb-altered candidates:

```json
[
  {
    "caption": "slice the onion and add it to the pan",
    "video_id": "abc123",
    "start": 31.2,
    "end": 38.6,
    "hard_negatives": [
      "grate the onion and add it to the pan",
      "peel the onion and add it to the pan"
    ]
  }
]
```

The example is abbreviated; the released benchmark provides nine negatives
per row. The positive caption is always candidate zero.

### InstrAct-Logic

Logic annotations use the same list structure, with order-swapped candidates:

```json
[
  {
    "caption": "crack the egg and then whisk it",
    "video_id": "abc123",
    "start": 42.0,
    "end": 49.5,
    "order_swapped_hn": [
      "whisk the egg and then crack it",
      "whisk it before cracking the egg"
    ]
  }
]
```

### InstrAct-Dynamics

Dynamics contains one JSON file per object pool under
`benchmarks/InstrAct-Dynamics/`. Each file is an object keyed by its positive
caption:

```json
{
  "slice the avocado into thin pieces": {
    "object": "avocado",
    "video_id": "xyz789",
    "start": 8.1,
    "end": 14.7,
    "operation": ["slice"]
  },
  "mash the avocado with a fork": {
    "object": "avocado",
    "video_id": "xyz456",
    "start": 21.0,
    "end": 27.3,
    "operation": ["mash"]
  }
}
```

Retrieval is computed independently within each file/object pool. `object` and
`operation` are metadata; evaluation requires the caption key, `video_id`,
`start`, and `end`.

## Evaluate InstrAct Bench

Run all three benchmarks with one checkpoint:

```bash
python eval.py \
  --benchmark all \
  --checkpoint checkpoints/instract.pth \
  --video-root /path/to/howto100m/videos
```

Use `--benchmark semantic`, `logic`, or `dynamics` for one task. Semantic reports R@1/R@5/MR, Logic reports accuracy, and Dynamics reports clip-count-weighted T2V/V2T/mean R@1, R@5, and R@10 across its 120 object pools. Metrics can be saved with `--output outputs/metrics.json`.

The paper uses cross-evaluation for the synthetic tasks: evaluate Semantic with the Order-Swapped-HN-trained checkpoint and Logic with the Verb-Altered-HN-trained checkpoint. The full-HN checkpoint is used for Dynamics.

## Train the full model

Training annotations are a JSON object keyed by caption. Each item contains `video_id`, `start`, `end`, `verb_phrases`, `hard_negatives`, `order_swapped_hn`, and `mam_mask_candidates`.

```bash
python main.py \
  --use-instract \
  --train-annotations /path/to/train.json \
  --video-root /path/to/howto100m/videos \
  --output-dir outputs \
  --run-name instract
```

For multi-GPU training, add `--multiprocessing-distributed`. See `python main.py --help` and `python eval.py --help` for all options.

## Code attribution

The implementation builds on the following works. Each corresponding file in `modules/` also contains source-level attribution and notes describing the InstrAct-specific changes.

- `contrastive_loss.py` is adapted from the [InternVideo repository](https://github.com/OpenGVLab/InternVideo). Its soft-target distillation formulation also follows [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (ALBEF)](https://proceedings.neurips.cc/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html), NeurIPS 2021.
- `dtw_loss.py` is adapted from the official implementation of [Representation Learning via Global Temporal Alignment and Cycle-Consistency](https://github.com/hadjisma/VideoAlignment), CVPR 2021. The differentiable soft-min formulation originates from [Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html), ICML 2017. InstrAct adds action-token/verb-phrase alignment and the reversed-order contrastive regularizer.
- `mask_action_modeling.py` is adapted from the [CoCa PyTorch implementation](https://github.com/lucidrains/CoCa-pytorch), based on [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://research.google/pubs/coca-contrastive-captioners-are-image-text-foundation-models/), TMLR 2022, and is specialized for masked action-word reconstruction.
- `perceiver.py` is adapted from the Knowledge Patcher in [PAXION: Patching Action Knowledge in Video-Language Foundation Models](https://github.com/MikeWangWZHL/Paxion), NeurIPS 2023, with InstrAct's temporal ordering, local temporal attention, and verb-guided teacher queries.
- `hard_negative_loss.py` implements InstrAct's grouped action-centric HardNeg-NCE objective. Its verb-altered negative and hardness-weighted contrastive setup follows [Verbs in Action: Improving Verb Understanding in Video-Language Models](https://openaccess.thecvf.com/content/ICCV2023/html/Momeni_Verbs_in_Action_Improving_Verb_Understanding_in_Video-Language_Models_ICCV_2023_paper.html), ICCV 2023.
