# InstrAct Bench annotations

Benchmark annotations are not bundled with the source release. Download them
from the [InstrAct Google Drive folder](https://drive.google.com/drive/folders/1bQzpNbi7BbhkBJBzFm4OLtwg9BdlqJSF)
and arrange the extracted files as follows:

```text
benchmarks/
├── InstrAct-Semantic/
│   └── annotations.json
├── InstrAct-Logic/
│   └── annotations.json
└── InstrAct-Dynamics/
    ├── almond.json
    ├── avocado.json
    └── ...                         # one JSON per object pool
```

These are the default paths used by `eval.py`. Alternatively, pass custom
locations with `--semantic-annotations`, `--logic-annotations`, and
`--dynamics-annotations`.
