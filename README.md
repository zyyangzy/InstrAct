# InstrAct

Official project repository for **“InstrAct: Towards Action-Centric Understanding in Instructional Videos”** (ECCV 2026 Poster).

- [Project page](https://zyyangzy.github.io/InstrAct/)
- [Model training and evaluation](model/)
- [Data-curation pipeline](data_curation/) — **Coming soon**
- [InstrAct Bench annotations](https://drive.google.com/drive/folders/1bQzpNbi7BbhkBJBzFm4OLtwg9BdlqJSF)
- [Paper](https://arxiv.org/abs/2604.08762)

## Repository structure

```text
InstrAct/
├── docs/             # GitHub Pages project website
├── model/            # InstrAct model, training, and evaluation code
└── data_curation/    # LLM-assisted data-curation pipeline (coming soon)
```

Please follow [`model/README.md`](model/README.md) for environment setup, backbone checkpoints, annotation formats, training, and evaluation.

## Citation

```bibtex
@inproceedings{yang2026instract,
  title     = {InstrAct: Towards Action-Centric Understanding in Instructional Videos},
  author    = {Yang, Zhuoyi and Yu, Jiapeng and Tan, Reuben and Li, Boyang Albert and Xu, Huijuan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Third-party components retain their original licenses and attribution. A license for the original InstrAct code will be added before the public code release.
