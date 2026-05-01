# RGB dataset clone target

Clone the RGB dataset into this directory before running the scripts:

```bash
git clone --depth 1 https://github.com/chen700564/RGB ./RGB
```

Expected layout after cloning:

```
rgb/data/RGB/data/en.json        # 300-sample noise-robustness subset (what we used)
rgb/data/RGB/data/en_fact.json
rgb/data/RGB/data/en_int.json
rgb/data/RGB/data/en_refine.json
```

The `RGB/` subdirectory is gitignored - we don't redistribute the
upstream dataset.
