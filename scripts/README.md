# Data acquisition

Every corpus this project uses is downloaded, never committed — several are
licensed against redistribution and all are far too large. These scripts are
the record of *how* it was acquired, which for a project about dataset
provenance is not an optional detail.

| script | source | size |
|---|---|---|
| `download-covid-19.py` | COVID-19 Radiography Database (Kaggle) | ~1.2 GB |
| `download-rsna-pneu.py` | RSNA Pneumonia Detection Challenge (Kaggle) | ~3.7 GB |
| `download-bimcv.sh` | BIMCV-COVID19 iterations 1+2, both partitions | 745 GB available |

Everything lands in `datasets/`, which is gitignored.

## BIMCV

The two b2drop shares total 745 GB, and most of it is CT: roughly half the
positive partition and 86% of the negative one is `.nii.gz` volumes this
project has no use for. Those bytes cannot be skipped at download time, since
they sit inside the same tarballs, but they need never touch the disk.

`download-bimcv.sh` therefore works one archive at a time — fetch, extract only
the radiographs and per-study metadata, delete the archive, continue. The
working set stays near 12 GB while ~80 GB crosses the wire.

```sh
./download-bimcv.sh --partition posi --parts 4     # ~950 COVID+ radiographs
./download-bimcv.sh --partition neg  --parts 12    # ~980 COVID- radiographs
```

Archives are sharded by subject, so any subset is a coherent sample. Each
completed part leaves a marker under `.done/`, so an interrupted run resumes
rather than restarting.

**The negative partition needs roughly 3x as many parts** for the same number
of images: its archives are 86% CT against the positive partition's 52%, so a
5 GB negative part yields ~82 radiographs where a positive one yields ~237.

**The negative sharding starts at `partab` with gaps.** The script lists the
share over WebDAV rather than generating part names, because a generated
`aa..zz` range requests archives that do not exist.

### What the labels mean

The partition (`posi`/`neg`) is molecular status. The `Labels` column in
`derivatives/labels/` is what the radiologist reported. They disagree often
enough to matter: 13.8% of COVID-positive studies are radiologically *normal*,
and 6.1% of the negative partition is labelled `covid 19`. That disagreement is
a ceiling on any model trained here, and it is the honest reason to expect
Track 2 in the high 0.7s rather than the 0.99 the pooled corpus produces.

Only 20% of the negative partition is radiologically normal — the rest carries
pleural effusion, pneumonia, cardiomegaly and so on. That is what makes it a
usable control group: the task is COVID against other pathology, not sick
against healthy.
