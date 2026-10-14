#!/usr/bin/env python
"""Build a Kaggle submission notebook from the author's 2nd-place inference notebook.

Every edit to the author's notebook is marked `CHANGED` in the generated code. Options:

    --weights-kernel K   replace the author's 6-model ensemble with one Whole model (EfficientNet-B0) whose
                         best.pth is in the output of notebook K (e.g. darrensheehan/darren-sheehan-03-baseline-training)
    --p1                 add lead-consistency post-processing (P1): fuse the two lead II predictions, then enforce
                         II = I + III and aVR + aVL + aVF = 0 (ideas from the 1st and 3rd place solutions)

Always applied, because these notebooks are created from the command line rather than copied in the browser:
the author's hard-coded /kaggle/input/<name> paths are redirected to links in /tmp (Kaggle now mounts inputs under
/kaggle/input/{datasets,competitions,notebooks}/... and /kaggle/input is read-only), and segmentation-models-pytorch
is installed from an attached wheel (internet must be off).

    python scripts/make_submission_notebook.py reference/.../physionet-2nd-place-submission.ipynb \\
        notebooks/kaggle/05-author-ensemble-p1 --slug darren-sheehan-05-author-ensemble-p1 --p1
"""
import argparse
import json
from pathlib import Path

LEGACY = ["hengck23-demo-submit-physionet", "physionet-final-submission-models", "physionet-ecg-image-digitization"]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:7a9d2a6b13b3566aa6cc5a22447f3b3f99acf55dce44408deb90584b6d4b0dde"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("source_notebook", type=Path, help="the author's inference notebook (.ipynb)")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--slug", required=True, help="Kaggle notebook slug, e.g. darren-sheehan-05-author-ensemble-p1")
    p.add_argument("--weights-kernel", default=None, help="notebook whose output holds whole_b0_baseline/best.pth")
    p.add_argument("--p1", action="store_true", help="add lead-consistency post-processing")
    return p.parse_args()


def text(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def code(s):
    return {"cell_type": "code", "metadata": {"trusted": True}, "execution_count": None, "outputs": [],
            "source": s.strip("\n")}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


SETUP = r"""
# CHANGED: Kaggle now mounts inputs under /kaggle/input/{datasets,competitions,notebooks}/..., while the author's
# code uses /kaggle/input/<name>, and /kaggle/input is read-only. His paths are redirected to /tmp/kaggle_input/<name>,
# which links to wherever each input is mounted.
import glob, os
os.makedirs("/tmp/kaggle_input", exist_ok=True)
for name in LEGACY_INPUTS:
    hits = [p for p in [f"/kaggle/input/{name}"] + glob.glob(f"/kaggle/input/*/{name}") + glob.glob(f"/kaggle/input/*/*/{name}")
            if os.path.isdir(p)]
    assert hits, f"input not found: {name}"
    link = f"/tmp/kaggle_input/{name}"
    if not os.path.islink(link):
        os.symlink(hits[0], link)
    print(link, "->", os.path.realpath(link))

# CHANGED: segmentation-models-pytorch 0.5.0 from an attached wheel (internet is off)
wheels = glob.glob("/kaggle/input/*/*/smp-wheel-0-5-0/*.whl") + glob.glob("/kaggle/input/smp-wheel-0-5-0/*.whl")
assert wheels, "smp wheel not found"
print("smp wheel:", wheels[0])
!pip install -q --no-index --no-deps {wheels[0]}
""".replace("for name in LEGACY_INPUTS:", "for name in " + repr(LEGACY) + ":")

OWN_WEIGHTS_SETUP = r"""
# CHANGED: own weights (output of notebook 03)
OWN_ENCODER = "tu-timm/tf_efficientnet_b0.ns_jft_in1k"
hits = sorted(glob.glob("/kaggle/input/**/outputs/whole_b0_baseline/best.pth", recursive=True))
assert len(hits) == 1, hits
OWN_WEIGHTS = hits[0]
print("weights:", OWN_WEIGHTS)
"""

OWN_LOADER = r"""
def get_whole_model(encoder_name, weight_path, device):
    stage2_model = WholeModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        decoder_name="unet",
        use_coord_conv=True,
        pretrained=False
    )
    state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage, weights_only=False)
    # CHANGED: train.py saves {"model_state_dict": ..., "optimizer_state_dict": ..., ...}; the author's .pth files
    # are the bare state dict. With strict=False a wrapped checkpoint would load nothing, silently.
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    result = stage2_model.load_state_dict(state_dict, strict=False)
    print(result)
    # Only the training-time auxiliary heads may be missing
    assert not result.unexpected_keys and all(k.startswith("aux_pixel_heads") for k in result.missing_keys), result
    stage2_model.to(device)
    stage2_model.eval()
    stage2_model.output_type = ['infer']
    return stage2_model
"""

P1_FUNCTION = r"""
# CHANGED: P1, lead-consistency post-processing (from the 1st and 3rd place solutions).
# The page shows I, II, III over 0-2.5 s and aVR, aVL, aVF over 2.5-5 s, each group recorded simultaneously, and
# lead II twice over 0-2.5 s (row 2 and the start of the rhythm strip). Physically II = I + III and
# aVR + aVL + aVF = 0 exactly, so any mismatch in the predictions is error.
P1_STATS = []

def lead_consistency(series_by_lead, ii_short):
    s = {k: np.asarray(v, dtype=np.float64).copy() for k, v in series_by_lead.items()}
    ii_short = np.asarray(ii_short, dtype=np.float64)
    n = min(len(ii_short), len(s['I']), len(s['III']), len(s['II']))
    m = min(len(s['aVR']), len(s['aVL']), len(s['aVF']))
    before = (np.abs(s['II'][:n] - s['I'][:n] - s['III'][:n]).mean(),
              np.abs(s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]).mean())

    # 1. Lead II fusion: average the rhythm strip's first quarter with the row-2 II segment, which the
    #    author's code otherwise discards
    s['II'][:n] = 0.5 * (s['II'][:n] + ii_short[:n])

    # 2. Einthoven, II = I + III. Weighted least-squares correction with weights II = 2 (it now averages two
    #    predictions), I = III = 1, as in the 3rd place solution: II -= 0.2 r, I += 0.4 r, III += 0.4 r
    r = s['II'][:n] - s['I'][:n] - s['III'][:n]
    s['II'][:n] -= 0.2 * r
    s['I'][:n] += 0.4 * r
    s['III'][:n] += 0.4 * r

    # 3. Goldberger, aVR + aVL + aVF = 0: subtract a third of the sum from each
    t = (s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]) / 3
    for k in ('aVR', 'aVL', 'aVF'):
        s[k][:m] -= t

    after = (np.abs(s['II'][:n] - s['I'][:n] - s['III'][:n]).mean(),
             np.abs(s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]).mean())
    P1_STATS.append(before + after)
    return {k: v.astype(np.float32) for k, v in s.items()}
"""

P1_CALL_OLD = "\t\t\tseries_by_lead['II'] = series[3]\n"
P1_CALL_NEW = (
    "\t\t\tii_short = series_by_lead['II']  # CHANGED: keep the row-2 lead II prediction for P1\n"
    "\t\t\tseries_by_lead['II'] = series[3]\n"
    "\t\t\ttry:  # CHANGED: P1 lead consistency; fall back to the author's output if anything goes wrong\n"
    "\t\t\t\tseries_by_lead = lead_consistency(series_by_lead, ii_short)\n"
    "\t\t\texcept Exception as e:\n"
    "\t\t\t\tprint('P1 skipped for', sample_id, e)\n"
)
P1_REPORT_OLD = "\tsubmit_df.to_csv('submission.csv',index=False)\n"
P1_REPORT_NEW = P1_REPORT_OLD + (
    "\t# CHANGED: P1 summary (mean absolute mismatch, mV)\n"
    "\tif P1_STATS:\n"
    "\t\tst = np.array(P1_STATS)\n"
    "\t\tprint(f'P1 on {len(st)} records: |II-I-III| {st[:,0].mean():.4f} -> {st[:,2].mean():.4f}, '\n"
    "\t\t      f'|aVR+aVL+aVF| {st[:,1].mean():.4f} -> {st[:,3].mean():.4f}')\n"
)


def main():
    args = parse_args()
    nb = json.load(open(args.source_notebook, encoding="utf-8"))
    cells = nb["cells"]

    def find(snippet):
        hits = [i for i, c in enumerate(cells) if snippet in text(c)]
        assert len(hits) == 1, (snippet, hits)
        return hits[0]

    cells[:] = [c for c in cells if "Unmodified copy" not in text(c)]  # header of the S1 copy, if present
    for c in cells:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    changes = []
    if args.weights_kernel:
        cells[find("def get_whole_model(")]["source"] = OWN_LOADER.strip("\n")
        i = find("whole_models = [")
        s = cells[i]["source"]
        start, end = s.index("    whole_models = ["), s.index("    start_timer = timer()")
        cells[i]["source"] = s[:start] + (
            "    # CHANGED: the author's 2 Whole + 4 Series ensemble replaced by one Whole model trained in notebook 03\n"
            "    whole_models = [\n"
            "        get_whole_model(OWN_ENCODER, OWN_WEIGHTS, device),\n"
            "    ]\n"
            "    lead_models = []\n\n") + s[end:]
        changes.append(f"**Own weights**: the author's 6-model ensemble is replaced by one Whole model (EfficientNet-B0) "
                       f"trained with his training script, from the output of `{args.weights_kernel}`. The loader "
                       f"unwraps the `train.py` checkpoint and stops if anything except the training-only auxiliary "
                       f"heads fails to load.")
    else:
        changes.append("**Weights**: the author's own 6-model ensemble (2 Whole + 4 Series models), unchanged.")

    if args.p1:
        i = find("def make_submission(")
        s = cells[i]["source"]
        assert s.count(P1_CALL_OLD) == 1 and s.count(P1_REPORT_OLD) == 1
        cells[i]["source"] = P1_FUNCTION.strip("\n") + "\n\n" + s.replace(P1_CALL_OLD, P1_CALL_NEW).replace(
            P1_REPORT_OLD, P1_REPORT_NEW)
        changes.append("**P1, lead consistency** (1st and 3rd place ideas): average the two lead II predictions over "
                       "0-2.5 s (the author keeps only the rhythm strip), then enforce II = I + III (weights II 2, "
                       "I 1, III 1) and aVR + aVL + aVF = 0.")

    n_redirected = 0
    for c in cells:
        if c["cell_type"] == "code":
            for name in LEGACY:
                n_redirected += c["source"].count(f"/kaggle/input/{name}")
                c["source"] = c["source"].replace(f"/kaggle/input/{name}", f"/tmp/kaggle_input/{name}")

    setup = SETUP + (OWN_WEIGHTS_SETUP if args.weights_kernel else "")
    first_code = next(k for k, c in enumerate(cells) if c["cell_type"] == "code")
    cells.insert(first_code, code(setup))
    title = args.slug.replace("darren-sheehan-", "Darren-Sheehan-")
    cells.insert(0, md(f"# {title}\n\nAssignment 1. Takashi Someya's 2nd-place inference notebook with these "
                       f"changes, each marked `CHANGED` in the code:\n\n" + "\n".join(f"- {c}" for c in changes) +
                       "\n- **Environment**: the author's hard-coded input paths are redirected to links in `/tmp` "
                       "(Kaggle's mount paths changed, and `/kaggle/input` is read-only), and "
                       "segmentation-models-pytorch is installed from an attached wheel.\n\n"
                       "Everything else (hengck23's Stage 0/1, flip TTA, decoding, pinned environment) is unchanged."))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"darrensheehan/{args.slug}", "title": title, "code_file": f"{args.slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["hengck23/hengck23-demo-submit-physionet", "takashisomeya/physionet-final-submission-models",
                            "darrensheehan/smp-wheel-0-5-0"],
        "competition_sources": ["physionet-ecg-image-digitization"],
        "kernel_sources": [args.weights_kernel] if args.weights_kernel else [],
        "model_sources": [], "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4",
    }
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"written {args.out_dir / (args.slug + '.ipynb')} ({n_redirected} paths redirected)")


if __name__ == "__main__":
    main()
