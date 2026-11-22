"""
Convert the official SMPL v1.1.0 model files into chumpy-free pickles that
smplx can load on modern Python / NumPy, and give them the names smplx expects.

The official files (basicmodel_*_lbs_10_207_0_v1.1.0.pkl) store some arrays as
`chumpy` objects. chumpy does not install on recent Python/NumPy, so loading
fails with "No module named 'chumpy'". This script unpickles them with a stub
instead of chumpy and saves plain NumPy arrays.

  python scripts/convert_smpl_pkl.py --src SMPL_python_v.1.1.0.zip --out data/body_models/smpl

--src can be the downloaded zip or the folder you extracted it to.
Result: data/body_models/smpl/SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl
"""
import argparse
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

NAMES = {"neutral": "SMPL_NEUTRAL.pkl", "_m_": "SMPL_MALE.pkl", "_f_": "SMPL_FEMALE.pkl"}


class _ChumpyStub:
    """Stands in for any chumpy class while unpickling; keeps the object's state."""
    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _ChumpyStub
        if module.startswith("scipy.sparse"):         # old private module paths (csc, compressed, ...)
            import scipy.sparse
            if hasattr(scipy.sparse, name):
                return getattr(scipy.sparse, name)
        return super().find_class(module, name)


def _to_array(key, v):
    """Turn a (stubbed) chumpy object into a NumPy array."""
    if isinstance(v, _ChumpyStub):
        d = v.__dict__
        if "x" in d:                       # a leaf chumpy array keeps its value in .x
            return np.asarray(d["x"])
        arrays = [val for val in d.values() if isinstance(val, np.ndarray)]
        if len(arrays) == 1:
            return arrays[0]
        raise ValueError(f"cannot convert chumpy field '{key}', attributes: {sorted(d)}")
    if hasattr(v, "r") and type(v).__module__.startswith("chumpy"):   # real chumpy installed
        return np.asarray(v.r)
    if hasattr(v, "toarray"):                    # scipy sparse (J_regressor) -> dense array
        return np.asarray(v.toarray())
    return v


def convert(raw: bytes):
    data = _Unpickler(io.BytesIO(raw), encoding="latin1").load()
    return {k: _to_array(k, v) for k, v in data.items()}


def find_models(src: Path):
    """Yield (filename, bytes) of every SMPL model .pkl inside a zip or a folder."""
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.endswith(".pkl") and "basicmodel" in Path(n).name.lower():
                    yield Path(n).name, z.read(n)
    else:
        for p in src.rglob("*.pkl"):
            if "basicmodel" in p.name.lower():
                yield p.name, p.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="SMPL_python_v.1.1.0.zip or its extracted folder")
    ap.add_argument("--out", default="data/body_models/smpl")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"not found: {src}  (check the folder and file name in Google Drive)")
    for name, raw in find_models(src):
        target = next((v for k, v in NAMES.items() if k in name.lower()), None)
        if target is None:
            continue
        model = convert(raw)
        with open(out / target, "wb") as fh:
            pickle.dump(model, fh, protocol=2)
        shapes = {k: getattr(v, "shape", type(v).__name__) for k, v in model.items()
                  if k in ("v_template", "shapedirs", "posedirs", "weights", "f")}
        print(f"{name} -> {out / target}  {shapes}")
        done.append(target)
    if not done:
        raise SystemExit(f"no basicmodel_*.pkl found in {a.src}; is this SMPL_python_v.1.1.0?")
    missing = set(NAMES.values()) - set(done)
    if missing:
        print("warning: missing", sorted(missing))
    print("next: python tests/check_smpl.py --smpl", out)


if __name__ == "__main__":
    main()
