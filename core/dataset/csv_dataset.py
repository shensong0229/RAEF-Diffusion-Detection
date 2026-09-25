# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _filter_df(df: pd.DataFrame, split: Optional[str] = None, domains=None) -> pd.DataFrame:
    need = {"path", "label", "domain"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"index_csv missing columns: {sorted(miss)}; need {sorted(need)}")

    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)]

    if domains:
        if isinstance(domains, str):
            doms = {d.strip() for d in domains.split(",") if d.strip()}
        else:
            doms = set(domains)
        df = df[df["domain"].astype(str).isin(doms)]

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No samples after filtering. split={split}, domains={domains}")
    return df


class CSVIndexDataset(Dataset):
    """CSV index dataset.

    CSV columns: path,label,domain,(optional)split

    Modes
    -----
    1) Default single-view:
       - paired_root=None, transform2=None
       - returns: img, label, domain

    2) Consistency double-view:
       - paired_root=None, transform2!=None
       - returns: (img1, img2), label, domain

    3) Paired RGB + auxiliary:
       - paired_root!=None
       - pair_mode="pair"     -> returns: (rgb, aux), label, domain
       - pair_mode="aux_only" -> returns: aux, label, domain

    Missing-file policy
    -------------------
    - strict:   immediately raise on missing/bad file
    - resample: retry with another sample up to max_resample_trials, then raise
    - fallback: retry then return a black-image fallback (legacy behavior)
    """

    def __init__(
        self,
        index_csv: str,
        data_root: str,
        split: str | None = None,
        domains=None,
        transform=None,
        max_retries: int = 30,  # legacy alias
        badlog_path: str | None = None,
        transform2=None,
        paired_root: str | None = None,
        paired_transform=None,
        pair_mode: str = "pair",
        missing_policy: str = "fallback",
        max_resample_trials: int | None = None,
        allow_black_fallback: bool | None = None,
    ) -> None:
        self.index_csv = index_csv
        self.data_root = data_root
        self.transform = transform
        self.transform2 = transform2

        self.paired_root = paired_root
        self.paired_transform = paired_transform
        self.pair_mode = str(pair_mode or "pair").lower().strip()
        if self.pair_mode not in ["pair", "aux_only"]:
            raise ValueError(f"Unsupported pair_mode={pair_mode}; choose pair/aux_only")

        self.missing_policy = str(missing_policy or "fallback").lower().strip()
        if self.missing_policy not in {"strict", "resample", "fallback"}:
            raise ValueError("missing_policy must be one of: strict, resample, fallback")

        # keep legacy compatibility
        if max_resample_trials is None:
            max_resample_trials = int(max_retries)
        self.max_resample_trials = int(max(0, max_resample_trials))
        if allow_black_fallback is None:
            allow_black_fallback = (self.missing_policy == "fallback")
        self.allow_black_fallback = bool(allow_black_fallback)

        df = pd.read_csv(index_csv)
        self.df = _filter_df(df, split=split, domains=domains)

        if badlog_path is None:
            badlog_path = os.path.join(os.path.dirname(os.path.abspath(index_csv)), "broken_images.txt")
        self.badlog_path = badlog_path

    def __len__(self) -> int:
        return len(self.df)

    def _abs_path(self, rel_path: str, root: str) -> str:
        rel_path = str(rel_path).replace("\\", "/")
        return os.path.join(root, rel_path)

    def _log_bad(self, abs_path: str, err: Exception) -> None:
        try:
            os.makedirs(os.path.dirname(self.badlog_path), exist_ok=True)
            with open(self.badlog_path, "a", encoding="utf-8") as f:
                f.write(f"{abs_path}\t{type(err).__name__}\t{str(err)}\n")
        except Exception:
            pass

    def _open_rgb(self, abs_path: str) -> Image.Image:
        img = Image.open(abs_path)
        img.load()
        return img.convert("RGB")

    def _resolve_aux_path(self, rel_path: str) -> str:
        abs_aux = self._abs_path(rel_path, self.paired_root)
        if os.path.isfile(abs_aux):
            return abs_aux

        rel_norm = str(rel_path).replace("\\", "/")
        rel_base, _ = os.path.splitext(rel_norm)
        rel_png = rel_base + ".png"
        abs_aux_png = self._abs_path(rel_png, self.paired_root)
        if os.path.isfile(abs_aux_png):
            return abs_aux_png

        raise FileNotFoundError(f"paired aux not found: tried [{abs_aux}] and [{abs_aux_png}]")

    def _load_pair(self, rel_path: str):
        abs_rgb = self._abs_path(rel_path, self.data_root)
        if not os.path.isfile(abs_rgb):
            raise FileNotFoundError(abs_rgb)
        abs_aux = self._resolve_aux_path(rel_path)
        img_rgb = self._open_rgb(abs_rgb)
        img_aux = self._open_rgb(abs_aux)
        return img_rgb, img_aux, abs_rgb, abs_aux

    def _build_fallback(self):
        fallback = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.paired_root is not None:
            if self.paired_transform is not None:
                fb_rgb, fb_aux = self.paired_transform(fallback, fallback.copy())
            else:
                fb_rgb, fb_aux = fallback, fallback.copy()
            if self.pair_mode == "aux_only":
                return fb_aux, 0, "fallback"
            return (fb_rgb, fb_aux), 0, "fallback"

        if self.transform2 is None:
            if self.transform is not None:
                fallback = self.transform(fallback)
            return fallback, 0, "fallback"

        fb1 = self.transform(fallback) if self.transform is not None else fallback
        fb2 = self.transform2(fallback) if self.transform2 is not None else fallback
        return (fb1, fb2), 0, "fallback"

    def _handle_read_error(self, rel_path: str, err: Exception) -> None:
        if self.paired_root is not None:
            try:
                self._log_bad(self._abs_path(rel_path, self.data_root), err)
                self._log_bad(self._abs_path(rel_path, self.paired_root), err)
            except Exception:
                pass
        else:
            try:
                self._log_bad(self._abs_path(rel_path, self.data_root), err)
            except Exception:
                pass

    def __getitem__(self, idx: int):
        tries = 0
        last_err: Exception | None = None

        while True:
            row = self.df.iloc[int(idx)]
            rel_path = row["path"]
            label = int(row["label"])
            domain = str(row["domain"])

            try:
                if self.paired_root is not None:
                    img_rgb, img_aux, _, _ = self._load_pair(rel_path)
                    if self.paired_transform is not None:
                        out_rgb, out_aux = self.paired_transform(img_rgb, img_aux)
                    else:
                        out_rgb, out_aux = img_rgb, img_aux
                    if self.pair_mode == "aux_only":
                        return out_aux, label, domain
                    return (out_rgb, out_aux), label, domain

                abs_path = self._abs_path(rel_path, self.data_root)
                if not os.path.isfile(abs_path):
                    raise FileNotFoundError(abs_path)

                img = self._open_rgb(abs_path)

                if self.transform2 is None:
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, label, domain

                img1 = self.transform(img) if self.transform is not None else img
                img2 = self.transform2(img) if self.transform2 is not None else img
                return (img1, img2), label, domain

            except (UnidentifiedImageError, OSError, FileNotFoundError, ValueError) as e:
                last_err = e
                self._handle_read_error(rel_path, e)

                if self.missing_policy == "strict":
                    raise

                tries += 1
                if tries > self.max_resample_trials:
                    if self.missing_policy == "fallback" and self.allow_black_fallback:
                        self._log_bad("FALLBACK_RETURNED", last_err if last_err else Exception("unknown"))
                        return self._build_fallback()
                    raise RuntimeError(
                        f"Exceeded max_resample_trials={self.max_resample_trials} while reading samples. "
                        f"Last error: {type(last_err).__name__}: {last_err}"
                    )

                idx = random.randint(0, len(self.df) - 1)


def inspect_sample_paths(
    index_csv: str,
    data_root: str,
    split: Optional[str] = None,
    domains=None,
    paired_root: Optional[str] = None,
    sample_count: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    """Sample-check path existence before formal training.

    Returns both the new keys (checked/ok) and compatibility aliases
    (checked_count/ok_count) so old and new training scripts can both read it.
    """
    rng = random.Random(seed)
    df = pd.read_csv(index_csv)
    df = _filter_df(df, split=split, domains=domains)

    total = len(df)
    n = min(int(sample_count), total)
    picks = list(range(total))
    if n < total:
        picks = rng.sample(picks, n)

    missing_rgb: List[str] = []
    missing_aux: List[str] = []
    checked = 0
    ok = 0

    for idx in picks:
        row = df.iloc[int(idx)]
        rel_path = str(row["path"]).replace("\\", "/")
        rgb_path = os.path.join(data_root, rel_path)
        checked += 1
        if not os.path.isfile(rgb_path):
            if len(missing_rgb) < 20:
                missing_rgb.append(rgb_path)
            continue

        if paired_root:
            aux_path = os.path.join(paired_root, rel_path)
            if not os.path.isfile(aux_path):
                base, _ = os.path.splitext(rel_path)
                aux_png = os.path.join(paired_root, base + ".png")
                if not os.path.isfile(aux_png):
                    if len(missing_aux) < 20:
                        missing_aux.append(f"{aux_path} | {aux_png}")
                    continue
        ok += 1

    report = {
        "index_csv": index_csv,
        "split": split,
        "domains": domains,
        "data_root": data_root,
        "paired_root": paired_root,
        "checked": checked,
        "ok": ok,
        "ok_ratio": float(ok) / float(max(1, checked)),
        "missing_rgb_examples": missing_rgb,
        "missing_aux_examples": missing_aux,
        # compatibility aliases for older train_supervised.py versions
        "checked_count": checked,
        "ok_count": ok,
    }
    return report
