# -*- coding: utf-8 -*-
"""
EpisodicTaskSampler — 频域检测专用元学习采样器（带缓存 & 坏图容错，且保证 batch/label 对齐）

核心设计：
1. 首次运行时扫描 data_root/<domain>/train/{ai,nature} 下所有图片，并缓存到 _episodic_index_<dom>.pt。
2. 每次 episode：
   - 从 ai/nature 各采样一半 support/query 路径；
   - _load_batch 逐个安全加载，坏图打印 WARN 并跳过；
   - 根据“实际成功数量 Ks/Kq”构造标签，保证和 logit 的 batch_size 完全一致。
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

# 允许截断图像；不限制像素数，尽量读坏图
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class EpisodicTaskSampler:
    """
    假定数据结构：
      data_root/
        adm/
          train/ai/*.png
          train/nature/*.png
        sdv4/
          train/ai/*.png
          train/nature/*.png
        ...

    index 结构：
      self.index[dom] = {
          "ai": [path1, path2, ...],
          "nature": [path1, path2, ...]
      }
    """

    def __init__(
        self,
        data_root: str,
        domains: List[str],
        support_k: int,
        query_k: int,
        transform,
        image_size: int = 224,
        show_index_progress: bool = True,
    ):
        self.data_root = Path(data_root)
        self.domains = domains
        self.support_k = support_k
        self.query_k = query_k
        self.transform = transform
        self.image_size = image_size
        self.show_index_progress = show_index_progress

        # {domain: {"ai": [paths...], "nature": [paths...]}}
        self.index: Dict[str, Dict[str, List[str]]] = {}

        self._build_or_load_index()

    # -------------------------------------------------------------------------
    # 索引构建 & 缓存
    # -------------------------------------------------------------------------
    def _build_or_load_index(self):
        print("[Sampler] 数据根目录:", self.data_root)
        for dom in self.domains:
            cache_path = self.data_root / f"_episodic_index_{dom}.pt"
            if cache_path.exists():
                print(f"[Sampler] 发现缓存索引: {cache_path.name}，正在加载...")
                saved = torch.load(cache_path, map_location="cpu")

                # 兼容旧字段名（ai_paths / nature_paths）
                ai_list = saved.get("ai", None)
                nat_list = saved.get("nature", None)
                if ai_list is None and "ai_paths" in saved:
                    ai_list = saved["ai_paths"]
                if nat_list is None and "nature_paths" in saved:
                    nat_list = saved["nature_paths"]

                # 若缓存内容异常，强制重扫
                if not ai_list or not nat_list:
                    print(
                        f"[Sampler] 旧缓存字段不兼容或列表为空，重新扫描 {dom} ...",
                        flush=True,
                    )
                    self.index[dom] = self._scan_domain(dom)
                    torch.save(self.index[dom], cache_path)
                else:
                    self.index[dom] = {"ai": ai_list, "nature": nat_list}
            else:
                print(f"[Sampler] 未发现缓存索引，将重新扫描 {dom}...", flush=True)
                self.index[dom] = self._scan_domain(dom)
                torch.save(self.index[dom], cache_path)

            na = len(self.index[dom]["ai"])
            nn = len(self.index[dom]["nature"])
            print(f"[Sampler] {dom}: ai={na}  nature={nn}")

    def _scan_domain(self, domain: str) -> Dict[str, List[str]]:
        dom_root = self.data_root / domain / "train"
        res = {"ai": [], "nature": []}
        for cls in ["ai", "nature"]:
            cls_root = dom_root / cls
            if not cls_root.exists():
                print(f"[WARN] {cls_root} 不存在，跳过该类")
                continue

            paths = []
            exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
            for ext in exts:
                paths.extend(cls_root.rglob(ext))

            if self.show_index_progress:
                _ = [
                    p
                    for p in tqdm(
                        paths,
                        desc=f"[索引] {domain}/train/{cls}",
                        leave=True,
                    )
                ]
            res[cls] = [str(p) for p in paths]
        return res

    # -------------------------------------------------------------------------
    # 图像安全加载 & 批采样
    # -------------------------------------------------------------------------
    def _safe_load_image(self, path: str):
        """
        尝试打开一张图：
          - 出错：打印 WARN，返回 None
          - 成功：返回 transform(img)
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  # ToGray224 内部再转 L
                if self.transform is not None:
                    return self.transform(img)
                else:
                    import numpy as np
                    import torch as _torch

                    arr = np.asarray(img, dtype="float32") / 255.0
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    else:
                        arr = arr.transpose(2, 0, 1)
                    return _torch.from_numpy(arr)
        except Exception as e:
            print(f"[WARN] 读取图像失败，已跳过: {path} | {e}")
            return None

    def _load_batch(self, paths: List[str]) -> torch.Tensor:
        """
        简单策略：逐个尝试加载，坏图跳过，不做复杂重采样。
        保证：
          - 返回张数 = 实际成功加载的数量
          - 若全部失败，直接抛错提醒数据坏得太严重
        """
        xs = []
        for p in paths:
            img_t = self._safe_load_image(p)
            if img_t is None:
                continue
            xs.append(img_t)

        if len(xs) == 0:
            raise RuntimeError(
                f"[EpisodicTaskSampler] 尝试加载 {len(paths)} 张图像全部失败，"
                f"请检查数据集是否严重损坏。第一条路径: {paths[0] if paths else 'N/A'}"
            )

        return torch.stack(xs, dim=0)

    # -------------------------------------------------------------------------
    # 采样 episode
    # -------------------------------------------------------------------------
    def _sample_paths(
        self, domain: str, k_ai: int, k_nat: int
    ) -> List[str]:
        """
        从指定 domain 的 ai/nature 中采样指定数量的路径。
        采用有放回采样，避免“采完就没图”的问题。
        """
        idx = self.index[domain]
        ai_list = idx["ai"]
        nat_list = idx["nature"]

        if len(ai_list) == 0 or len(nat_list) == 0:
            raise RuntimeError(
                f"[EpisodicTaskSampler] domain={domain} 的 ai/nature 中有空列表，"
                "请确认数据集目录结构是否完整。"
            )

        ai_paths = random.choices(ai_list, k=k_ai)
        nat_paths = random.choices(nat_list, k=k_nat)
        return ai_paths + nat_paths

    def next_episode_cpu(
        self, domain: str = None
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          dom, xs_sup, ys_sup, xq, yq

        约定理想情况：
          - support_k 总数：self.support_k
          - query_k   总数：self.query_k
          - 一半为假(ai)=1，一半为真(nature)=0

        但遇到坏图时，会少若干张，因此标签长度以“实际成功数量”为准，保证与 batch 对齐。
        """
        if domain is None:
            domain = random.choice(self.domains)

        # 理想采样数（label 只按成功数量构造，不再死按这个值）
        k_sup_ai = self.support_k // 2
        k_sup_nat = self.support_k - k_sup_ai
        k_q_ai = self.query_k // 2
        k_q_nat = self.query_k - k_q_ai

        sup_paths = self._sample_paths(domain, k_sup_ai, k_sup_nat)
        que_paths = self._sample_paths(domain, k_q_ai, k_q_nat)

        xs_sup = self._load_batch(sup_paths)  # [Ks,C,H,W]，Ks ≤ support_k
        xq = self._load_batch(que_paths)      # [Kq,C,H,W]，Kq ≤ query_k

        Ks = xs_sup.shape[0]
        Kq = xq.shape[0]

        if Ks < 2 or Kq < 2:
            raise RuntimeError(
                f"[EpisodicTaskSampler] 成功样本过少：Ks={Ks}, Kq={Kq}，"
                f"domain={domain}，请检查数据集中是否存在大量坏图。"
            )

        # 按 1/2:1/2 拆分标签，保证 len(ys_sup) == Ks, len(yq) == Kq
        k_sup_ai_ok = Ks // 2
        k_sup_nat_ok = Ks - k_sup_ai_ok
        k_q_ai_ok = Kq // 2
        k_q_nat_ok = Kq - k_q_ai_ok

        ys_sup = torch.cat(
            [
                torch.ones(k_sup_ai_ok, dtype=torch.long),
                torch.zeros(k_sup_nat_ok, dtype=torch.long),
            ],
            dim=0,
        )
        yq = torch.cat(
            [
                torch.ones(k_q_ai_ok, dtype=torch.long),
                torch.zeros(k_q_nat_ok, dtype=torch.long),
            ],
            dim=0,
        )

        return domain, xs_sup, ys_sup, xq, yq
