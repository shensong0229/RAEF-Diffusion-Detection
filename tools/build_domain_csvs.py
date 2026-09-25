from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLIT_NAMES = {'train', 'val', 'test'}

REAL_HINTS = {
    'real', '0_real', 'nature', 'natural', 'pristine', 'gt', 'clean'
}
FAKE_HINTS = {
    'fake', '1_fake', 'ai', 'generated', 'gen', 'synth', 'synthetic'
}
DEFAULT_FAKE_ONLY_DOMAINS = {
    'adm', 'glide', 'sd21', 'sdturbo', 'sdxl', 'vqdm'
}
MIXED_DOMAINS = {
    'sdv4', 'sdv5'
}


@dataclass
class SummaryRow:
    domain: str
    split: str
    real_count: int
    fake_count: int
    unknown_count: int
    total_count: int
    csv_path: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='快速扫描每个域并生成一个 CSV（Windows 友好，边扫边更新）。'
    )
    ap.add_argument('--data_root', required=True, type=str, help='例如 F:\\FakeImageDetect\\data')
    ap.add_argument(
        '--domains',
        default='glide,sd21,sdturbo,sdv4,sdv5',
        type=str,
        help='逗号分隔的域名列表；为空时自动扫描 data_root 下一层所有子目录。'
    )
    ap.add_argument('--out_dir', required=True, type=str, help='每个域一个 csv 的输出目录')
    ap.add_argument('--summary_csv', default='', type=str, help='可选：保存统计汇总 csv')
    ap.add_argument('--summary_json', default='', type=str, help='可选：保存统计汇总 json')
    ap.add_argument('--update_every', default=1000, type=int, help='每扫描多少张图更新一次进度后缀')
    return ap.parse_args()


def list_domains(data_root: Path, domains_arg: str) -> List[Path]:
    if domains_arg.strip():
        names = [x.strip() for x in domains_arg.split(',') if x.strip()]
        domains = [data_root / x for x in names]
    else:
        domains = [p for p in data_root.iterdir() if p.is_dir()]
    return sorted(domains, key=lambda x: x.name.lower())


def infer_split_from_rel_parts(rel_parts: Sequence[str]) -> str:
    for part in rel_parts:
        if part in SPLIT_NAMES:
            return part
    return 'all'


def infer_label_from_rel_parts(domain_name: str, rel_parts: Sequence[str]) -> int:
    for part in rel_parts:
        if part in REAL_HINTS:
            return 0
        if part in FAKE_HINTS:
            return 1

    if domain_name == 'real_shards' or domain_name in REAL_HINTS:
        return 0
    if domain_name in DEFAULT_FAKE_ONLY_DOMAINS:
        return 1
    if domain_name in MIXED_DOMAINS:
        return -1
    return -1


def _update_counts(counts: Dict[str, Dict[str, int]], split: str, label: int) -> None:
    if split not in counts:
        counts[split] = {'real': 0, 'fake': 0, 'unknown': 0}
    if 'all' not in counts:
        counts['all'] = {'real': 0, 'fake': 0, 'unknown': 0}

    if label == 0:
        counts[split]['real'] += 1
        counts['all']['real'] += 1
    elif label == 1:
        counts[split]['fake'] += 1
        counts['all']['fake'] += 1
    else:
        counts[split]['unknown'] += 1
        counts['all']['unknown'] += 1


def iter_image_relpaths(domain_root: Path) -> Iterable[str]:
    # os.walk 在 Windows 上通常比 Path.rglob('*') 更快，且不会为每个条目重复做 Path 对象/stat 开销。
    root_str = str(domain_root)
    for cur_root, _, files in os.walk(root_str):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            abs_path = os.path.join(cur_root, name)
            rel_from_domain = os.path.relpath(abs_path, root_str)
            yield rel_from_domain.replace('\\', '/')


def scan_one_domain(
    domain_root: Path,
    data_root: Path,
    out_dir: Path,
    update_every: int,
) -> Tuple[List[dict], List[SummaryRow]]:
    rows: List[dict] = []
    counts: Dict[str, Dict[str, int]] = {}

    domain_name = domain_root.name
    domain_name_lower = domain_name.lower()
    pbar = tqdm(
        desc=f'[scan] {domain_name}',
        unit='img',
        dynamic_ncols=True,
        leave=False,
        position=1,
        mininterval=0.5,
    )

    scanned = 0
    domain_prefix = f'{domain_name}/'

    for rel_from_domain in iter_image_relpaths(domain_root):
        rel_parts = [x.lower() for x in Path(rel_from_domain).parts[:-1]]
        split = infer_split_from_rel_parts(rel_parts)
        label = infer_label_from_rel_parts(domain_name_lower, rel_parts)
        _update_counts(counts, split, label)

        rows.append({
            'path': domain_prefix + rel_from_domain,
            'label': label,
            'domain': domain_name,
            'split': split,
        })

        scanned += 1
        pbar.update(1)
        if scanned <= 5 or scanned % max(1, update_every) == 0:
            all_counts = counts.get('all', {'real': 0, 'fake': 0, 'unknown': 0})
            pbar.set_postfix(
                real=all_counts['real'],
                fake=all_counts['fake'],
                unknown=all_counts['unknown'],
                refresh=False,
            )

    pbar.close()

    ordered_splits = [s for s in ['train', 'val', 'test', 'all'] if s in counts]
    other_splits = sorted([s for s in counts.keys() if s not in {'train', 'val', 'test', 'all'}])
    csv_path = out_dir / f'{domain_name}.csv'

    summary_rows: List[SummaryRow] = []
    for split in ordered_splits + other_splits:
        c = counts[split]
        total = c['real'] + c['fake'] + c['unknown']
        summary_rows.append(
            SummaryRow(
                domain=domain_name,
                split=split,
                real_count=c['real'],
                fake_count=c['fake'],
                unknown_count=c['unknown'],
                total_count=total,
                csv_path=str(csv_path),
            )
        )

    return rows, summary_rows


def write_domain_csv(rows: Sequence[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'label', 'domain', 'split'])
        w.writeheader()
        w.writerows(rows)


def save_summary(rows: Sequence[SummaryRow], summary_csv: Optional[str], summary_json: Optional[str]) -> None:
    payload = [asdict(r) for r in rows]

    if summary_csv:
        p = Path(summary_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(
                f,
                fieldnames=['domain', 'split', 'real_count', 'fake_count', 'unknown_count', 'total_count', 'csv_path']
            )
            w.writeheader()
            w.writerows(payload)
        print(f'[OK] saved summary csv: {p}')

    if summary_json:
        p = Path(summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            json.dump({'num_rows': len(payload), 'rows': payload}, f, ensure_ascii=False, indent=2)
        print(f'[OK] saved summary json: {p}')


def print_summary(rows: Sequence[SummaryRow]) -> None:
    if not rows:
        print('[WARN] no summary rows.')
        return
    print('\n===== Domain Scan Summary =====')
    print(f"{'domain':<12} {'split':<8} {'real':>8} {'fake':>8} {'unknown':>10} {'total':>8}")
    print('-' * 62)
    for r in rows:
        print(f"{r.domain:<12} {r.split:<8} {r.real_count:>8} {r.fake_count:>8} {r.unknown_count:>10} {r.total_count:>8}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'data_root not found: {data_root}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = list_domains(data_root, args.domains)
    if not domains:
        raise RuntimeError(f'No domain folders found under: {data_root}')

    print(f'[INFO] data_root = {data_root}')
    print(f'[INFO] out_dir = {out_dir}')
    print(f'[INFO] num_domains = {len(domains)}')
    for d in domains:
        print(f'  - {d.name}')

    all_summary_rows: List[SummaryRow] = []
    domain_pbar = tqdm(domains, desc='[domains]', unit='domain', dynamic_ncols=True, position=0)

    for domain_root in domain_pbar:
        domain_pbar.set_postfix(current=domain_root.name, refresh=False)

        if not domain_root.exists():
            print(f'[WARN] skip missing domain: {domain_root}')
            continue

        rows, summary_rows = scan_one_domain(
            domain_root,
            data_root=data_root,
            out_dir=out_dir,
            update_every=args.update_every,
        )
        if not rows:
            print(f'[WARN] no images found: {domain_root}')
            continue

        out_csv = out_dir / f'{domain_root.name}.csv'
        write_domain_csv(rows, out_csv)
        all_summary_rows.extend(summary_rows)
        print(f'[OK] wrote: {out_csv}  (rows={len(rows)})')

    domain_pbar.close()
    print_summary(all_summary_rows)
    save_summary(all_summary_rows, args.summary_csv or None, args.summary_json or None)


if __name__ == '__main__':
    main()
