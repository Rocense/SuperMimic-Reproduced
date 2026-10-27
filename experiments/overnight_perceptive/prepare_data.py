"""Build a subject-held-out walking set on one immutable terrain."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from supermimic.perception.paired import sha256
from supermimic.perception.tcrs import Synthesizer, TCRSCfg
from supermimic.perception.terrain import TerrainCfg, TerrainField

ROOT = Path('data/overnight_walk_20261027')
TERRAIN = TerrainCfg(enabled=True, kind='stairs', size=12, resolution=0.1, height=0.025,
                     stairs_surface='heightfield')  # Preserve the historical experiment geometry.
TCRS = TCRSCfg(acceptance_rule="legacy")  # Preserve the historical acceptance criteria.
FRAME_KEYS = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
              'body_lin_vel_w', 'body_ang_vel_w')


def prepare_sources(source):
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    specs = []
    for path in sorted(Path(source).glob('walk*.npz')):
        with np.load(path) as f:
            motion = {k: f[k] for k in f.files}
        pelvis = list(motion['body_names'].astype(str)).index('pelvis')
        split = 'validation' if 'subject5' in path.stem else 'train'
        for start in (200, 1200):
            for position, x in enumerate((-0.75, 0.75)):
                name = f'{path.stem}_f{start}_p{position}'
                raw = {k: v[start:start + 256].copy() if k in FRAME_KEYS else v.copy()
                       for k, v in motion.items()}
                shift = np.array([x, 0.0]) - raw['body_pos_w'][0, pelvis, :2]
                raw['body_pos_w'][..., :2] += shift
                directory = ROOT / split
                (directory / 'raw').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / 'raw' / f'{name}.npz', **raw)
                specs.append(dict(name=name, split=split, source=path.name,
                                  source_sha256=sha256(path), start_frame=start, frames=256,
                                  translation_xy=shift.tolist()))
    (ROOT / 'sources.json').write_text(json.dumps(specs, indent=2))
    print(f'Prepared {len(specs)} windows; subject 5 is validation only.', flush=True)


def synthesize(spec):
    torch.set_num_threads(1)
    directory = ROOT / spec['split']
    path = directory / 'raw' / (spec['name'] + '.npz')
    with np.load(path) as f:
        raw = {k: f[k] for k in f.files}
    adapted, metrics = Synthesizer(TerrainField(TERRAIN), TCRS).synthesize(raw)
    output = directory / ('adapted' if metrics['accepted'] else 'rejected') / path.name
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output, **adapted)
    return dict(spec, metrics=metrics, raw_sha256=sha256(path), adapted_sha256=sha256(output))


def generate():
    specs = json.loads((ROOT / 'sources.json').read_text())
    manifests = {split: dict(format='supermimic-tcrs-v1', terrain=asdict(TERRAIN),
                            tcrs=asdict(TCRS), split=split, clips=[])
                 for split in ('train', 'validation')}
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(synthesize, spec): spec for spec in specs}
        for future in as_completed(futures):
            record = future.result()
            manifest = manifests[record['split']]
            manifest['clips'].append(record)
            manifest['clips'].sort(key=lambda item: item['name'])
            (ROOT / record['split'] / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(record['name'], json.dumps(record['metrics']), flush=True)
    for split, manifest in manifests.items():
        count = sum(r['metrics']['accepted'] for r in manifest['clips'])
        if count < (8 if split == 'train' else 2):
            raise RuntimeError(f'Too few accepted {split} clips: {count}')
        print(split, count, 'accepted of', len(manifest['clips']), flush=True)
    (ROOT / 'ready.json').write_text(json.dumps({split: sum(r['metrics']['accepted'] for r in m['clips'])
                                               for split, m in manifests.items()}, indent=2))


if __name__ == '__main__':
    prepare_sources(sys.argv[1]) if len(sys.argv) > 1 else generate()
