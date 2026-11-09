#!/usr/bin/env python3
"""Preserve the bounded people-screen pipeline and its complete failed attempts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha

BASE = Path('/mnt/v/output/zensally')
INPUT = Path('/mnt/v/input/zensally')
DATE = '2026-11-09'
ROOTS = {
    'acquisition': INPUT/'miap-fit-development-gray'/DATE,
    'captions': BASE/'miap-content-captions'/DATE,
    'relationships': BASE/'miap-pilot-relationships'/DATE,
    'review': BASE/'miap-content-review'/DATE,
    'detectors': BASE/'miap-detector-screen'/DATE,
}
SOURCES = {
    'acquisition': ['acquire_miap_pilot.py', 'acquire_miap_witnesses.py', 'miap_owned_preview.py', 'check_miap_preview_parity.py'],
    'captions': ['caption_miap_content.py'],
    'relationships': ['screen_miap_relationships.py'],
    'review': ['build_miap_content_review.py', 'finalize_miap_content_review.py', 'unsplash_content_audit.html', 'unsplash_content_browser_check.mjs'],
    'detectors': ['prepare_miap_detector_screen.py', 'score_miap_detectors.py', 'test_miap_detector_matching.py'],
}
LOGS = {
    'acquisition': ['pilot-acquire', 'pilot-acquire-gray', 'pilot-parity', 'gray-probe', 'gray-readback'],
    'captions': ['content-witnesses', 'content-pilot'],
    'relationships': ['relationships'],
    'review': ['content-review-build', 'content-browser', 'content-finalize', 'legacy-content-browser'],
    'detectors': ['detectors-prepare', 'detectors', 'detectors-score'],
}


def read(root, name):
    return json.loads((root/name).read_text())


def legacy():
    original = BASE/'unsplash-content-audit'/DATE
    if sha(original/'SHA256SUMS.json') != '2440fb879fe7bd8df6dc0e6fb740459627bf5c221962a8fc2e07132f0e02ac6a':
        raise ValueError('legacy gallery identity differs')
    output = ROOTS['review']/'legacy-gallery'
    output.mkdir(exist_ok=False)
    data = read(original, 'sources.json')
    if len(data['sources']) != 119 or data['test_sources']:
        raise ValueError('legacy source contract differs')
    for row in data['sources']:
        source = original/row['preview']
        if sha(source) != row['preview_sha256']:
            raise ValueError('legacy preview differs')
        target = output/row['preview']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(original/'sources.json', output/'sources.json')
    template = Path(__file__).with_name('unsplash_content_audit.html').read_text()
    (output/'index.html').write_text(template.replace('/*DATA*/null', json.dumps(data).replace('</', '<\\/')))
    print(json.dumps(dict(legacy_sources=119, output=str(output))))


def validate(kind, root):
    if kind == 'acquisition':
        summary = read(root, 'summary.json')
        if summary['sources'] != 384 or summary['independent_png_readbacks'] != 384 or summary['source_modes'] != {'RGB':369, 'L':15}:
            raise ValueError('complete acquisition/readback required')
        for row in read(root, 'manifest.json')['rows']:
            directory = root/row['id'].split(':')[1]
            for name in ['preview.png', 'preview.rgb']:
                if sha(directory/name) != row[name+'_sha256']:
                    raise ValueError('acquired pixel identity differs')
            if sha(directory/'source.jpg') != row['download']['sha256']:
                raise ValueError('acquired JPEG differs')
        for name, source in [('prior-rgb-only-attempt', INPUT/'miap-fit-development'/DATE),
                             ('preview-parity', BASE/'miap-preview-parity'/DATE),
                             ('gray-probe', BASE/'miap-gray-probe'/DATE)]:
            shutil.copytree(source, root/name)
    elif kind == 'captions':
        summary = read(root, 'pilot-summary.json')
        if (summary['completed'], summary['invalid_calls'], summary['cached_calls']) != (384, 1, 8):
            raise ValueError('all cached calls, including capped failure, required')
        if len(list((root/'responses').glob('*.json'))) != 388:
            raise ValueError('complete unique response cache required')
    elif kind == 'relationships':
        summary = read(root, 'summary.json')
        if summary['pairs'] != 170688 or len(summary['flags']) != 6 or len(summary['flagged_new_sources']) != 7 or summary['exact_matches']:
            raise ValueError('complete relationship evidence required')
    elif kind == 'review':
        freeze = read(root, 'mask-freeze.json')
        if sha(root/'source-mask.json') != 'e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80':
            raise ValueError('independent source mask differs')
        if any(sha(root/name) != digest for name, digest in freeze['bindings'].items()):
            raise ValueError('review binding differs')
        legacy_report = read(root/'legacy-gallery', 'browser-unsplash-content.json')
        if legacy_report['decoded'] != 119 or legacy_report['exceptions']:
            raise ValueError('legacy compatibility replay required')
    elif kind == 'detectors':
        summary = read(root/'analysis', 'summary.json')
        if summary['primary_sources'] != 91 or summary['primary_individual_targets'] != 491 or len(summary['models']) != 5:
            raise ValueError('complete independently screened recall required')
        for model in summary['models']:
            result = root/'full'/model['id']/'results.json'
            if sha(result) != model['results_sha256'] or len(read(result.parent, result.name)) != 384:
                raise ValueError('complete fixed detector outputs required')


def freeze(kind):
    root = ROOTS[kind]
    if (root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    validate(kind, root)
    files = ['tools/composition/'+name for name in SOURCES[kind]]
    files += ['tools/composition/artifact_identity.py', 'tools/composition/freeze_miap_screen.py', 'justfile']
    if kind == 'review':
        files += ['tools/crop-first/browser_check.py']
    if kind == 'detectors':
        files += ['tools/runtime-eval/run_models.py', 'tools/runtime-eval/mediapipe_probe.py']
    producers = {}
    for name in files:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
        producers[name] = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', name], text=True).strip()
    if kind == 'captions':
        # Retain the exact generating revision as well as the later status correction.
        path = 'tools/composition/caption_miap_content.py'
        (root/'source/caption-generation-775bd2fc.py').write_bytes(subprocess.check_output(['git', 'show', '775bd2fc:'+path]))
    (root/'logs').mkdir()
    for prefix in LOGS[kind]:
        paths = list(Path('/home/lilith/tmp').glob('zensally-miap-'+prefix+'-20261109.*'))
        if not paths:
            raise ValueError('missing complete run log: '+prefix)
        for source in paths:
            shutil.copy2(source, root/'logs'/source.name)
    manifest = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), kind=kind,
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers, source_roles_changed=False, training_labels_admitted=False,
        human_artistic_labels=False, complete_face_coverage_certified=False, paid_model_api_calls=0,
        retained_caption_invalid_calls=1, detector_source_mask_frozen_before_scoring=True)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    total=sum(row['bytes'] for row in entries)
    limits={'acquisition':2*1024**3,'captions':1024**3,'relationships':64*1024**2,'review':768*1024**2,'detectors':256*1024**2}
    if total>limits[kind]:
        raise ValueError('packet storage budget exceeded')
    with (root/'SHA256SUMS.json').open('x') as stream:
        stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(kind=kind,members=len(entries),payload_bytes=total,manifest_sha256=sha(root/'SHA256SUMS.json'))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['legacy',*ROOTS])
    a=p.parse_args()
    if a.action=='legacy':
        legacy()
    else:
        freeze(a.action)


if __name__=='__main__':
    main()
