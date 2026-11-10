#!/usr/bin/env python3
"""Prepare identical cached inputs for a separate natural-frequency loss study."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if sha(a.baseline/'SHA256SUMS.json')!='3e603f65c94814b360fe8bee64484cffe07c39024ef32a4dda6e062c85a1698a':raise ValueError('original retention study differs')
    if sha(a.baseline/'features.pt')!='2b36c1215840dbc889481f6716eaddb5ba43d4b5af6fa434e3d1be6e91a9a3f1':raise ValueError('qualified feature cache differs')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['features.pt','qualification.json']:shutil.copy2(a.baseline/name,a.output/name)
    value=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        baseline_packet_sha256=sha(a.baseline/'SHA256SUMS.json'),feature_sha256=sha(a.output/'features.pt'),
        same_inputs=True,changed_axis='equal class means to natural candidate-frequency BCE only',
        threshold_unchanged=.5,selection='same original fourteen-family development Brier only',
        no_supplementary_selection=True,new_test_inference=False,new_teacher_calls=0,paid_model_api_calls=0)
    (a.output/'input-protocol.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
