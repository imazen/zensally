#!/usr/bin/env python3
"""Copy frozen weak targets and exact initialization into a new experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['original','auxiliary','checkpoint','output']:
        p.add_argument(name,type=Path)
    p.add_argument('--experiment',choices=['matched','augmented'],default='matched')
    args = p.parse_args()
    expected = [
        (args.original/'extent-cache/features.pt','f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f'),
        (args.original/'extent-fits/protocol.json','a700c88fa9e84fa22e35dd127f1e564321f834e8023bc898b4c6e48153238a9d'),
        (args.original/'source-plan.json','6dc76f4d8730dcb91e17db659a5be330ebd18a88401b9caf2f594610590befa3'),
        (args.auxiliary/'auxiliary.pt','463e4884be5ada2128d08b9a755cf836d23e116059e6716390d64c0712b768a3'),
        (args.checkpoint,'910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71')]
    for path,digest in expected:
        if sha(path) != digest:
            raise ValueError('registered source/target/initializer differs: '+str(path))
    args.output.mkdir(parents=True,exist_ok=False)
    for relative in ['extent-cache/features.pt','extent-cache/admission.json',
                     'extent-fits/protocol.json','source-plan.json']:
        destination = args.output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.original/relative,destination)
    (args.output/'initializers').mkdir()
    shutil.copy2(args.checkpoint,args.output/'initializers/epoch11.pt')
    binding = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        original_packet_sha256='684d74cf1070af1378dc49b45bc14104c1e7be93356161b22dfbb0548ada3b3c',
        inputs=[dict(path=str(p),sha256=s) for p,s in expected],
        copied_mse_protocol='dependency for fixed source permutation only; no new MSE fits claimed',
        source_roles_changed=False,paid_model_api_calls=0,new_test_inference=False)
    with (args.output/'input-bindings.json').open('x') as stream:
        stream.write(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sources=56,fit=48,dev=8,registered_fits=6 if args.experiment=='augmented' else 15)))


if __name__ == '__main__':
    main()
