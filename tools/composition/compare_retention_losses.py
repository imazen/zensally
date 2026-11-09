#!/usr/bin/env python3
"""Compare both frozen-loss studies without selecting a new model or threshold."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_unsplash_ranks import paired


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('natural',type=Path);p.add_argument('balanced',type=Path);a=p.parse_args()
    new=json.loads((a.natural/'analysis.json').read_text());old=json.loads((a.balanced/'analysis.json').read_text())
    if new['exact_selected_replays']!=9 or old['exact_selected_replays']!=9 or new['development_families']!=14 or old['development_families']!=14:raise ValueError('all original fourteen-family qualified comparisons required')
    result=dict(natural=new['mean_metrics'],balanced=old['mean_metrics'],
        paired_natural_minus_balanced={arm:paired(new['family_metrics'][arm],old['family_metrics'][arm]) for arm in ['image','geometry','permuted']},
        decisions={kind:{arm:{scope:{k:statistics.mean(r['metrics']['decisions'][scope][k] for r in analysis['runs'] if r['arm']==arm)
            for k in analysis['runs'][0]['metrics']['decisions'][scope]} for scope in ['all','essential','secondary']}
            for arm in ['image','geometry','permuted']} for kind,analysis in [('natural',new),('balanced',old)]},
        source_roles_changed=False,threshold_changed=False,model_promoted=False,selection_data=True,
        face_coverage_certified=False,new_test_inference=False,paid_model_api_calls=0)
    with (a.natural/'loss-comparison.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(natural=result['natural'],balanced=result['balanced'],image_paired=result['paired_natural_minus_balanced']['image']),indent=2))


if __name__=='__main__':main()
