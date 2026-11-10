#!/usr/bin/env python3
"""Bounded API-only worker, using the configured key and shared immutable cache."""
import argparse,json,os,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from direct_zones import call_cached
import review

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('API concurrency must be 1..8')
    a.output.mkdir(parents=True,exist_ok=True);jobs=json.loads(a.plan.read_text());jobs=jobs[:a.limit] if a.limit else jobs;failures=[];done=0
    def one(job):
        task=json.loads(Path(job['task']).read_text());kind=job['kind'];path=a.output/(job['id']+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            # Revalidate via the content-addressed cache; never trust a progress file.
        kwargs={'request':review.build(task,job['model']),'validator':review.validate} if kind=='review' else {}
        result=call_cached(task,a.cache,**kwargs)
        if not path.exists():path.write_text(json.dumps(result,indent=2)+'\n')
        if not result['annotation_valid']:raise ValueError(f'{job["id"]}: completed response failed annotation validation: {result.get("validation_error")}')
        return job['id'],result['cache_hit'],result['network_requests']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                identity,hit,posts=future.result();done+=1;print(json.dumps({'id':identity,'cache_hit':hit,'network_requests':posts,'completed':done}),flush=True)
            except Exception as e:
                record={'id':job['id'],'error':str(e),'traceback':traceback.format_exc()};failures.append(record)
                (a.output/(job['id']+'.failure.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'failed':job['id'],'error':str(e)}),flush=True)
            (a.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':'running','completed':done,'required':len(jobs),'failed':len(failures),'checked_at':datetime.now(timezone.utc).isoformat()}))
    (a.output/'summary.json').write_text(json.dumps({'completed':done,'required':len(jobs),'failures':failures,'stage':'failed' if failures else 'complete'},indent=2)+'\n')
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
