#!/usr/bin/env python3
"""Own and clean up isolated browser/server processes for a read-only gallery check."""
import argparse,json,os,subprocess,time,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('chromium',type=Path);p.add_argument('scratch',type=Path);p.add_argument('--suite',choices=['main','audit','both','direct','fit','sunburst','composition','local-crops','preference','owned-screen','person-roi','pose-head-review','efficientdet','unsplash-rank','unsplash-purpose'],default='both');p.add_argument('--expected-sources',type=int,choices=[101,144]);a=p.parse_args()
    a.scratch.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'TMPDIR':str(a.scratch)}
    with (a.scratch/'http.log').open('w') as http_log,(a.scratch/'chromium.log').open('w') as chrome_log:
        server=subprocess.Popen(['python3','-m','http.server','3899','--bind','127.0.0.1','--directory',str(a.root)],stdout=http_log,stderr=subprocess.STDOUT,env=env)
        browser=subprocess.Popen([str(a.chromium),'--headless','--no-sandbox','--disable-dev-shm-usage','--remote-debugging-port=3994','--user-data-dir='+str(a.scratch/'profile'),'about:blank'],stdout=chrome_log,stderr=subprocess.STDOUT,env=env)
        try:
            for _ in range(100):
                if server.poll() is not None or browser.poll() is not None:raise RuntimeError('owned browser/server failed to start')
                try:
                    urllib.request.urlopen('http://127.0.0.1:3994/json/list',timeout=1).close();break
                except OSError:time.sleep(.1)
            suites=[('',448,'main'),('coverage-audit/',16,'audit')] if a.suite not in ['direct','fit'] else [('',4,'direct') if a.suite=='direct' else ('',96,'fit')]
            if a.suite=='sunburst':suites=[('',len(json.loads((a.root/'plan.json').read_text())),'sunburst')]
            if a.suite=='composition':suites=[('',8,'composition')]
            if a.suite=='local-crops':suites=[('',56,'local-crops')]
            if a.suite=='preference':suites=[('',5,'preference')]
            if a.suite=='owned-screen':suites=[('',101,'owned-screen')]
            if a.suite=='person-roi':suites=[('',18,'person-roi')]
            if a.suite=='pose-head-review':suites=[('',8,'pose-head-review')]
            if a.suite=='efficientdet':suites=[('',a.expected_sources or 101,'efficientdet')]
            if a.suite=='unsplash-rank':suites=[('',6,'unsplash-rank')]
            if a.suite=='unsplash-purpose':suites=[('',6,'unsplash-purpose')]
            for prefix,count,name in suites:
                if a.suite not in ['both',name]:continue
                checker=Path(__file__).with_name('direct_browser_check.mjs' if name=='direct' else 'fit_browser_check.mjs') if name in ['direct','fit'] else Path(__file__).with_suffix('.mjs')
                if name=='sunburst':checker=Path(__file__).with_name('sunburst_browser_check.mjs')
                if name=='composition':checker=Path(__file__).parent.parent/'composition/browser_check.mjs'
                if name=='local-crops':checker=Path(__file__).parent.parent/'composition/crop_browser_check.mjs'
                if name=='preference':checker=Path(__file__).parent.parent/'composition/preference_browser_check.mjs'
                if name=='owned-screen':checker=Path(__file__).parent.parent/'composition/owned_screen_browser_check.mjs'
                if name=='person-roi':checker=Path(__file__).parent.parent/'composition/person_roi_browser_check.mjs'
                if name=='pose-head-review':checker=Path(__file__).parent.parent/'composition/pose_head_review_browser_check.mjs'
                if name=='efficientdet':checker=Path(__file__).parent.parent/'composition/efficientdet_browser_check.mjs'
                if name=='unsplash-rank':checker=Path(__file__).parent.parent/'composition/unsplash_rank_browser_check.mjs'
                if name=='unsplash-purpose':checker=Path(__file__).parent.parent/'composition/unsplash_purpose_browser_check.mjs'
                report=a.root/f'browser-{name}.json'
                if report.exists():report.rename(report.with_name(report.name+'.bak-'+str(time.time_ns())))
                subprocess.run(['node',str(checker),'http://127.0.0.1:3994','http://127.0.0.1:3899/'+prefix+'index.html',str(count),str(report)],check=True,env=env)
        finally:
            for process in [browser,server]:
                if process.poll() is None:process.terminate()
            for process in [browser,server]:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':main()
