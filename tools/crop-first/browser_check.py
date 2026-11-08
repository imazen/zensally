#!/usr/bin/env python3
"""Own and clean up isolated browser/server processes for a read-only gallery check."""
import argparse,os,subprocess,time,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('chromium',type=Path);p.add_argument('scratch',type=Path);p.add_argument('--suite',choices=['main','audit','both','direct'],default='both');a=p.parse_args()
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
            suites=[('',448,'main'),('coverage-audit/',16,'audit')] if a.suite!='direct' else [('',4,'direct')]
            for prefix,count,name in suites:
                if a.suite not in ['both',name]:continue
                checker=Path(__file__).with_name('direct_browser_check.mjs') if name=='direct' else Path(__file__).with_suffix('.mjs')
                subprocess.run(['node',str(checker),'http://127.0.0.1:3994','http://127.0.0.1:3899/'+prefix+'index.html',str(count),str(a.root/f'browser-{name}.json')],check=True,env=env)
        finally:
            for process in [browser,server]:
                if process.poll() is None:process.terminate()
            for process in [browser,server]:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':main()
