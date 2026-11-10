#!/usr/bin/env python3
"""Exercise the standalone resizer's strided IPC, batches and bounds."""
import argparse,subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('executable');a=p.parse_args()
    w,h=9,7;packed=bytes((i*37+11)%256 for i in range(w*h*3))
    strided=b''.join(packed[y*w*3:(y+1)*w*3]+(b'\xfd'*5 if y<h-1 else b'') for y in range(h))
    def run(data,stride,spec,ok=True):
        r=subprocess.run([a.executable,str(w),str(h),str(stride),spec],input=data,capture_output=True)
        if ok:assert r.returncode==0,r.stderr.decode()
        else:assert r.returncode!=0 and not r.stdout
        return r.stdout
    expected=b''.join(packed[((y+1)*w+2)*3:((y+1)*w+7)*3] for y in range(5))
    assert run(packed,27,'2,1,5,5,5,5')==expected
    first='2,1,5,5,7,7';second='0,0,9,7,3,2'
    x=run(packed,27,first);assert len(x)==7*7*3
    assert x==run(strided,32,first)
    assert run(packed,27,first+';'+second)==x+run(packed,27,second)
    for spec in ['8,0,2,7,3,2','0,0,9,7,0,2','0,0,9,7,9999999,2']:
        run(packed,27,spec,False)
    run(packed[:-1],27,first,False)
    run(packed,26,first,False)
    print('preview oracle: packed/strided exact, batched/individual exact, bounds and input lengths rejected')


if __name__=='__main__':main()
