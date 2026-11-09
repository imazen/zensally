"""Bounded source-head relationships; predicted geometry, not social truth."""
import numpy as np

NAMES=('head_count','centers_retained','whole_heads_retained','pairs_retained','pairs_separated',
 'near_pairs_retained','near_pairs_separated','attention_pairs_retained','attention_pairs_separated',
 'group_center_x','group_center_y','group_span_x','group_span_y','group_edge_margin','mean_head_area','pair_distance')


def features(boxes,people):
    boxes=np.asarray(boxes,dtype=np.float64)
    if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all() or not ((boxes[:,:2]>=0).all() and (boxes[:,2:]<=1).all() and (boxes[:,2:]>boxes[:,:2]).all()):raise ValueError('normalized finite nonempty crop boxes required')
    if len(people)>16:raise ValueError('bounded head proposals required')
    heads=np.asarray([p['head'] for p in people],float).reshape(-1,4);targets=np.asarray([p['target'] for p in people],float).reshape(-1,2);confidence=np.asarray([p['in_frame_score'] for p in people],float)
    if not np.isfinite(heads).all() or not np.isfinite(targets).all() or not np.isfinite(confidence).all() or (heads<0).any() or (heads>1).any() or (heads[:,2:]<=heads[:,:2]).any() or (targets<0).any() or (targets>1).any() or (confidence<0).any() or (confidence>1).any():raise ValueError('invalid predicted head/gaze geometry')
    n=len(people);result=np.zeros((len(boxes),16),np.float64)
    if not n:return result.astype(np.float32)
    centers=(heads[:,:2]+heads[:,2:])/2
    i,j=np.where(~np.eye(n,dtype=bool));dist=np.linalg.norm(centers[i]-centers[j],axis=-1)
    proximity=np.exp(-dist**2/.25**2)
    attention=confidence[i]*np.exp(-np.sum((targets[i]-centers[j])**2,axis=-1)/.15**2)
    def weighted(value,weight):return float(np.sum(value*weight)/weight.sum()) if weight.sum()>0 else 0.
    for row,box in zip(result,boxes):
        extent=box[2:]-box[:2];kept=((centers>=box[:2])&(centers<=box[2:])).all(-1)
        whole=((heads[:,:2]>=box[:2])&(heads[:,2:]<=box[2:])).all(-1)
        both=kept[i]&kept[j];one=kept[i]^kept[j]
        row[:3]=[n/16,kept.mean(),whole.mean()]
        if len(i):row[3:9]=[both.mean(),one.mean(),weighted(both,proximity),weighted(one,proximity),weighted(both,attention),weighted(one,attention)]
        if kept.any():
            local=(centers[kept]-box[:2])/extent
            row[9:11]=local.mean(0);row[11:13]=np.ptp(local,axis=0)
            row[13]=np.minimum(local,1-local).min()*2
            row[14]=np.minimum(1,np.prod(heads[kept,2:]-heads[kept,:2],axis=-1)/np.prod(extent)).mean()
        if len(i) and both.any():row[15]=np.minimum(1,dist[both]/np.linalg.norm(extent)).mean()
    if not np.isfinite(result).all() or (result<0).any() or (result>1+1e-9).any():raise ValueError('relationship feature bounds differ')
    return result.astype(np.float32)
