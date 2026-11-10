import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path


class CliCaptureTests(unittest.TestCase):
    def fixture(self,root,prompts):
        label={'preferred':'A','a_acceptable':True,'b_acceptable':False,'face_cut_a':'no','face_cut_b':'no','reason':'Keep context'}
        images=[]
        for i in range(3):
            path=root/f'{i}.png';data=b'\x89PNG\r\n\x1a\n'+bytes([i]);path.write_bytes(data)
            images.append({'id':str(i),'path':str(path),'sha256':hashlib.sha256(data).hexdigest()})
        task={'task_id':'case','source_id':'fixture','source_role':'train','source':images[0],'candidates':images[1:]}
        taskfile=root/'tasks.json';taskfile.write_text(json.dumps([task]))
        def event(kind,**payload):return {'type':'event_msg','payload':{'type':kind,**payload}}
        def user(text):return {'type':'response_item','payload':{'type':'message','role':'user','content':[{'type':'input_text','text':text}]}}
        usage={'input_tokens':100,'cached_input_tokens':0,'output_tokens':10,'total_tokens':110,'cache_write_input_tokens':0}
        rows=[user('New bounded vision-annotation task from a previous turn'),event('token_count',info={'total_token_usage':usage}),event('task_complete',last_agent_message=json.dumps(label),turn_id='probe'),{'type':'turn_context','payload':{'model':'gpt-6.1-terra'}}]
        rows += [user(x) for x in prompts]
        rows += [event('token_count',info={'total_token_usage':{k:v*2 for k,v in usage.items()}}),event('task_complete',last_agent_message=json.dumps([{'task_id':'case',**label}]),turn_id='case')]
        rollout=root/'rollout.jsonl';rollout.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        output=root/'labels.json'
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('capture_cli.py')),str(rollout),str(taskfile),str(output)],capture_output=True,text=True)
        return result,output

    def test_midnight_environment_message_does_not_replace_annotation_prompt(self):
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            result,out=self.fixture(Path(name),['New bounded vision-annotation task. Inspect these candidates.','<environment_context>date changed</environment_context>'])
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(out.read_text());self.assertEqual(data['prompt'],'New bounded vision-annotation task. Inspect these candidates.')
            self.assertNotIn('date changed',out.read_text());self.assertFalse(data['records'][0]['training_eligible'])

    def test_old_prompt_cannot_be_reused_for_a_missing_current_prompt(self):
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            result,out=self.fixture(Path(name),['<environment_context>date changed</environment_context>'])
            self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists())

    def test_ambiguous_annotation_prompts_fail(self):
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            result,out=self.fixture(Path(name),['New bounded vision-annotation task A','New bounded vision-annotation task B'])
            self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
