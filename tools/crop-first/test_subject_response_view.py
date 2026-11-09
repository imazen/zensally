import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_subject_zones as fixtures
from direct_zones import sha
from derive_subject_response_view import main

class ResponseViews(unittest.TestCase):
    def test_protocol_finishes_before_marker_and_existing_pixels_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);cache=root/'cache';cache.mkdir();responses=root/'responses';responses.mkdir()
            task,result=fixtures.SubjectZones().fixture();task['role']='fit'
            source=root/'fixture.json';source.write_text(json.dumps(task));tasks=root/'tasks.json';tasks.write_text(json.dumps([str(source)]))
            response=fixtures.SubjectZones().response(result)
            (cache/'response.json').write_text(json.dumps(response))
            (cache/'metadata.json').write_text(json.dumps(dict(source_id='fixture',source_role='fit')))
            (cache/'hashes.json').write_text(json.dumps({n:sha((cache/n).read_bytes()) for n in ['response.json','metadata.json']}))
            (responses/'PROGRESS.json').write_text(json.dumps(dict(stage='complete',completed=1,required=1,errors=0)))
            (responses/'fixture.json').write_text(json.dumps(dict(entry=str(cache),annotation_valid=True)))
            output=root/'view';args=['derive',str(tasks),str(responses),str(output),'--source-prefix','fixture','--expected','1']
            with patch('sys.argv',args):main()
            protocol=json.loads((output/'protocol.json').read_text());self.assertEqual(protocol['parent_tasks_sha256'],sha(tasks.read_bytes()))
            before=(output/'tasks.json').read_bytes();marker=(output/'PROGRESS.json').read_bytes()
            with patch('sys.argv',args+['--verify-existing']):main()
            self.assertEqual((output/'tasks.json').read_bytes(),before)
            self.assertEqual((output/'PROGRESS.pre-provenance-repair.bak.json').read_bytes(),marker)
            self.assertEqual(json.loads((output/'PROGRESS.json').read_text())['new_model_api_calls'],0)
            (output/'tasks/fixture.json').write_text('{}')
            with patch('sys.argv',args+['--verify-existing']),self.assertRaisesRegex(ValueError,'view differs'):main()

if __name__=='__main__':unittest.main()

