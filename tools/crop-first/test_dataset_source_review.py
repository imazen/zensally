import copy
import unittest
from verify_dataset_blobs import verify_review

class SourceReview(unittest.TestCase):
    def fixture(self):
        old=dict(png_sha256='old-png',scenario_family='same-family',registered_role='dev')
        new=dict(png_sha256='new-png',scenario_family='same-family',registered_role='dev',replaces_source_id='study-old')
        row=dict(stored_source_id='old',replacement_stored_source_id='new',source_id='study-old',png_sha256='old-png',
            positive_training_reference_eligible=False,positive_evaluation_reference_eligible=False)
        return dict(records=[row]),dict(old=old,new=new)
    def test_replacement_retains_split_and_old_source_is_not_a_positive(self):
        review,metadata=self.fixture();self.assertEqual(verify_review(review,metadata),1)
        for key,value in [('registered_role','fit'),('scenario_family','other')]:
            changed=copy.deepcopy(metadata);changed['new'][key]=value
            with self.assertRaisesRegex(ValueError,'family/role'):verify_review(review,changed)
    def test_override_must_bind_exact_original_and_exclude_positive_admission(self):
        for field,value in [('png_sha256','wrong-png'),('positive_training_reference_eligible',True),('positive_evaluation_reference_eligible',True)]:
            review,metadata=self.fixture();review['records'][0][field]=value
            with self.assertRaises(ValueError):verify_review(review,metadata)

if __name__=='__main__':unittest.main()

