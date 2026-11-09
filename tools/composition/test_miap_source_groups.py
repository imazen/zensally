"""Source-alias tests guard against metadata split leakage and false joins."""
import base64
import unittest

from plan_miap_people_pilot import components,identity_keys,slot_role
from plan_miap_group_pilot import group_cell


class Sources(unittest.TestCase):
    def test_flickr_landing_and_resized_cdn_share_photo(self):
        landing=identity_keys({'source_url':'https://www.flickr.com/photos/public-name/12345678901/'})
        cdn=identity_keys({'OriginalURL':'http://c7.staticflickr.com/1/42/12345678901_ab12cd_z.jpg'})
        self.assertEqual(landing & cdn,{'flickr-photo:12345678901'})

    def test_author_profile_and_photo_alias(self):
        profile=identity_keys({'AuthorProfileURL':'http://www.flickr.com/people/12345@N06/'})
        photo=identity_keys({'OriginalLandingURL':'https://flickr.com/photos/12345@N06/987654321/'})
        self.assertIn('flickr-author:12345@N06',profile & photo)

    def test_license_and_human_names_do_not_join(self):
        self.assertEqual(identity_keys({'License':'https://creativecommons.org/licenses/by/2.0/','Author':'Same Name'}),set())

    def test_transitive_author_and_md5_group(self):
        digest=base64.b64encode(bytes(range(16))).decode()
        records=[{'id':'a','AuthorProfileURL':'https://flickr.com/people/111@N01/'},
                 {'id':'b','AuthorProfileURL':'https://flickr.com/people/111@N01/','OriginalMD5':digest},
                 {'id':'c','AuthorProfileURL':'https://flickr.com/people/222@N02/','OriginalMD5':digest},
                 {'id':'d','AuthorProfileURL':'https://flickr.com/people/333@N03/'}]
        self.assertEqual(components(records),[['a','b','c'],['d']])
        self.assertEqual(components(list(reversed(records))),components(records))

    def test_unsplash_native_case_and_stable_role(self):
        upper=identity_keys({'id':'unsplash:AbCDefGh123'})
        lower=identity_keys({'id':'unsplash:abcdefgh123'})
        self.assertFalse(upper & lower)
        self.assertEqual(slot_role('anchor'),slot_role('anchor'))

    def test_group_box_does_not_turn_crowd_into_two_people(self):
        self.assertEqual(group_cell(dict(individual_records=1,group_records=1,maximum_individual_area=.01,maximum_group_area=.9)),'group+individual/small')
        self.assertEqual(group_cell(dict(individual_records=0,group_records=1,maximum_individual_area=None,maximum_group_area=.9)),'group-only/unresolved')
        self.assertEqual(group_cell(dict(individual_records=2,group_records=0,maximum_individual_area=.04,maximum_group_area=None)),'individual-2/medium')


if __name__=='__main__':unittest.main()
