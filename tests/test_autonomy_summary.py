"""Scientific accounting gates for research summaries, no geometric claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_autonomy_run import checked_evaluation, summarize, mesh_shape_statistics


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root/'run/upper_val'
        self.folder.mkdir(parents=True)
        self.provenance = dict(split='val', training_epoch=5, checkpoint_sha256='c', config_sha256='f')
        self.result = dict(split='val', requested_case_count=2, successful_case_count=1, failed_case_count=1,
            score_mm=None, failures=['b'], budget={}, cases=[
                dict(case_id='a', patient_id='pa', split='val', status='EVALUATED_NOT_QUALITY_ACCEPTED',
                     geometry=dict(score_mm=2., per_tooth={'11':dict(observed_to_mesh_sample_mean_mm=1.),
                                                        '12':dict(observed_to_mesh_sample_mean_mm=3.)})),
                dict(case_id='b', patient_id='pb', split='val', status='FAILED', error='no finite zero surface',
                     components={'11':dict(points=8, own_sdf_abs_mean_model=.1)})])
        self.save()

    def save(self):
        (self.folder/'provenance.json').write_text(json.dumps(self.provenance))
        (self.folder/'evaluation.json').write_text(json.dumps(self.result))

    def tearDown(self): self.tmp.cleanup()

    def test_nested_batch_checkpoints_keep_distinct_evaluation_names(self):
        import shutil
        for name in ('baseline_common','candidate_common'):
            destination=self.root/'run'/name/'checkpoint_0000'
            shutil.copytree(self.folder,destination)
        result=summarize(self.root/'run',self.root/'summary')
        self.assertEqual({item['evaluation'] for item in result['evaluations']},
            {'upper_val','baseline_common/checkpoint_0000','candidate_common/checkpoint_0000'})

    def test_preserves_failed_cohort_and_component_diagnostics(self):
        result = summarize(self.root/'run', self.root/'summary')
        self.assertIsNone(result['evaluations'][0]['score_mm'])
        self.assertEqual(result['failure_counts'], {'no_zero_surface':1})
        import csv
        with (self.root/'summary/cases.csv').open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['case_id'], 'b')
        with (self.root/'summary/teeth.csv').open() as stream: teeth = list(csv.DictReader(stream))
        self.assertEqual(len(teeth), 3)
        self.assertEqual(teeth[-1]['geometry_metric_available'], 'False')

    def test_rejects_failed_mean_and_wrong_equal_tooth_denominator(self):
        self.result['score_mm'] = 2.
        self.save()
        with self.assertRaisesRegex(ValueError, 'failed cohort'):
            checked_evaluation(self.folder)
        self.result['score_mm'] = None
        self.result['cases'][0]['geometry']['score_mm'] = 1.
        self.save()
        with self.assertRaisesRegex(ValueError, 'equal-tooth'):
            checked_evaluation(self.folder)

    def test_rejects_test_before_opening_metric_content(self):
        self.provenance['split'] = 'test'
        self.save()
        (self.folder/'evaluation.json').write_text('must not parse this metric file')
        with self.assertRaisesRegex(ValueError, 'excludes test'):
            checked_evaluation(self.folder)

    def test_mesh_gum_and_component_diagnostics_are_area_weighted_in_mm(self):
        import numpy as np
        vertices = np.asarray([[0,0,0],[1,0,0],[0,1,0], [10,0,0],[12,0,0],[10,1,0]], dtype=float)
        mesh = self.root/'mesh.npz'
        np.savez(mesh, vertices_model=vertices, faces=np.asarray([[0,1,2],[3,4,5]]),
                 face_labels=np.asarray([0,11]), model_unit_mm=50.)
        result = mesh_shape_statistics(mesh)
        self.assertEqual(result['mesh_face_component_count'], 2)
        self.assertEqual(result['mesh_area_mm2'], 3750.)
        self.assertAlmostEqual(result['mesh_gum_area_fraction'], 1/3)
        self.assertAlmostEqual(result['mesh_largest_component_area_fraction'], 2/3)
