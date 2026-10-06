"""Fixed geometry and point protocols for repeatability; no anatomical claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resample_autonomy_meshes import sample_metrics, resample


class MeshResamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.folder=self.root/'evaluation'
        artifact=self.folder/'case_0000'
        artifact.mkdir(parents=True)
        self.mesh=artifact/'mesh.npz'
        np.savez(self.mesh,vertices_model=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype=float),
            faces=np.asarray([[0,1,2]]),face_labels=np.asarray([11]),model_unit_mm=50,
            observed_points_model=np.asarray([[.1,.1,0],[.2,.2,0]],dtype=np.float32),observed_labels=np.asarray([11,11]))
        geometry=sample_metrics(self.mesh,['11'],32,4)
        self.result=dict(split='val',requested_case_count=1,successful_case_count=1,failed_case_count=0,failures=[],
            score_mm=geometry['score_mm'],budget=dict(mesh_samples_per_component=32,seed=4),cases=[
                dict(case_id='a',patient_id='pa',split='val',status='EVALUATED_NOT_QUALITY_ACCEPTED',
                    geometry=geometry,components={'0':{},'11':{}},artifact_directory='case_0000')])
        (self.folder/'provenance.json').write_text(json.dumps(dict(split='val',checkpoint_sha256='frozen')))
        self.save()

    def save(self): (self.folder/'evaluation.json').write_text(json.dumps(self.result))
    def tearDown(self): self.tmp.cleanup()

    def test_repeats_stored_metrics_and_only_mesh_seed_varies(self):
        first=sample_metrics(self.mesh,['11'],32,4)
        self.assertEqual(first,sample_metrics(self.mesh,['11'],32,4))
        result=resample(self.folder,self.root/'summary',samples=(32,128),seeds=(4,5,6))
        self.assertEqual(result['cases'][0]['stored_metric_parity'],'PASS')
        self.assertEqual(len(result['cohorts']),6)
        self.assertEqual(result['cases'][0]['budgets'][0]['seed_count'],3)
        self.assertIn(str(self.mesh.resolve()),result['input_sha256'])

    def test_failed_cases_keep_cohort_score_null(self):
        self.result.update(requested_case_count=2,failed_case_count=1,score_mm=None,failures=['b'])
        self.result['cases'].append(dict(case_id='b',patient_id='pb',split='val',status='FAILED',error='no finite zero surface'))
        self.save()
        result=resample(self.folder,self.root/'summary',samples=(32,),seeds=(4,))
        self.assertIsNone(result['cohorts'][0]['score_mm'])
        self.assertEqual(result['cases'][1]['status'],'FAILED')

    def test_missing_tooth_cannot_be_dropped(self):
        self.result['cases'][0]['components']['12']={}
        self.save()
        with self.assertRaisesRegex(ValueError,'omits a present tooth'):
            resample(self.folder,self.root/'summary')
