"""Fixed learned-code geometry, isolation and original-grid parity gates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from remesh_autonomy_train import train_artifact, parity, remesh
from inspect_autonomy_latents import sha
from dmm.provenance import source_fingerprint, decoder_contract
from training.validation import mesh_metrics


class Sphere(torch.nn.Module):
    labels = (0,11)
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
    def query(self, points, codes):
        return dict(sdf=points.norm(dim=-1)-.3+.01*codes[11].sum(),
                    weights=points.new_tensor([[.1,.9]]).expand(len(points),-1))


class RemeshingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def fixture(self):
        folder = self.root/'evaluation'
        folder.mkdir()
        codes = {0:torch.zeros(2),11:torch.tensor([.1,-.1])}
        sample = dict(points=torch.tensor([[.3,0,0],[0,.3,0]]),labels=torch.tensor([11,0]),components=(0,11))
        budget = dict(mesh_resolution=8,mesh_samples_per_component=32,seed=4)
        geometry = mesh_metrics(Sphere(),codes,sample,[[-.6]*3,[.6]*3],budget,artifact_dir=folder/'case_0000')
        saved = dict(source_sha256=source_fingerprint(),decoder_contract=decoder_contract(),arch='upper',epoch=5,
                     specs={},manifest_sha256='f'*64,canonical_reference_id='canonical',sampling_domain_model=[[-.6]*3,[.6]*3],
                     training_case_ids=['a'],case_rows=[dict(case_id='a',patient_id='pa',embedding_row=0,
                     split='train',augmentation_parent_id=None,presence={'11':True})],
                     system_state={'model.marker':torch.zeros(()),'latents.0.weight':codes[0][None],
                                   'latents.11.weight':codes[11][None]})
        checkpoint = self.root/'checkpoint.pth'
        torch.save(saved,checkpoint)
        (folder/'provenance.json').write_text(json.dumps(dict(split='train',checkpoint_sha256=sha(checkpoint),
            manifest_sha256=saved['manifest_sha256'],canonical_reference_id='canonical')))
        evaluation = dict(split='train',requested_case_count=1,successful_case_count=1,failed_case_count=0,failures=[],
            score_mm=geometry['score_mm'],budget=budget,cases=[dict(case_id='a',patient_id='pa',split='train',
            status='EVALUATED_NOT_QUALITY_ACCEPTED',code_mode='learned_train',geometry=geometry,
            components={'0':{},'11':{}},artifact_directory='case_0000')])
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        return folder,checkpoint,evaluation

    def test_heldout_rejected_before_geometry_or_checkpoint_read(self):
        folder = self.root/'heldout'
        folder.mkdir()
        for split in ('val','test'):
            (folder/'provenance.json').write_text(json.dumps(dict(split=split)))
            with self.assertRaisesRegex(ValueError,'only train'):
                remesh(self.root/'missing_checkpoint',folder,'a',self.root/'output')
        self.assertFalse((self.root/'output').exists())

    def test_real_grid_restores_learned_codes_and_checks_original_metrics(self):
        folder,checkpoint,evaluation = self.fixture()
        before = sha(checkpoint)
        with patch('networks.dmm_net.DMM',side_effect=lambda *a,**kw:Sphere()):
            result = remesh(checkpoint,folder,'a',self.root/'output',resolutions=(8,12))
        self.assertEqual(result['failed_grid_count'],0)
        self.assertEqual(result['grids'][0]['parity'],'PASS_1e-5mm')
        self.assertEqual(result['grids'][0]['geometry'],evaluation['cases'][0]['geometry'])
        self.assertEqual(sha(checkpoint),before)
        with np.load(self.root/'output/grid_12/codes.npz') as codes:
            np.testing.assert_array_equal(codes['code_11'],np.asarray([.1,-.1],dtype=np.float32))

    def test_checkpoint_swap_and_wrong_first_grid_rejected(self):
        folder,checkpoint,_ = self.fixture()
        with self.assertRaisesRegex(ValueError,'first grid'):
            remesh(checkpoint,folder,'a',self.root/'wrong_grid',resolutions=(12,))
        with checkpoint.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            remesh(checkpoint,folder,'a',self.root/'wrong_checkpoint',resolutions=(8,))

    def test_failed_original_parity_retains_partial_and_stops_other_grids(self):
        folder,checkpoint,evaluation = self.fixture()
        evaluation['cases'][0]['geometry']['per_tooth']['11']['observed_to_mesh_sample_p95_mm'] += 1
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        with patch('networks.dmm_net.DMM',side_effect=lambda *a,**kw:Sphere()):
            with self.assertRaisesRegex(ValueError,'original grid failed'):
                remesh(checkpoint,folder,'a',self.root/'failed',resolutions=(8,12))
        partial = json.loads((self.root/'failed/study_partial.json').read_text())
        self.assertEqual(partial['grids'][0]['status'],'FAILED')
        self.assertIn('parity failed',partial['grids'][0]['error'])
        self.assertFalse((self.root/'failed/grid_12').exists())


if __name__ == '__main__': unittest.main()
