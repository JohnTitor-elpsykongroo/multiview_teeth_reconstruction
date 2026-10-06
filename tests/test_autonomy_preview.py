"""Upper crowns must be viewed from inferior, lower crowns from superior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from preview_training_evaluation import preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.folder=self.root/'evaluation'
        artifact=self.folder/'case_0000'; artifact.mkdir(parents=True)
        xy=np.asarray([[-1.,-1.],[1.,-1.],[0.,1.]])
        vertices=np.vstack([np.column_stack([xy,[-.1]*3]),np.column_stack([xy,[.1]*3])])
        np.savez(artifact/'mesh.npz',vertices_model=vertices,faces=np.asarray([[0,1,2],[3,4,5]]),
            face_labels=np.asarray([11,0]),model_unit_mm=50,observed_points_model=np.empty((0,3)))
    def save(self,case_id):
        (self.folder/'evaluation.json').write_text(json.dumps(dict(cases=[dict(case_id=case_id,
            status='EVALUATED_NOT_QUALITY_ACCEPTED',artifact_directory='case_0000')])))
    def tearDown(self): self.temp.cleanup()
    def test_upper_inferior_lower_superior_painter_order(self):
        for arch in ('upper','lower'):
            self.save('case_'+arch)
            output=self.root/arch
            preview(self.folder,output)
            with Image.open(output/'case_0000.png') as picture: pixel=picture.getpixel((1000,260))
            if arch=='upper': self.assertNotEqual(pixel,(170,170,170))
            else: self.assertEqual(pixel,(170,170,170))
            metadata=json.loads((output/'preview.json').read_text())
            self.assertEqual(metadata['occlusal_direction'],'inferior' if arch=='upper' else 'superior')
    def test_ambiguous_and_conflicting_arch_rejected(self):
        self.save('case')
        with self.assertRaisesRegex(ValueError,'explicit --arch'): preview(self.folder,self.root/'ambiguous')
        self.save('case_upper')
        with self.assertRaisesRegex(ValueError,'disagrees'): preview(self.folder,self.root/'conflict','lower')


if __name__=='__main__':unittest.main()
