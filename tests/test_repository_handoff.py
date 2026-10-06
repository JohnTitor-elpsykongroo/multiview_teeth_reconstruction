"""Transfer contracts: Git identity binding and useful feedback on failure."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_training_package as verifier
from collect_training_feedback import collect


class RepositoryHandoffTests(unittest.TestCase):
    def test_git_identity_rejects_modified_and_untracked_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            def git(*args):
                return subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root), *args],
                                      check=True, capture_output=True, text=True)
            git('init', '-b', 'main')
            (root/'config.json').write_text('{}')
            git('add', 'config.json')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
            with patch.object(verifier, 'ROOT', root):
                identity = verifier.git_identity()
                self.assertEqual(identity['commit'], git('rev-parse', 'HEAD').stdout.strip())
                (root/'extra.py').write_text('pass')
                with self.assertRaisesRegex(ValueError, 'commit project changes'):
                    verifier.git_identity()
                (root/'extra.py').unlink()
                (root/'config.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'commit project changes'):
                    verifier.git_identity()

    def test_proof_binds_exact_git_commit_and_runtime(self):
        identity = dict(kind='git', commit='a', tree='b')
        proof = dict(status='TARGET_READY_FOR_BOUNDED_PILOT', source_sha256='source',
                     source_identity=identity, runtime={'torch':'fixture'})
        with patch.object(verifier, 'read_json', return_value=proof), \
             patch.object(verifier, 'source_fingerprint', return_value='source'), \
             patch('dmm.provenance.training_runtime', return_value={'torch':'fixture'}):
            verifier.verify_proof('fixture.json', identity)
            with self.assertRaisesRegex(ValueError, 'revision/config mismatch'):
                verifier.verify_proof('fixture.json', dict(kind='git',commit='c',tree='b'))

    def test_feedback_keeps_failure_and_inventory_but_excludes_weights_and_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root/'failed'; run.mkdir()
            (run/'failure.json').write_text('{"error":"fixture"}')
            (run/'final.pth').write_bytes(b'fixture weight')
            (run/'truth').mkdir(); (run/'truth/secret.json').write_text('{}')
            report = collect([run], [], root/'feedback.zip')
            self.assertEqual(report['status'], 'COLLECTED')
            self.assertEqual(len(report['runs'][0]['checkpoints'][0]['sha256']), 64)
            with zipfile.ZipFile(root/'feedback.zip') as archive:
                self.assertIn('run_00/failure.json', archive.namelist())
                self.assertFalse(any('truth' in n or n.endswith('.pth') for n in archive.namelist()))
            with self.assertRaisesRegex(ValueError, 'fresh'):
                collect([run], [], root/'feedback.zip')


if __name__ == '__main__':
    unittest.main()
