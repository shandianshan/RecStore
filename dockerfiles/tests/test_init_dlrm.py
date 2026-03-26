import pathlib
import unittest


class InitDlrmScriptTest(unittest.TestCase):
    def test_builds_torchrec_stack_from_source_against_installed_torch(self) -> None:
        script = pathlib.Path('dockerfiles/init_dlrm.sh').read_text()

        self.assertNotIn('pip install fbgemm-gpu==1.0', script)
        self.assertNotIn('pip install torchrec==1.0', script)
        self.assertIn('https://github.com/pytorch/FBGEMM.git', script)
        self.assertIn('https://github.com/pytorch/torchrec.git', script)
        self.assertIn('asynchronous_complete_cumsum', script)
        self.assertIn('setuptools-git-versioning', script)
        self.assertIn('Using ninja binary', script)
        self.assertIn('CMAKE_MAKE_PROGRAM', script)
        self.assertIn('python_shims', script)
        self.assertIn('setup.py bdist_wheel', script)
        self.assertIn('--no-deps dist/*.whl', script)
        self.assertIn('patchelf --add-needed libtbb.so.12', script)
        self.assertIn('FBGEMM_REF="${FBGEMM_REF:-v1.1.2}"', script)
        self.assertIn('TORCHREC_REF="${TORCHREC_REF:-v1.1.0}"', script)


if __name__ == '__main__':
    unittest.main()
