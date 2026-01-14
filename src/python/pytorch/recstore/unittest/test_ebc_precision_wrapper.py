import unittest
import os
import sys
import argparse
import torch
import importlib.util
import subprocess

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

TEST_SCRIPTS_PATH = os.path.join(RECSTORE_PATH, 'test/scripts')
if TEST_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, TEST_SCRIPTS_PATH)

from ps_server_runner import ps_server_context
from ps_server_helpers import should_skip_server_start, get_server_config

TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), 'test_ebc_precision.py')
spec = importlib.util.spec_from_file_location("test_ebc_precision_module", TEST_MODULE_PATH)
test_ebc_precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_ebc_precision)

MP_TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), 'test_ebc_precision_multiprocess.py')
spec_mp = importlib.util.spec_from_file_location("test_ebc_precision_multiprocess_module", MP_TEST_MODULE_PATH)
test_ebc_precision_multiprocess = importlib.util.module_from_spec(spec_mp)
spec_mp.loader.exec_module(test_ebc_precision_multiprocess)

_server_runner = None
_test_result = None


def setUpModule():
    global _server_runner
    
    try:
        skip_server, reason = should_skip_server_start()
        if skip_server:
            print(f"\n[{reason}] Running tests without starting ps_server (assuming already running)\n")
            return
        
        config = get_server_config()
        
        print(f"\n{'='*70}")
        print("Starting PS Server for EBC Precision Tests")
        print(f"Server path: {config['server_path']}")
        print(f"Config: {config['config_path'] or 'default'}")
        print(f"Log dir: {config['log_dir']}")
        print(f"Timeout: {config['timeout']}s")
        print(f"{'='*70}\n")
        
        from ps_server_runner import PSServerRunner
        _server_runner = PSServerRunner(
            server_path=config['server_path'],
            config_path=config['config_path'],
            log_dir=config['log_dir'],
            timeout=config['timeout'],
            num_shards=config['num_shards'],
            verbose=True
        )
        
        if not _server_runner.start():
            raise RuntimeError("Failed to start PS Server")
    except Exception as e:
        print(f"\n❌ setUpModule failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def tearDownModule():
    global _server_runner
    
    if _server_runner is None:
        print("\nNo server runner to clean up")
        return
    
    try:
        print(f"\n{'='*70}")
        print("Stopping PS Server")
        print(f"{'='*70}\n")
        
        if _server_runner.is_running():
            if not _server_runner.stop():
                print("⚠️ Server stop returned False, but continuing...")
        
        print("✅ PS Server stopped gracefully\n")
    except Exception as e:
        print(f"\n⚠️ tearDownModule exception (non-fatal): {e}")
        import traceback
        traceback.print_exc()
        # Do NOT raise - we want tests to pass even if cleanup has issues
    finally:
        _server_runner = None


class TestEBCPrecision(unittest.TestCase):
    def test_basic_precision_cpu(self):
        print("\n" + "="*70)
        print("Running Basic EBC Precision Test (CPU)")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=1000,
            embedding_dim=128,  # Backend fixed to 128
            batch_size=64,
            seed=42,
            cpu=True
        )
        
        try:
            # Call the standalone test main function
            test_ebc_precision.main(args)
            print("\n✅ Basic precision test completed successfully")
        except AssertionError as e:
            self.fail(f"Basic precision test failed: {e}")
        except Exception as e:
            self.fail(f"Basic precision test raised unexpected exception: {e}")
    
    def test_small_batch_precision(self):
        print("\n" + "="*70)
        print("Running Small Batch EBC Precision Test (CPU)")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=500,
            embedding_dim=128,
            batch_size=16,
            seed=42,
            cpu=True
        )
        
        try:
            test_ebc_precision.main(args)
            print("\n✅ Small batch precision test completed successfully")
        except AssertionError as e:
            self.fail(f"Small batch precision test failed: {e}")
        except Exception as e:
            self.fail(f"Small batch precision test raised unexpected exception: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_precision(self):
        print("\n" + "="*70)
        print("Running CUDA EBC Precision Test")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=1000,
            embedding_dim=128,
            batch_size=64,
            seed=42,
            cpu=False
        )
        
        try:
            test_ebc_precision.main(args)
            print("\n✅ CUDA precision test completed successfully")
        except ImportError as e:
            self.skipTest(f"CUDA test skipped due to import error (likely FBGEMM): {e}")
        except AssertionError as e:
            self.fail(f"CUDA precision test failed: {e}")
        except Exception as e:
            self.fail(f"CUDA precision test raised unexpected exception: {e}")

    def test_multiprocess_precision(self):
        print("\n" + "="*70)
        print("Running Multiprocess EBC Precision Test (Subprocess)")
        print("="*70)
        
        # We run this as a subprocess to avoid multiprocessing context/pickling issues
        # when running under unittest discovery.
        cmd = [
            sys.executable, 
            MP_TEST_MODULE_PATH,
            "--num-embeddings", "1000",
            "--embedding-dim", "128",
            "--batch-size", "32",
            "--world-size", "2",
            "--cpu",
            "--seed", "42"
        ]
        
        print(f"Executing command: {' '.join(cmd)}")
        
        try:
            # check_call raises CalledProcessError if return code != 0
            subprocess.check_call(cmd)
            print("\n✅ Multiprocess precision test completed successfully")
        except subprocess.CalledProcessError as e:
            self.fail(f"Multiprocess precision test failed with exit code {e.returncode}")
        except Exception as e:
            self.fail(f"Multiprocess precision test raised unexpected exception: {e}")
