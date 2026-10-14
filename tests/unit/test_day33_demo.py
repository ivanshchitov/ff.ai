"""The recording launcher may only use an empty, explicitly allocated workspace."""
from argparse import Namespace
import pytest
from scripts import day33_demo

def test_launcher_workspace_must_be_empty(tmp_path):
    (tmp_path/'user-data').write_text('preserve me')
    with pytest.raises(ValueError,match='пуст'):
        day33_demo.workspace_context(Namespace(visible=True,workspace=tmp_path))
    assert (tmp_path/'user-data').read_text()=='preserve me'

def test_launcher_workspace_is_preserved_for_recorder(tmp_path):
    with day33_demo.workspace_context(Namespace(visible=True,workspace=tmp_path)) as name:
        assert name==str(tmp_path.resolve())
        (tmp_path/'own-demo').write_text('demo')
    assert (tmp_path/'own-demo').exists()

def test_launcher_workspace_rejects_symlink(tmp_path):
    target=tmp_path/'target';target.mkdir()
    link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError):
        day33_demo.workspace_context(Namespace(visible=True,workspace=link))

def test_termination_unwinds_driver_cleanup(tmp_path):
    import signal
    import subprocess
    import sys
    code='''import signal, sys
from pathlib import Path
from scripts.day33_demo import install_signal_handlers, DemoCancelled
install_signal_handlers()
try:
    print('ready', flush=True)
    signal.pause()
except DemoCancelled:
    sys.exit(130)
finally:
    Path(sys.argv[1]).write_text('closed')
'''
    path=tmp_path/'closed'
    process=subprocess.Popen([sys.executable,'-c',code,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='ready'
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5)==130
        assert path.read_text()=='closed'
    finally:
        if process.poll() is None:
            process.kill();process.wait()
