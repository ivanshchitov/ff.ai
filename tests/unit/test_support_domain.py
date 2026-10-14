import json
import shutil
import pytest
from core import domains, mcp_registry

def test_support_domain_and_registry(tmp_path, monkeypatch):
    domain = domains.load_domain('product-support')
    assert domain.support.data_path == 'support.json'
    assert domain.corpus.include_extensions == ('.md',)
    monkeypatch.delenv('FFAI_MCP_COMMAND', raising=False)
    monkeypatch.delenv('FFAI_MCP_URL', raising=False)
    specs = mcp_registry.registry(domain, tmp_path)
    assert any(spec.name == 'support' and '--data' in spec.args for spec in specs)
    assert domains.load_domain('aurora-qt5').support is None

@pytest.mark.parametrize('value', ['../outside.json', '/outside.json', '', 5])
def test_invalid_support_config(tmp_path, value):
    source = domains.load_domain('aurora-qt5').path
    shutil.copytree(source, tmp_path / 'aurora-qt5')
    (tmp_path / 'aurora-qt5' / 'support.json').write_text(json.dumps({'data_path': value, 'tool': 'support_get_ticket'}))
    with pytest.raises(domains.DomainSchemaError, match='support.json'):
        domains.load_domain('aurora-qt5', tmp_path)
