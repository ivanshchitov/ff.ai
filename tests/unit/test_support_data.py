"""Selected, fresh, bounded MCP ticket data."""
import json
import pytest
from mcp_server import support_data

@pytest.fixture
def source(tmp_path):
    data = {'schema': 1, 'users': [{'id': 'U-1', 'name': 'Анна', 'email': 'private@example.test', 'password': 'hidden'}, {'id': 'U-2', 'name': 'Другой'}],
            'tickets': [{'id': 'T-1', 'user_id': 'U-1', 'status': 'open', 'provider': 'YandexDisk', 'description': 'access_token=abcdefghijk', 'password': 'hidden'}, {'id': 'T-2', 'user_id': 'U-2', 'description': 'Чужой тикет'}]}
    path = tmp_path / 'support.json'
    path.write_text(json.dumps(data))
    return tmp_path, path, data

def test_selected_fields_and_freshness(source):
    root, path, data = source
    result = support_data.get_ticket(root, 'support.json', 'T-1')
    assert result['ticket']['id'] == 'T-1'
    assert result['user'] == {'id': 'U-1', 'name': 'Анна'}
    text = json.dumps(result)
    assert 'hidden' not in text and 'abcdefghijk' not in text and 'Чужой' not in text
    data['tickets'][0]['status'] = 'resolved'
    path.write_text(json.dumps(data))
    assert support_data.get_ticket(root, 'support.json', 'T-1')['ticket']['status'] == 'resolved'

@pytest.mark.parametrize('mode', ['missing', 'invalid', 'schema', 'duplicate', 'orphan', 'large', 'symlink', 'output'])
def test_invalid_sources(source, mode):
    root, path, data = source
    if mode == 'missing': path.unlink()
    elif mode == 'invalid': path.write_text('{')
    elif mode == 'large': path.write_bytes(b' ' * (support_data.MAX_BYTES + 1))
    elif mode == 'symlink':
        actual = root / 'actual.json'
        path.rename(actual)
        path.symlink_to(actual)
    else:
        if mode == 'schema': data['schema'] = 2
        if mode == 'duplicate': data['users'].append(data['users'][0])
        if mode == 'orphan': data['tickets'][0]['user_id'] = 'unknown'
        if mode == 'output': data['tickets'][0]['description'] = 'x' * 17000
        path.write_text(json.dumps(data))
    with pytest.raises(support_data.SupportDataError):
        support_data.get_ticket(root, 'support.json', 'T-1')

@pytest.mark.parametrize('ticket', ['missing', '../support.json', '/tmp/file', ''])
def test_invalid_id(source, ticket):
    with pytest.raises(support_data.SupportDataError):
        support_data.get_ticket(source[0], 'support.json', ticket)

@pytest.mark.parametrize('path', ['../outside.json', '/tmp/outside.json'])
def test_path_boundary(source, path):
    with pytest.raises(support_data.SupportDataError):
        support_data.get_ticket(source[0], path, 'T-1')

def test_mcp_stdio_discovery_error_and_refresh(source):
    from core import domains, mcp_registry
    from core.mcp_client import MCPClient
    root, path, data = source
    client = MCPClient(mcp_registry.support_server_spec(root, domains.load_domain('product-support').support))
    connection = client.connect()
    assert connection.available, connection.error
    tools = {tool.name:tool for tool in connection.tools}
    assert set(tools) == {'support_health','support_get_ticket'}
    assert tools['support_get_ticket'].read_only is True
    with client.session() as session:
        result = session.call_tool('support_get_ticket', {'ticket_id':'T-1'})
        assert not result.is_error and json.loads(result.text)['ticket']['status']=='open'
        data['tickets'][0]['status']='resolved';path.write_text(json.dumps(data))
        assert json.loads(session.call_tool('support_get_ticket', {'ticket_id':'T-1'}).text)['ticket']['status']=='resolved'
        failed = session.call_tool('support_get_ticket', {'ticket_id':'unknown'})
        assert failed.is_error and 'Тикет не найден' in failed.text

def test_directory_replaced_by_symlink_before_open(source, tmp_path, monkeypatch):
    import os
    root, path, data=source
    nested=root/'nested';nested.mkdir();path.rename(nested/'support.json')
    outside=tmp_path/'outside';outside.mkdir();(outside/'support.json').write_text(json.dumps(data))
    original=os.open
    replaced=False
    def racing_open(name, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced and (str(name)=='nested' or Path(name).name=='support.json'):
            replaced=True
            nested.rename(root/'saved');nested.symlink_to(outside, target_is_directory=True)
        return original(name,flags,*args,**kwargs)
    from pathlib import Path
    monkeypatch.setattr(os,'open',racing_open)
    with pytest.raises(support_data.SupportDataError):
        support_data.get_ticket(root,'nested/support.json','T-1')

@pytest.mark.parametrize('assignment', ['refresh_token=abcdefghijklmnop', 'token: abcdefghijklmnop'])
def test_oauth_credentials_redacted_in_free_text(source, assignment):
    root,path,data=source
    data['tickets'][0]['description']=assignment
    path.write_text(json.dumps(data))
    assert 'abcdefghijklmnop' not in json.dumps(support_data.get_ticket(root,'support.json','T-1'))
