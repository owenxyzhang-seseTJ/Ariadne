from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import pytest
from ariadne.agent.planner import choose
from ariadne.agent import engine
from create_demo import write_model


class Server(BaseHTTPRequestHandler):
    def do_POST(self):
        content=self.rfile.read(int(self.headers['Content-Length']))
        self.server.requests.append(json.loads(content))
        if getattr(self.server, 'redirect_to', None):
            self.send_response(302)
            self.send_header('Location', self.server.redirect_to)
            self.end_headers()
            return
        decision=self.server.decisions.pop(0)
        payload=json.dumps({'choices':[{'message':{'content':json.dumps(decision)}}]}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        self.server.requests.append({'unexpected_redirect_get':self.path})
        payload=json.dumps({'choices':[{'message':{'content':'{"action":"pause"}'}}]}).encode()
        self.send_response(200);self.end_headers();self.wfile.write(payload)
    def log_message(self,*args):pass


@pytest.fixture
def endpoint():
    server=ThreadingHTTPServer(('127.0.0.1',0),Server);server.requests=[];server.decisions=[]
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    yield server,{'kind':'chat-completions','base_url':f'http://127.0.0.1:{server.server_port}/v1',
                  'model':'protocol-fixture','allow_local_http':True}
    server.shutdown();server.server_close();worker.join()


def test_chat_completions_protocol_returns_a_typed_decision(endpoint):
    server,config=endpoint;server.decisions=[{'action':'run','node':'a'}]
    result=choose(config,{'ready':['a']},{'actions':{}})
    assert result=={'action':'run','node':'a'}
    assert server.requests[0]['response_format']=={'type':'json_object'}


def test_model_can_propose_a_valid_registered_branch(tmp_path,endpoint):
    server,config=endpoint;model=tmp_path/'m.cif';write_model(model)
    request=dict(mode='custom',inputs={'cif':str(model)},planner=config,
                 nodes=[engine.node('original','crystal.audit',{'cif':engine.input_ref('cif')})])
    root=tmp_path/'session';engine.initialize(request,root)
    server.decisions=[{'action':'propose','node':engine.node('extra','crystal.audit',{'cif':engine.input_ref('cif')})},
                      {'action':'run','node':'extra'},{'action':'run','node':'original'}]
    result=engine.run(root)
    assert result['actions_used']==2 and result['decisions_used']==3
    assert all(n['status']=='succeeded' for n in result['nodes'])


def test_model_cannot_execute_nonready_nodes(tmp_path,endpoint):
    server,config=endpoint;model=tmp_path/'m.cif';write_model(model)
    root=tmp_path/'session';engine.initialize(dict(mode='audit',inputs={'cif':str(model)},planner=config),root)
    server.decisions=[{'action':'run','node':'package'}]
    with pytest.raises(ValueError,match='not ready'):engine.run(root)
    assert engine.status(root)['actions_used']==0


def test_model_cannot_propose_shell_execution(tmp_path,endpoint):
    server,config=endpoint;model=tmp_path/'m.cif';write_model(model)
    root=tmp_path/'session';engine.initialize(dict(mode='audit',inputs={'cif':str(model)},planner=config),root)
    server.decisions=[{'action':'propose','node':{'id':'bad','tool':'shell','args':{'command':'pwd'}}}]
    with pytest.raises(ValueError,match='unregistered'):engine.run(root)


def test_http_requires_explicit_loopback_permission():
    with pytest.raises(ValueError,match='HTTPS'):
        choose({'kind':'chat-completions','base_url':'http://example.org','model':'x'},{},{})


def test_secret_is_read_from_environment_not_in_persisted_request(tmp_path,endpoint,monkeypatch):
    server,config=endpoint;monkeypatch.setenv('ARIADNE_TEST_KEY','not-a-real-key-92381')
    config={**config,'api_key_env':'ARIADNE_TEST_KEY'}
    model=tmp_path/'m.cif';write_model(model)
    root=tmp_path/'session';engine.initialize(dict(mode='audit',inputs={'cif':str(model)},planner=config),root)
    server.decisions=[{'action':'pause','reason':'test pause'}];engine.run(root)
    assert 'not-a-real-key-92381' not in (root/'request.json').read_text()
    assert b'not-a-real-key-92381' not in (root/'agent.sqlite').read_bytes()


def test_mcp_reply_schema_is_explicit():
    from ariadne.agent.mcp_tools import Reply
    assert 'data' in Reply.model_json_schema()['properties']


def test_real_mcp_registration_when_installed():
    pytest.importorskip('mcp',reason='MCP runtime is not installed in the reconstruction container')
    from mcp.server.fastmcp import FastMCP
    from ariadne.agent.mcp_tools import register
    import asyncio
    mcp=FastMCP('test');register(mcp)
    tools=asyncio.run(mcp.list_tools())
    assert len(tools)==8 and all(t.outputSchema for t in tools)


def test_provider_redirect_is_not_followed(endpoint):
    server, config = endpoint
    server.redirect_to = f'http://127.0.0.1:{server.server_port}/unintended-destination'
    with pytest.raises(RuntimeError, match='planner request failed'):
        choose(config, {'ready':['a']}, {'actions':{}})
    assert len(server.requests) == 1
