"""Agent/CLI provenance and actual optional MCP transport integration.

The stdio transport test is opt-in because this test environment blocks the
socket wakeups used by AnyIO's reader thread.  It has also been run outside
that sandbox with the official MCP Python SDK 2.3.0:

    ARIADNE_MCP_STDIO_TEST=1 python -m pytest tests/test_agent.py -q

No test calls an official SHELX binary or treats process success as structure
validation.  The legacy policy test uses a mocked, explicitly allowed bridge.
"""

from __future__ import annotations

import asyncio
import copy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ariadne_xtal.agent import AgentWorkspace, serve_jsonl, sha256
from ariadne_xtal.cli import main
from ariadne_xtal.io import read_reflections, read_structure, write_reflections, write_structure
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities


@pytest.fixture
def project(tmp_path):
    truth = Structure(Cell(8, 9, 10), [
        Atom("C1", "C", [.13, .22, .31], u_iso=.024),
        Atom("O1", "O", [.37, .41, .55], u_iso=.031),
    ])
    hkl = np.array([(h, k, l) for h in range(1, 4) for k in range(-2, 3) for l in range(-2, 3)])
    flags = np.arange(len(hkl)) % 7 == 0
    observed = 1.7 * calculated_intensities(truth, hkl)
    data = Reflections(hkl, observed, np.full(len(hkl), 1.0), free_flags=flags)
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.035, -.027, .021]
    start.atoms[1].u_iso = .05
    write_structure(truth, tmp_path / "truth.json")
    write_structure(start, tmp_path / "start.json")
    write_reflections(data, tmp_path / "reflections.json")
    return tmp_path, truth, data


def _read_result(root, response):
    return json.loads((root / response["result"]).read_text())


def _read_manifest(root, response):
    return json.loads((root / response["manifest"]).read_text())


def test_isolated_runs_have_exact_input_snapshots_and_output_hashes(project):
    root, _, data = project
    before = {name: (root / name).read_bytes() for name in ("truth.json", "reflections.json")}
    workspace = AgentWorkspace(root)
    request = {"action": "inspect", "structure": "truth.json", "hkl": "reflections.json"}
    first, second = workspace.dispatch(request), workspace.dispatch(request)
    assert first["ok"] and second["ok"]
    assert first["run_id"] != second["run_id"]
    manifest = _read_manifest(root, first)
    assert manifest["status"] == "completed"
    assert manifest["scientific_acceptance"] == "not_assessed"
    run = (root / first["manifest"]).parent
    assert manifest["environment"]["gemmi"]
    for item in manifest["inputs"]:
        snapshot = run / item["snapshot"]
        assert snapshot.read_bytes() == before[item["path"]]
        assert item["sha256"] == sha256(snapshot)
    for item in manifest["outputs"]:
        assert item["sha256"] == sha256(run / item["path"])
    for name, content in before.items():
        assert (root / name).read_bytes() == content
    snapshot_hkl = next(run / item["snapshot"] for item in manifest["inputs"] if item["role"] == "hkl")
    assert np.array_equal(read_reflections(snapshot_hkl).free_flags, data.free_flags)


def test_inputs_changed_during_execution_fail_with_original_snapshot(project, monkeypatch):
    import ariadne_xtal.agent as agent

    root, _, _ = project
    original = (root / "truth.json").read_bytes()
    ordinary_read = agent.read_structure

    def mutate_after_snapshot(path):
        model = ordinary_read(path)
        (root / "truth.json").write_bytes(original + b"\n")
        return model

    monkeypatch.setattr(agent, "read_structure", mutate_after_snapshot)
    response = AgentWorkspace(root).dispatch({"action": "inspect", "structure": "truth.json"})
    assert response["ok"] is False
    assert "changed during" in response["error"]["message"]
    manifest = _read_manifest(root, response)
    assert manifest["status"] == "failed"
    run = (root / response["manifest"]).parent
    assert (run / manifest["inputs"][0]["snapshot"]).read_bytes() == original


@pytest.mark.parametrize("via_symlink", [False, True])
def test_workspace_input_escape_is_rejected(project, via_symlink):
    root, _, _ = project
    outside = root.parent / f"outside_{root.name}.json"
    outside.write_text((root / "truth.json").read_text())
    if via_symlink:
        (root / "escaped.json").symlink_to(outside)
        supplied = "escaped.json"
    else:
        supplied = str(outside)
    response = AgentWorkspace(root).dispatch({"action": "inspect", "structure": supplied})
    assert response["ok"] is False
    assert "within the agent workspace" in response["error"]["message"]


def test_symlink_run_directory_is_rejected_before_writing(project):
    root, _, _ = project
    outside = root.parent / f"runs_{root.name}"
    outside.mkdir()
    (root / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        AgentWorkspace(root).dispatch({"action": "inspect", "structure": "truth.json"})
    assert not list(outside.iterdir())


@pytest.mark.parametrize("payload", [
    {"action": "shell", "command": "true"},
    {"action": "capabilities", "unrecognized": True},
    {"action": "inspect", "structure": "truth.json", "options": {"unknown": 1}},
    {"action": "maps", "structure": "truth.json", "hkl": "reflections.json", "constraints": []},
    {"action": "refine", "structure": "truth.json", "hkl": "reflections.json", "scale": 2.0},
    [],
])
def test_unknown_actions_and_inapplicable_fields_are_rejected(project, payload):
    root, _, _ = project
    with pytest.raises(ValueError):
        AgentWorkspace(root).dispatch(payload)


def test_unknown_refinement_option_is_archived_as_failure(project):
    root, _, _ = project
    response = AgentWorkspace(root).dispatch({"action": "refine", "structure": "truth.json",
                                              "hkl": "reflections.json", "graphics": False,
                                              "options": {"invented_parameter": True}})
    assert response["ok"] is False
    assert "invented_parameter" in response["error"]["message"]
    assert _read_manifest(root, response)["status"] == "failed"


def test_failed_resource_budget_is_structured(project):
    root, _, _ = project
    response = AgentWorkspace(root, atom_limit=1).dispatch({"action": "inspect", "structure": "truth.json"})
    assert response["ok"] is False
    assert "atom budget" in response["error"]["message"]
    response = AgentWorkspace(root).dispatch({"action": "refine", "structure": "truth.json",
                                              "hkl": "reflections.json", "graphics": False,
                                              "options": {"max_nfev": 5001}})
    assert response["ok"] is False and "budget" in response["error"]["message"]


def test_jsonl_errors_do_not_desynchronize_following_requests(project):
    root, _, _ = project
    incoming = StringIO("\n{broken JSON\n" + json.dumps(["not an object"]) + "\n" +
                        json.dumps({"action": "capabilities", "ignored": 1}) + "\n" +
                        json.dumps({"action": "inspect", "structure": "truth.json"}) + "\n")
    outgoing = StringIO()
    serve_jsonl(AgentWorkspace(root), incoming, outgoing)
    lines = outgoing.getvalue().splitlines()
    assert len(lines) == 4
    responses = [json.loads(line) for line in lines]
    assert [r["ok"] for r in responses] == [False, False, False, True]
    assert responses[0]["error"]["type"] == "JSONDecodeError"
    assert _read_result(root, responses[-1])["asymmetric_unit_atoms"] == 2


def test_jsonl_oversized_request_is_rejected(project):
    root, _, _ = project
    incoming = StringIO(json.dumps({"action": "capabilities", "padding": "a" * 1_048_576}) + "\n")
    outgoing = StringIO()
    serve_jsonl(AgentWorkspace(root), incoming, outgoing)
    result = json.loads(outgoing.getvalue())
    assert result["ok"] is False and "1 MiB" in result["error"]["message"]


def test_holdout_json_survives_transport_and_never_fits_contaminated_holdouts(project):
    root, truth, reflections = project
    for atom in truth.atoms:
        atom.refine_xyz = False
        atom.refine_u = False
    write_structure(truth, root / "fixed.json")
    corrupted = copy.deepcopy(reflections)
    corrupted.intensity[corrupted.free_flags] += 10_000
    write_reflections(corrupted, root / "corrupted.json")
    workspace = AgentWorkspace(root)
    outputs = []
    for filename in ("reflections.json", "corrupted.json"):
        response = workspace.dispatch({"action": "refine", "structure": "fixed.json", "hkl": filename,
                                       "graphics": False, "options": {"max_nfev": 20}})
        assert response["ok"]
        outputs.append(_read_result(root, response))
    assert outputs[0]["scale"] == pytest.approx(1.7, abs=1e-8)
    assert outputs[1]["scale"] == pytest.approx(outputs[0]["scale"], abs=1e-10)
    assert outputs[0]["metrics"]["n"] == int(np.sum(~reflections.free_flags))
    assert outputs[0]["metrics"]["free"]["n"] == int(np.sum(reflections.free_flags))
    assert outputs[1]["metrics"]["free"]["sigma_rmse"] > 9000


def test_native_nonconvergence_is_separate_from_process_and_scientific_acceptance(project):
    root, _, _ = project
    response = AgentWorkspace(root).dispatch({"action": "refine", "structure": "start.json",
                                              "hkl": "reflections.json", "graphics": False,
                                              "options": {"max_nfev": 1, "scale": .8}})
    assert response["ok"] is True  # the declared computation and archiving completed
    assert response["summary"]["success"] is False  # the optimizer did not converge
    result = _read_result(root, response)
    assert result["success"] is False
    assert "maximum number" in result["message"].lower()
    manifest = _read_manifest(root, response)
    assert manifest["status"] == "completed"
    assert manifest["scientific_acceptance"] == "not_assessed"
    assert (root / response["result"]).parent.joinpath("refined.json").is_file()


def test_legacy_requires_startup_allowlist_and_passes_only_explicit_request(project, monkeypatch):
    import ariadne_xtal.compat as compat

    root, _, _ = project
    (root / "model.ins").write_text("TITL test\nHKLF 4\nEND\n")
    request = {"action": "legacy", "engine": "shelxl", "input": "model.ins"}
    response = AgentWorkspace(root).dispatch(request)
    assert response["ok"] is False
    assert "not enabled" in response["error"]["message"]
    recorded = {}

    def fake(engine, executable, input_path, output_dir, **kwargs):
        recorded.update(engine=engine, executable=executable, input_path=str(input_path), kwargs=kwargs)
        return {"status": "completed", "returncode": 0, "scientific_validation_performed": False}

    monkeypatch.setattr(compat, "run_legacy", fake)
    response = AgentWorkspace(root, {"shelxl": sys.executable}).dispatch({**request, "extra_args": ["-m1000"]})
    assert response["ok"] is True
    assert recorded["engine"] == "shelxl"
    assert recorded["executable"] == str(Path(sys.executable).resolve())
    assert recorded["kwargs"]["extra_args"] == ["-m1000"]
    assert _read_result(root, response)["scientific_validation_performed"] is False


def test_legacy_output_symlinks_are_not_hashed_or_read(project, monkeypatch):
    import ariadne_xtal.compat as compat
    import ariadne_xtal.agent as agent

    root, _, _ = project
    outside = root.parent / f"do_not_read_{root.name}.txt"
    outside.write_text("unrelated bytes")
    (root / "model.ins").write_text("TITL test\nHKLF 4\nEND\n")
    ordinary_hash = agent.sha256

    def protected_hash(path):
        assert not Path(path).is_symlink(), "agent must not dereference backend output symlinks"
        return ordinary_hash(path)

    def fake(engine, executable, input_path, output_dir, **kwargs):
        (Path(output_dir) / "unsafe_output.txt").symlink_to(outside)
        return {"status": "completed", "returncode": 0, "scientific_validation_performed": False}

    monkeypatch.setattr(compat, "run_legacy", fake)
    monkeypatch.setattr(agent, "sha256", protected_hash)
    response = AgentWorkspace(root, {"shelxl": sys.executable}).dispatch(
        {"action": "legacy", "engine": "shelxl", "input": "model.ins"})
    assert response["ok"] is True
    assert all(item["path"] != "unsafe_output.txt" for item in _read_manifest(root, response)["outputs"])


def test_cli_returns_json_exit_status_and_run_paths(project, capsys):
    root, _, _ = project
    assert main(["inspect", "truth.json", "--hkl", "reflections.json", "--root", str(root)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is True
    assert (root / response["manifest"]).exists()
    assert main(["inspect", "missing.json", "--root", str(root)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert main(["capabilities"]) == 0
    assert json.loads(capsys.readouterr().out)["native_full_shelx_parity"] is False


def test_cli_module_entrypoint_has_machine_readable_output(project):
    root, _, _ = project
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run([sys.executable, "-m", "ariadne_xtal.cli", "inspect", "truth.json", "--root", str(root)],
                            capture_output=True, text=True, timeout=20,
                            env={**os.environ, "PYTHONPATH": str(source)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_official_mcp_sdk_creates_and_lists_typed_tools(project):
    pytest.importorskip("mcp")
    from ariadne_xtal.mcp_server import create_server

    root, _, _ = project
    server = create_server(root)
    tools = asyncio.run(server.list_tools())
    serialized = [tool.model_dump(by_alias=True) for tool in tools]
    assert {tool["name"] for tool in serialized} == {"crystal_capabilities", "crystal_execute"}
    execute = next(tool for tool in serialized if tool["name"] == "crystal_execute")
    assert execute["inputSchema"]["required"] == ["request"]
    assert execute["inputSchema"]["properties"]["request"]["type"] == "object"


@pytest.mark.skipif(os.environ.get("ARIADNE_MCP_STDIO_TEST") != "1",
                    reason="Set ARIADNE_MCP_STDIO_TEST=1 for the local MCP stdio integration test")
def test_actual_mcp_stdio_handshake_list_and_scientific_tool(project):
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    root, _, _ = project
    source = Path(__file__).resolve().parents[1] / "src"

    async def exercise():
        with anyio.fail_after(20):
            parameters = StdioServerParameters(command=sys.executable,
                args=["-m", "ariadne_xtal.mcp_server", "--root", str(root)],
                cwd=str(root), env={**os.environ, "PYTHONPATH": str(source)})
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    initialization = await client.initialize()
                    wire = initialization.model_dump(by_alias=True)
                    assert wire["serverInfo"]["name"] == "Ariadne-Xtal"
                    listing = await client.list_tools()
                    assert {tool.name for tool in listing.tools} == {"crystal_capabilities", "crystal_execute"}
                    returned = await client.call_tool("crystal_execute", {"request": {
                        "action": "inspect", "structure": "truth.json", "hkl": "reflections.json"}})
                    payload = returned.model_dump(by_alias=True)
                    assert not payload.get("isError", False)
                    result = payload.get("structuredContent")
                    if result is None:
                        result = json.loads(next(c["text"] for c in payload["content"] if c["type"] == "text"))
                    assert result["ok"] is True
                    assert _read_result(root, result)["asymmetric_unit_atoms"] == 2
                    assert _read_manifest(root, result)["scientific_acceptance"] == "not_assessed"

    anyio.run(exercise)
