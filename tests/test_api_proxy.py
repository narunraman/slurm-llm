"""The local OpenAI-compatible proxy, with a fake vLLM backend."""

import asyncio
import json
import os
import signal
import subprocess

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from slurm_llm.api_proxy import ActiveModel, OpenAIAPIProxy, PortAllocator, run_proxy_server
from slurm_llm.tunnels import TunnelHandle


def _run(coro):
    return asyncio.run(coro)


async def _with_proxy(check, backend_port=None):
    proxy = OpenAIAPIProxy(auto_activate=False)
    if backend_port:
        proxy.active_models["models--Qwen--Qwen3-8B"] = ActiveModel(
            model_name="models--Qwen--Qwen3-8B", job_id="1", cluster_name="cluster-a", node="g1",
            remote_port=30001, local_port=backend_port, served_name="Qwen3-8B",
        )
    async with TestClient(TestServer(proxy.app)) as client:
        return await check(client, proxy)


def test_requests_are_validated():
    async def check(client, proxy):
        assert (await client.post("/v1/chat/completions", json={})).status == 400
        missing = await client.post("/v1/chat/completions", json={"model": "no-such-model"})
        assert missing.status == 404
        inactive = await client.post("/v1/chat/completions", json={"model": "qwen-3-8b"})
        assert inactive.status == 503 and "activate" in (await inactive.json())["error"]["message"]

    _run(_with_proxy(check))


def test_request_is_forwarded_with_the_served_model_name():
    seen = {}

    async def completions(request):
        seen.update(await request.json())
        return web.json_response({"choices": [{"message": {"content": "hi"}}]})

    async def scenario():
        backend = web.Application()
        backend.router.add_post("/v1/chat/completions", completions)
        server = TestServer(backend)
        await server.start_server()
        try:
            async def check(client, proxy):
                models = (await (await client.get("/v1/models")).json())["data"]
                # listed under the short name the README tells clients to use, with the others as aliases
                assert models[0]["id"] == "qwen-3-8b"
                assert models[0]["hf_id"] == "Qwen/Qwen3-8B"
                assert models[0]["aliases"] == ["Qwen/Qwen3-8B", "models--Qwen--Qwen3-8B", "Qwen3-8B"]
                for name in [models[0]["id"], *models[0]["aliases"]]:  # every listed name routes
                    resp = await client.post("/v1/chat/completions", json={"model": name, "messages": []})
                    assert resp.status == 200, name
                    assert (await resp.json())["choices"][0]["message"]["content"] == "hi"

            await _with_proxy(check, backend_port=server.port)
        finally:
            await server.close()

    _run(scenario())
    assert seen["model"] == "Qwen3-8B"


def test_models_lists_instances_under_their_short_name():
    async def check(client, proxy):
        proxy.active_models["models--Qwen--Qwen3-8B:2"] = ActiveModel(
            model_name="models--Qwen--Qwen3-8B:2", job_id="2", cluster_name="cluster-a", node="g2",
            remote_port=30002, local_port=1, instance=2,
        )
        models = (await (await client.get("/v1/models")).json())["data"]
        assert (models[0]["id"], models[0]["instance"]) == ("qwen-3-8b:2", 2)
        assert "Qwen/Qwen3-8B:2" in models[0]["aliases"]
        assert proxy._routing_key("Qwen/Qwen3-8B:2")[0] == "models--Qwen--Qwen3-8B:2"

    _run(_with_proxy(check))


def test_tunnels_are_added_and_cancelled_on_the_shared_connection(monkeypatch, completed, fake_ssh, loader):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or completed(0))
    proxy = OpenAIAPIProxy(auto_activate=False)
    ssh = fake_ssh(loader.get_cluster("cluster-a"))
    proxy.ssh_manager.get_client = lambda name: ssh

    handle = proxy._create_tunnel("cluster-a", "g1", 30000, 35000)
    assert isinstance(handle, TunnelHandle)
    assert calls[0][-5:] == ["-O", "forward", "-L", "30000:g1:35000", "hpc1.example.org"]

    am = ActiveModel("k", "1", "cluster-a", "g1", 35000, 30000, tunnel=handle)
    proxy._close_tunnel(am)
    assert calls[1][-5:] == ["-O", "cancel", "-L", "30000:g1:35000", "hpc1.example.org"]
    # only the forward is cancelled; the master connection (shared with other tools) stays up
    assert all("exit" not in c for c in calls)


def test_port_allocator():
    ports = PortAllocator(30000, 30002)
    assert [ports.allocate("a"), ports.allocate("b")] == [30000, 30001]
    ports.free(30000)
    assert ports.allocate("c") == 30000


def _dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_proxy_records_its_forwards_in_the_tunnel_registry(monkeypatch, completed, fake_ssh, loader):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: completed(0))
    proxy = OpenAIAPIProxy(auto_activate=False)
    proxy.ssh_manager.get_client = lambda name: fake_ssh(loader.get_cluster("cluster-a"))

    handle = proxy._create_tunnel("cluster-a", "g1", 30000, 35000, model="models--Qwen--Qwen3-8B")
    entry = json.loads(proxy.tunnel_registry._registry_path.read_text())["tunnels"]["30000"]
    assert (entry["forward_spec"], entry["owner_pid"], entry["cluster"]) == ("30000:g1:35000", os.getpid(), "cluster-a")

    proxy._close_tunnel(ActiveModel("k", "1", "cluster-a", "g1", 35000, 30000, tunnel=handle))
    assert json.loads(proxy.tunnel_registry._registry_path.read_text())["tunnels"] == {}


def test_sigterm_stops_the_proxy_and_cancels_its_forwards(monkeypatch, completed, fake_ssh, loader):
    """`sllm kill` sends SIGTERM; the forwards on the shared SSH connection must be cancelled."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or completed(0))
    started = asyncio.Event()

    # a forward left by an earlier proxy that was killed with SIGKILL
    from slurm_llm.tunnels import TunnelRegistry

    TunnelRegistry().register_tunnel(
        TunnelHandle(30050, "30050:g9:35050", "hpc1.example.org", "/tmp/sock", (), _dead_pid()), "old", "cluster-a"
    )

    async def start_server(self):
        self.ssh_manager.get_client = lambda name: fake_ssh(loader.get_cluster("cluster-a"))
        handle = self._create_tunnel("cluster-a", "g1", 30000, 35000, model="m")
        self.active_models["m"] = ActiveModel("m", "1", "cluster-a", "g1", 35000, 30000, tunnel=handle)
        started.set()

        class Runner:
            async def cleanup(self):
                pass

        return Runner()

    monkeypatch.setattr(OpenAIAPIProxy, "start_server", start_server)

    async def scenario():
        server = asyncio.create_task(run_proxy_server(0))
        await asyncio.wait_for(started.wait(), 5)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(server, 5)

    _run(scenario())
    cancels = [c[-2] for c in calls if "cancel" in c]
    assert cancels == ["30050:g9:35050", "30000:g1:35000"]  # the stale one at start, ours at exit
    assert json.loads(TunnelRegistry()._registry_path.read_text())["tunnels"] == {}
    # the handlers are removed again
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
