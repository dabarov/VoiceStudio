"""A server must not write past a range or a declared file size."""
import asyncio
import json

import httpx
import pytest

from services import segmented_download as sd


@pytest.mark.parametrize("chunk_size", [2, 1024 * 1024])
def test_overlong_range_cannot_corrupt_an_already_completed_segment(tmp_path, monkeypatch, chunk_size):
    dest = tmp_path / "weights"
    part = tmp_path / "weights.part"
    part.write_bytes(b"0000GOOD")
    (tmp_path / "weights.part.done").write_text(json.dumps({"size": 8, "done": [[4, 7]]}))
    monkeypatch.setattr(sd, "_plan_segments", lambda *_: [(0, 3), (4, 7)])
    monkeypatch.setattr(sd, "_READ_CHUNK", chunk_size)
    progress = []

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": "8", "accept-ranges": "bytes"})
        assert request.headers["range"] == "bytes=0-3"
        return httpx.Response(206, headers={"content-range": "bytes 0-3/8"}, content=b"BAD!EVIL")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError):
                await sd.segmented_download("https://example.test/model", str(dest), client=client,
                                            expected_size=8, on_bytes=progress.append)
    asyncio.run(run())
    assert part.read_bytes()[4:] == b"GOOD"
    assert sum(progress) <= 4
    assert not dest.exists()
    assert json.loads((tmp_path / "weights.part.done").read_text())["done"] == [[4, 7]]

    async def resume():
        def valid(request):
            if request.method == "HEAD":
                return respond(request)
            assert request.headers["range"] == "bytes=0-3"
            return httpx.Response(206, headers={"content-range": "bytes 0-3/8"}, content=b"DATA")
        async with httpx.AsyncClient(transport=httpx.MockTransport(valid)) as client:
            await sd.segmented_download("https://example.test/model", str(dest), client=client,
                                        expected_size=8)
    asyncio.run(resume())
    assert dest.read_bytes() == b"DATAGOOD"


@pytest.mark.parametrize("chunk_size", [2, 1024 * 1024])
def test_single_stream_checks_size_before_writing_an_overlong_chunk(tmp_path, monkeypatch, chunk_size):
    monkeypatch.setattr(sd, "_READ_CHUNK", chunk_size)
    dest = tmp_path / "weights"
    progress = []

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": "4"})
        return httpx.Response(200, content=b"TOO-LONG")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError):
                await sd.segmented_download("https://example.test/model", str(dest), client=client,
                                            expected_size=4, on_bytes=progress.append)
    asyncio.run(run())
    assert (tmp_path / "weights.part").stat().st_size <= 4
    assert sum(progress) <= 4
    assert not dest.exists()
