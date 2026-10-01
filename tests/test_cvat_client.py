import json

import pytest
from omegaconf import OmegaConf

from src.labeling.cvat_client import CvatClient, CvatError, _batches


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._body


class FakeSession:
    """Records requests; answers each from the first route whose method matches
    and whose path the URL ends with. A route's response may be a callable that
    receives the recorded call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}
        self.auth = None

    def request(self, method, url, params=None, timeout=None, **kwargs):
        call = {"method": method, "url": url, "params": params, **kwargs}
        self.calls.append(call)
        for route_method, suffix, response in self.routes:
            if route_method == method and url.endswith(suffix):
                return response(call) if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")


def make_client(routes):
    session = FakeSession(routes)
    return CvatClient("https://cvat.example", 42, username="u", password="p", session=session), session


def test_every_request_carries_org_id():
    client, session = make_client(
        [("GET", "/api/labels", FakeResponse(200, {"results": [{"name": "vegetation", "id": 7}], "next": None}))]
    )
    assert client.label_ids(3) == {"vegetation": 7}
    call = session.calls[0]
    assert call["url"] == "https://cvat.example/api/labels"
    assert call["params"]["org_id"] == 42 and call["params"]["project_id"] == 3
    assert session.auth == ("u", "p")


def test_list_all_follows_pages():
    pages = {1: {"results": [{"id": 1}], "next": "x"}, 2: {"results": [{"id": 2}], "next": None}}
    client, _ = make_client([("GET", "/api/jobs", lambda call: FakeResponse(200, pages[call["params"]["page"]]))])
    assert [job["id"] for job in client.list_all("jobs")] == [1, 2]


def test_upload_protocol(tmp_path):
    def data_endpoint(call):
        headers = call.get("headers", {})
        if "Upload-Start" in headers:
            return FakeResponse(202)
        if "Upload-Multiple" in headers:
            return FakeResponse(200)
        if "Upload-Finish" in headers:
            return FakeResponse(202, {"rq_id": "rq1"})
        raise AssertionError(f"data request without an upload header: {headers}")

    client, session = make_client([
        ("POST", "/api/tasks/5/data/", data_endpoint),
        ("GET", "/api/requests/rq1", FakeResponse(200, {"status": "finished"})),
    ])
    paths = []
    for n in range(3):
        path = tmp_path / f"img{n}.jpg"
        path.write_bytes(b"x" * 40)
        paths.append(path)

    client.upload_images(5, paths, image_quality=95, max_request_bytes=100)

    posts = [c for c in session.calls if c["method"] == "POST"]
    assert list(posts[0]["headers"]) == ["Upload-Start"]
    bulk = posts[1:-1]
    # 3 x 40-byte files under a 100-byte request cap -> 2 bulk requests.
    assert [len(c["files"]) for c in bulk] == [2, 1]
    assert [name for name, _ in bulk[0]["files"]] == ["client_files[0]", "client_files[1]"]
    assert bulk[1]["files"][0][1][0] == "img2.jpg"
    assert "Upload-Finish" in posts[-1]["headers"] and posts[-1]["json"] == {"image_quality": 95, "sorting_method": "natural"}
    assert session.calls[-1]["url"].endswith("/api/requests/rq1")


def test_failed_request_raises():
    client, _ = make_client([("GET", "/api/requests/rq9", FakeResponse(200, {"status": "failed", "message": "bad zip"}))])
    with pytest.raises(CvatError, match="bad zip"):
        client.wait_for_request("rq9")


def test_http_errors_raise():
    client, _ = make_client([("GET", "/api/projects", FakeResponse(403, {"detail": "no"}))])
    with pytest.raises(CvatError, match="HTTP 403"):
        client.find_project("x")


def test_batches_rejects_oversized_file(tmp_path):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * 11)
    with pytest.raises(CvatError, match="upload limit"):
        list(_batches([big], max_bytes=10))


def test_from_config_reads_keys_file(tmp_path):
    keys = tmp_path / "keys.yaml"
    keys.write_text("cvat:\n  url: https://app.cvat.ai\n  username: me\n  password: pw\n  org_id: 12\n")
    ccfg = OmegaConf.create({"url": None, "org_id": None, "credentials_file": str(keys),
                             "request_timeout_s": 10, "poll_seconds": 1})
    client = CvatClient.from_config(ccfg)
    assert client.base == "https://app.cvat.ai" and client.org_id == 12
    assert client.session.auth == ("me", "pw")
