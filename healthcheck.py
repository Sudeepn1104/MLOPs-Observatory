"""Container health check: passes if the API or the dashboard in this container answers.

Docker runs health checks as a separate process, so they can't see variables
set by the entrypoint. Probing both endpoints works for either service.
Proxies are bypassed: a corporate HTTP_PROXY must not intercept a localhost probe.
PORT is honoured, as hosts like Render choose the port.
"""
import os
import sys
import urllib.request

PORT = os.environ.get("PORT")
ENDPOINTS = [
    f"http://127.0.0.1:{PORT or 8000}/ready",
    f"http://127.0.0.1:{PORT or 8501}/_stcore/health",
]

# never route the local probe through an HTTP(S)_PROXY set in the environment
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main():
    for url in ENDPOINTS:
        try:
            with _opener.open(url, timeout=4) as r:
                if r.status == 200:
                    return 0
        except OSError:
            continue
    return 1


if __name__ == "__main__":
    sys.exit(main())
