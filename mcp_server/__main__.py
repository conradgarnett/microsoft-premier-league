import sys

from .server import build_server

server = build_server()
if "--http" in sys.argv:
    port = int(sys.argv[sys.argv.index("--http") + 1])
    server.run("streamable-http", port=port)
else:
    server.run("stdio")
