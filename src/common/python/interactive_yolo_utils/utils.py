import os

def workspace_dir():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    workspace_dir, tail = os.path.split(workspace_dir)
    while tail not in ["src", "venv", "install", "build"]:
        workspace_dir, tail = os.path.split(workspace_dir)

    return workspace_dir
