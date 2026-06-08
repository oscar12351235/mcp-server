from pathlib import Path

ROOT = Path(".")
CONSTANT = 1


def load_config(path: str) -> dict[str, str]:
    return {"path": path}


class Worker:
    def run(self) -> str:
        return "done"
