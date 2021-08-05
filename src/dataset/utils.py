from typing import Any, Generator, Tuple

def dfs_unpack_json(
    parent: Any, branch: str = "", separator: str = "."
) -> Generator[Tuple[str, str], None, None]:
    if isinstance(parent, dict):
        for key, value in parent.items():
            yield from dfs_unpack_json(
                parent=value,
                branch=branch + (separator if branch else "") + key,
                separator=separator,
            )
    elif type(parent) in [list, tuple]:
        for value in parent:
            yield from dfs_unpack_json(parent=value, branch=branch, separator=separator)
    else:
        yield branch, str(parent)
