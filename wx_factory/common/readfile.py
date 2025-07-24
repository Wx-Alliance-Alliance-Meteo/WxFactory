def readfile(filename: str) -> str:
    with open(filename) as f:
        content = "\n".join(f.readlines())

    return content
