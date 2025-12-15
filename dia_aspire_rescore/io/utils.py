from pathlib import Path


def find_ms_files(directory: str, file_type: str) -> dict[str, str]:
    """Find MS files in directory by extension. Returns {raw_name: file_path}."""
    ext_map = {"mzml": [".mzML", ".mzml"], "hdf5": [".hdf5", ".hdf"]}
    extensions = ext_map.get(file_type, [f".{file_type}"])
    files = []
    for ext in extensions:
        pattern = f"*{ext}"
        for file in Path(directory).glob(pattern):
            if file.is_file():
                files.append(str(file))
    return ms_files_to_dict(files)


def ms_files_to_dict(ms_files: list[str]) -> dict[str, str]:
    """Convert list of MS file paths to {raw_name: file_path} dict."""
    result = {}
    for f in ms_files:
        p = Path(f)
        # Remove all extensions (e.g., "sample.mzML.hdf5" -> "sample")
        raw_name = p.name
        while p.suffix:
            raw_name = p.stem
            p = Path(raw_name)
        result[raw_name] = f
    return result
