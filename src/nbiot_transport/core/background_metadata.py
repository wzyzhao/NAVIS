def background_info(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("background")
    return value if isinstance(value, dict) else {}


def background_required(metadata: dict | None) -> bool:
    info = background_info(metadata)
    return str(info.get("mode") or "") == "required"
