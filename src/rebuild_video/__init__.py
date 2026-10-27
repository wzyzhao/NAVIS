__all__ = ["rebuild_video"]


def rebuild_video(*args, **kwargs):
    from rebuild_video.video import rebuild_video as _rebuild_video

    return _rebuild_video(*args, **kwargs)
