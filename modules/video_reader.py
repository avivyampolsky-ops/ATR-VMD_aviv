
_read_frame = None
_input_is_gray = None
_frame_shape = None


def _read_frame_color(cap):
    return cap.read()


def _read_frame_gray_2d(cap):
    ret, frame = cap.read()
    if not ret:
        return ret, None
    return ret, frame


def _read_frame_gray_3d(cap):
    ret, frame = cap.read()
    if not ret:
        return ret, None
    return ret, frame[:, :, 0]


def init_reader(cap, input_is_gray=None):
    global _read_frame, _input_is_gray, _frame_shape
    if input_is_gray is None:
        raise ValueError("input_is_gray must be True or False.")
    ret, frame = _read_frame_color(cap)
    if not ret:
        return ret, None

    frame_is_2d = frame.ndim == 2
    _input_is_gray = bool(input_is_gray)

    if _input_is_gray:
        _read_frame = _read_frame_gray_2d if frame_is_2d else _read_frame_gray_3d
        if not frame_is_2d:
            frame = frame[:, :, 0]
    else:
        _read_frame = _read_frame_color

    _frame_shape = frame.shape
    return ret, frame


def init_from_frame(frame, input_is_gray=None):
    global _read_frame, _input_is_gray, _frame_shape
    if input_is_gray is None:
        raise ValueError("input_is_gray must be True or False.")
    frame_is_2d = frame.ndim == 2
    _input_is_gray = bool(input_is_gray)

    if _input_is_gray:
        if not frame_is_2d:
            frame = frame[:, :, 0]
        _read_frame = None
    else:
        _read_frame = None

    _frame_shape = frame.shape
    return frame


def read_frame(cap):
    if _read_frame is None:
        raise RuntimeError("Video reader not initialized. Call init_reader() first.")
    return _read_frame(cap)


def input_is_gray():
    if _input_is_gray is None:
        raise RuntimeError("Video reader not initialized. Call init_reader() first.")
    return _input_is_gray


def frame_shape():
    return _frame_shape
