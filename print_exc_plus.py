import sys
import traceback

MAX_LINE_LENGTH = 120


def print_exc_plus():
    """
    Print the usual traceback information, followed by a listing of all the
    local variables in each frame.
    """
    tb = sys.exc_info()[2]

    stack = []

    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next
    for frame in stack:
        print("Frame %s in %s at line %s" % (frame.f_code.co_name,
                                             frame.f_code.co_filename,
                                             frame.f_lineno))
        for key, value in frame.f_locals.items():
            # We have to be careful not to cause a new error in our error
            # printer! Calling str() on an unknown object could cause an
            # error we don't want.

            # noinspection PyBroadException
            try:
                key_string = str(key)
            except Exception:
                key_string = "<ERROR WHILE PRINTING KEY>"

            # noinspection PyBroadException
            try:
                value_string = str(value)
            except Exception:
                value_string = "<ERROR WHILE PRINTING VALUE>"

            line: str = '\t' + key_string + ' : ' + str(type(value)) + ' = ' + value_string
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH - 1] + '...'
            print(line)

    traceback.print_exc()
