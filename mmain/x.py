def left_pad(input, length, fill_character=' '):
    """
    Returns a string, which will be padded on the left with characters if necessary. If the input string is longer
    than the specified length, it will be returned unchanged.

    >>> left_pad('foo', 5)
    '  foo'

    >>> left_pad('foobar', 6)
    'foobar'

    >>> left_pad('toolong', 2)
    'toolong'

    >>> left_pad(1, 2, '0')
    '01'

    >>> left_pad(17, 5, 0)
    '00017'

    :param input: 
    :param length: The return string's desired length.
    :param fill_character: 
    :rtype str:
    """
    return str(input).rjust(length, str(fill_character))
