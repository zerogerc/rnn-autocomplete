import os
import tempfile
import webbrowser
from urllib.request import pathname2url

from zerogercrnn.lib.utils.file import DEFAULT_ENCODING


def get_diff(text, actual):
    """Return two html-colored strings. Green if text[i] == actual[i], red otherwise"""

    assert (len(text) == len(actual))

    out_text = ''
    out_actual = ''

    green = '<span style="background: #4DB6AC; ">{}</span>'
    red = '<span style="background: #E57373; ">{}</span>'

    for i in range(len(text)):
        if text[i] == actual[i]:
            out_text += green.format(text[i])
            out_actual += green.format(actual[i])
        else:
            out_text += red.format(text[i])
            out_actual += red.format(actual[i])
    return out_text, out_actual


def show_diff(text, actual, file=None):
    """
    Shows difference between two strings in html.
    
    :param text: text got from some algorithm
    :param actual: actual text to compare with
    :param file: if specified html will be stored there
    """
    assert (len(text) == len(actual))
    html_path = file or os.path.join(tempfile.gettempdir(), 'diff.html')
    diff_text, diff_actual = get_diff(text, actual)

    f = open(html_path, encoding=DEFAULT_ENCODING, mode='w')
    message = """
        <html>
        <head>
        <style>
        .column {{
            float: left;
            width: 50%;
        }}
        
        /* Clear floats after the columns */
        .row:after {{
            content: "";
            display: table;
            clear: both;
        }}
        </style>
        </head>

        <body>
            <div class="row">
                <div class="column">
                    <h3 style="margin-top: 20px; ">Text</h3>
                    <div>{}</div>
                </div>
                <div class="column">
                    <h3 style="margin-top: 20px; ">Actual</h3>
                    <div>{}</div>
                </div>
            </div> 
        </body>
        </html>
        """.format(diff_text, diff_actual)

    f.write(message)
    f.close()

    webbrowser.open(url='file:{}'.format(pathname2url(html_path)))


if __name__ == '__main__':
    show_diff(
        "Here we are\t\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ["
        "\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥©ÂÃ",
        "Hear We are\t\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ["
        "\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥©ÂÃ"
    )
