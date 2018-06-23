import os
import tempfile
import webbrowser
from urllib.request import pathname2url

from zerogercrnn.lib.constants import ENCODING

VIS_PACKAGE = os.path.dirname(os.path.abspath(__file__))
POPUP_CSS = os.path.join(VIS_PACKAGE, 'html_tools/popup.css')

htmlCodes = {
    "'": '&#39;',
    '"': '&quot;',
    '>': '&gt;',
    '<': '&lt;',
    '&': '&amp;',
    '\n': '</br>',
    '\t': '&emsp;',
    ' ': '&nbsp;'
}


def char_to_html(c):
    if str(c) in htmlCodes:
        return htmlCodes[str(c)]
    else:
        return c


def string_to_html(s):
    return ''.join([char_to_html(c) for c in s])


def show_html_page(page, save_file=None):
    """Show string *page* as an html in the browser. If save_file specified will save the page there."""
    html_path = save_file or os.path.join(tempfile.gettempdir(), 'diff.html')
    f = open(html_path, encoding=ENCODING, mode='w')
    f.write(page)
    f.close()
    webbrowser.open(url='file:{}'.format(pathname2url(html_path)))


class HtmlBuilder:
    HEAD = """
        <head>
            <link href="{}" rel="stylesheet" />
        </head>
    """.format(POPUP_CSS)

    BODY = """
        <body>
            <div class = "page">
                {}
            </div>
        </body>
    """

    def __init__(self):
        self.message = ""

    def add_popup(self, anchor, popup, background=None):
        """Add popup with two texts: one for anchor text and other for popup.
        You should not append any html elements to anchor or popup because this texts will be *converted*.
        """
        self.message += HtmlBuilder.get_popup_html(anchor, popup, background)

    def build(self):
        return HtmlBuilder.get_popup_html_page(self.message)

    @staticmethod
    def get_popup_html(anchor, popup, background=None):
        background = background or '#FFF'
        return """
            <span class = "popup-container">
                <span class = "anchor" style="background: {}">{}</span>
                <span class = "popup">{}</span>
            </span>
        """.format(background, string_to_html(anchor), string_to_html(popup))

    @staticmethod
    def get_popup_html_page(body):
        return HtmlBuilder.HEAD + HtmlBuilder.BODY.format(body)


if __name__ == '__main__':
    builder = HtmlBuilder()
    builder.add_popup(
        anchor="first",
        popup="First\nSecond",
        background="#81C784"

    )
    for i in range(100):
        builder.add_popup(
            anchor="second",
            popup="1\n2\n3",
            background="#EF9A9A"
        )
    show_html_page(
        page=builder.build()
    )
