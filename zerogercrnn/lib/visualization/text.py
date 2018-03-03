from zerogercrnn.lib.visualization.html_helper import char_to_html, string_to_html, show_html_page


def get_diff(text, actual):
    """Return two html-colored strings. Green if text[i] == actual[i], red otherwise"""

    assert (len(text) == len(actual))

    out_text = ''
    out_actual = ''

    green = '<span style="background: #4DB6AC; ">{}</span>'
    red = '<span style="background: #E57373; ">{}</span>'

    for i in range(len(text)):
        if text[i] == actual[i]:
            out_text += green.format(char_to_html(text[i]))
            out_actual += green.format(char_to_html(actual[i]))
        else:
            out_text += red.format(char_to_html(text[i]))
            out_actual += red.format(char_to_html(actual[i]))
    return out_text, out_actual


def show_diff(text, actual, file=None):
    """
    Shows difference between two strings in html.
    
    :param text: text got from some algorithm
    :param actual: actual text to compare with
    :param file: if specified html will be stored there
    """
    assert (len(text) == len(actual))
    diff_text, diff_actual = get_diff(text, actual)
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

    show_html_page(
        page=message,
        save_file=file
    )


def show_token_diff(predicted, actual, file=None):
    assert len(predicted) == len(actual)
    predicted_html = string_to_html(' '.join(predicted))
    actual_html = string_to_html(' '.join(actual))

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
            """.format(predicted_html, actual_html)

    show_html_page(
        page=message,
        save_file=file
    )


if __name__ == '__main__':
    show_diff(
        "  Here we are\te\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ["
        "\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥©ÂÃ",
        "  Hear We are\te\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ["
        "\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥©ÂÃ"
    )
