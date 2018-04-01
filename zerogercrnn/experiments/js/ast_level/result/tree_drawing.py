import json
import os

from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

DIR_DATASET = '/Users/zerogerc/Documents/datasets/js_dataset.tar'
FILE_TRAINING_DATASET = os.path.join(DIR_DATASET, 'programs_eval.json')
FILE_TRAINING_PROCESSED = os.path.join(DIR_DATASET, 'programs_processed_eval.json')

ENCODING = 'ISO-8859-1'

RESULTS_DIR = '/Users/zerogerc/Documents/diploma/results_tree/'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'eval_prediction.json')

NT_TARGET = 'nt_target'
T_TARGET = 't_target'
NT_PREDICTION = 'nt_prediction'
T_PREDICTION = 't_prediction'


def get_numbered_tree_recursively(node_id, json):
    current = json[node_id]
    if 'children' not in current:
        return '{}'.format(node_id)
    else:
        child_strs = []
        for child in current['children']:
            child_strs.append(get_numbered_tree_recursively(child, json))

        return '({}){}'.format(','.join(child_strs), node_id)


class TreeDrawer():
    def __init__(self, tree_json, prediction):
        self.tree_json = tree_json
        self.prediction = prediction

    def draw_tree(self):
        str_tree = '{};'.format(get_numbered_tree_recursively(0, self.tree_json))

        t = Tree(str_tree, format=1)

        ts = TreeStyle()
        ts.show_leaf_name = False

        def my_layout(node):
            json_node_id = int(node.name)
            json_node = self.tree_json[json_node_id]

            # Type
            if self.prediction[NT_TARGET][json_node_id] == self.prediction[NT_PREDICTION][json_node_id]:
                color = 'green'
            else:
                color = 'red'

            text_type = TextFace(json_node['type'], fgcolor=color, tight_text=False)
            add_face_to_node(text_type, node, column=0, position='branch-top')

            # Value
            # TODO: think if I need to show EMPTY tokens predicions here
            if 'value' in json_node:
                if self.prediction[T_TARGET][json_node_id] == self.prediction[T_PREDICTION][json_node_id]:
                    color = 'green'
                else:
                    color = 'red'
                text_value = TextFace(json_node['value'], fsize=14, fgcolor=color, tight_text=False)
                add_face_to_node(text_value, node, column=0, position='branch-right')

        ts.layout_fn = my_layout
        t.show(tree_style=ts)


def main(data_path, prediction_path):
    f_read_prediction = open(prediction_path, mode='r', encoding=ENCODING)

    predictions = []
    for l in f_read_prediction:
        predictions.append(json.loads(l))

    f_read_data = open(data_path, mode='r', encoding=ENCODING)
    it = 0
    needed = 2
    for l in f_read_data:
        if it == needed:
            raw_json = json.loads(l)
            drawer = TreeDrawer(raw_json, predictions[it])
            drawer.draw_tree()

        it += 1


if __name__ == '__main__':
    # f = open(FILE_TRAINING_DATASET, mode='r', encoding=ENCODING)
    # it = 0
    # lim = 10
    # for l in f:
    #     if it == lim:
    #         break
    #     j = json.loads(l)
    #     print(len(j))
    #     it += 1
    #
    # print('\n\n')
    #
    # f = open(FILE_TRAINING_PROCESSED, mode='r', encoding=ENCODING)
    # it = 0
    # lim = 10
    # for l in f:
    #     if it == lim:
    #         break
    #     j = json.loads(l)
    #     print(len(j))
    #     it += 1
    main(FILE_TRAINING_DATASET, RESULTS_FILE)
    # t = Tree("(A:1,(B:1,(E:1,D:1):0.5):0.5);")

    # t = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")

    # right_c0_r0 = TextFace("right_col0_row0")
    # right_c0_r1 = TextFace("right_col0_row1")
    # right_c1_r0 = TextFace("right_col1_row0")
    # right_c1_r1 = TextFace("right_col1_row1")
    # right_c1_r2 = TextFace("right_col1_row2")
    #
    # top_c0_r0 = TextFace("top_col0_row0")
    # top_c0_r1 = TextFace("top_col0_row1")
    #
    # bottom_c0_r0 = TextFace("bottom_col0_row0")
    # bottom_c0_r1 = TextFace("bottom_col0_row1")
    #
    # aligned_c0_r0 = TextFace("aligned_col0_row0")
    # aligned_c0_r1 = TextFace("aligned_col0_row1")
    #
    # aligned_c1_r0 = TextFace("aligned_col1_row0")
    # aligned_c1_r1 = TextFace("aligned_col1_row1")
    #
    # all_faces = [right_c0_r0, right_c0_r1, right_c1_r0, right_c1_r1, right_c1_r2, top_c0_r0, \
    #              top_c0_r1, bottom_c0_r0, bottom_c0_r1, aligned_c0_r0, aligned_c0_r1, \
    #              aligned_c1_r0, aligned_c1_r1]
    #
    # # set a border in all faces
    # for f in all_faces:
    #     f.margin_bottom = 5
    #     f.margin_top = 5
    #     f.margin_right = 10
    #
    # t.add_face(right_c0_r0, column=0, position="branch-right")
    # t.add_face(right_c0_r1, column=0, position="branch-right")
    #
    # t.add_face(right_c1_r0, column=1, position="branch-right")
    # t.add_face(right_c1_r1, column=1, position="branch-right")
    # t.add_face(right_c1_r2, column=1, position="branch-right")
    #
    # t.add_face(top_c0_r0, column=0, position="branch-top")
    # t.add_face(top_c0_r1, column=0, position="branch-top")
    #
    # t.add_face(bottom_c0_r0, column=0, position="branch-bottom")
    # t.add_face(bottom_c0_r1, column=0, position="branch-bottom")
    #
    # a = t & "a"
    # a.set_style(NodeStyle())
    # a.img_style["bgcolor"] = "lightgreen"
    #
    # b = t & "b"
    # b.set_style(NodeStyle())
    # b.img_style["bgcolor"] = "indianred"
    #
    # c = t & "c"
    # c.set_style(NodeStyle())
    # c.img_style["bgcolor"] = "lightblue"
    #
    # t.set_style(NodeStyle())
    # t.img_style["bgcolor"] = "lavender"
    # t.img_style["size"] = 12
    #
    # for leaf in t.iter_leaves():
    #     leaf.img_style["size"] = 12
    #     leaf.add_face(right_c0_r0, 0, "branch-right")
    #     leaf.add_face(aligned_c0_r1, 0, "aligned")
    #     leaf.add_face(aligned_c0_r0, 0, "aligned")
    #     leaf.add_face(aligned_c1_r1, 1, "aligned")
    #     leaf.add_face(aligned_c1_r0, 1, "aligned")
    #
    # ts = TreeStyle()
    # ts.show_scale = False
    # t.render("face_positions.png", w=800, tree_style=ts)
