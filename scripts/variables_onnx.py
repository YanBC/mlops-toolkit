'''
Example run:
python scripts/onnx_has_variable_names.py sigmoid.onnx factor,x
'''

import argparse
import onnx
from typing import List


def get_variable_names(graph: onnx.GraphProto) -> List[str]:
    valueinfo_names = [vi.name for vi in graph.input] + [vi.name for vi in graph.output]
    tensor_names = [t.name for t in graph.initializer]
    node_io_names = []
    for node in graph.node:
        if node.op_type == "Loop":
            body_graph = node.attribute[0].g
            body_valueinfo_names, body_tensor_names, body_node_io_names = \
                    get_variable_names(body_graph)
            valueinfo_names.extend(body_valueinfo_names)
            tensor_names.extend(body_tensor_names)
            node_io_names.extend(body_node_io_names)
        node_io_names += [i for i in node.input]
        node_io_names += [o for o in node.output]
    return valueinfo_names, tensor_names, node_io_names


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('onnx_file', help="Path to onnx file")
    p.add_argument('--names', default="", help="Names of tensors and valueinfos, comma seperated. If omitted, prints all names.")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    onnx_path = args.onnx_file
    name_str = args.names
    names: List[str] = name_str.split(',')
    names: List[str] = [name for name in sorted(names) if name != '']

    model = onnx.load(onnx_path)
    graph = model.graph

    valueinfo_names, tensor_names, node_io_names = get_variable_names(graph)
    # there might be some intersection between
    # valueinfo_names, tensor_names and node_io_names
    all_names = set(valueinfo_names + tensor_names + node_io_names)

    if len(names) != 0:
        ret_str = ""
        for name in names:
            if name in all_names:
                ret_str += f"{name}: True\n"
            else:
                ret_str += f"{name}: False\n"
    else:
        ret_str = "\n".join(sorted(list(all_names)))

    print(ret_str)
