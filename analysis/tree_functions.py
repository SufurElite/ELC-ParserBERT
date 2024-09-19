"""
Tree functions helper implemented based on the algorithms from the StructFormer paper: https://arxiv.org/pdf/2012.00857
"""
from nltk.tree import Tree
from typing import List


def build_tree_whole_words(
    subwords: List,
    distances: List,
    heights: List,
    space_char: str = "Ġ",
    debug: bool = False,
):
    """
    params:
        subwords
        distances
        heights
        debug

    before running build tree, given that the model creates subwords, we
    recombine the subwords into their original words and average
    the heights/distances across the subwords
    """

    assert len(subwords) == len(distances) and len(distances) == len(heights)
    words = []
    new_heights = []
    new_distances = []
    last_idx = 0
    if subwords[0][0] == space_char:
        subwords[0] = subwords[0][1:]
    for idx in range(len(subwords)):
        new_idx = 0
        if subwords[idx][0] == space_char:
            new_idx = idx
        else:
            continue
        tmp_height = 0
        tmp_distance = 0
        for i in range(last_idx, new_idx):
            subwords[i] = subwords[i].replace(space_char, "")
            tmp_height += heights[i]
            tmp_distance += distances[i]
        tmp_height = tmp_height / (new_idx - last_idx)
        tmp_distance = tmp_height / (new_idx - last_idx)
        new_heights.append(tmp_height)
        new_distances.append(tmp_distance)
        words = words + ["".join(subwords[last_idx:new_idx])]
        last_idx = new_idx

    tmp_height = 0
    tmp_distance = 0
    for i in range(last_idx, len(subwords)):
        subwords[i] = subwords[i].replace(space_char, "")
        tmp_height += heights[i]
        tmp_distance += distances[i]
    tmp_height = tmp_height / (len(subwords) - last_idx)
    tmp_distance = tmp_distance / (len(subwords) - last_idx)
    new_heights.append(tmp_height)
    new_distances.append(tmp_distance)

    words = words + ["".join(subwords[last_idx:])]

    assert len(new_distances) == len(words) and len(words) == len(new_heights)

    return build_tree(words, new_distances, new_heights, debug)


def build_tree(words: List, distances: List, heights: List, debug: bool = False):
    """
    params:
        words
        distances
        heights
        debug
    """
    t = None
    dependency_graph = []
    parent = None
    height = None
    if debug:
        print("distances: ", distances)
        print("heights: ", heights)
        print("words: ", words)
        print("\n\n")

    if len(distances) == 0 and len(words) == 1 and len(heights) == 1:
        t = Tree(words[0], [])
        dependency_graph = []
        parent = words[0]
        height = heights[0]
    else:
        # get the index of the next largest distance
        max_value = max(distances)
        max_index = distances.index(max_value)
        Tree_left, dependency_left, parent_left, height_left = build_tree(
            words[: max_index + 1], distances[:max_index], heights[: max_index + 1]
        )
        if len(words[max_index + 1 :]) == 0:
            return Tree_left, dependency_left, parent_left, height_left
        Tree_right, dependency_right, parent_right, height_right = build_tree(
            words[max_index + 1 :], distances[max_index + 1 :], heights[max_index + 1 :]
        )
        dependency_graph = dependency_left + dependency_right
        t = Tree("*", [Tree_left, Tree_right])
        if height_left > height_right:
            height = height_left
            parent = parent_left
            dependency_graph.append([parent_left, parent_right])
        else:
            parent = parent_right
            height = height_right
            dependency_graph.append([parent_right, parent_left])

    return t, dependency_graph, parent, height


def model_to_tree_output(input: str, tokenizer, model):
    space_char = tokenizer.convert_ids_to_tokens(tokenizer(" ").input_ids)[0]
    subwords = tokenizer.convert_ids_to_tokens(tokenizer(input).input_ids)
    tree_values, _ = model(tokenizer.encode(input, return_tensors="pt"))
    heights = tree_values["height"][0]
    distances = tree_values["distance"][0]
    tree, _, _, _ = build_tree_whole_words(
        subwords, distances, heights, space_char=space_char
    )
    tree.pretty_print()
    return tree


if __name__ == "__main__":
    subwords = [
        "ĠThe",
        "ĠBaby",
        "L",
        "M",
        "ĠCha",
        "llen",
        "ge",
        "Ġwill",
        "Ġbe",
        "Ġheld",
        "Ġagain",
        "Ġas",
        "Ġthe",
        "Ġshared",
        "Ġtask",
        "Ġfor",
        "ĠCo",
        "N",
        "LL",
        "Ġ",
        "2",
        "0",
        "2",
        "4",
    ]

    distances = [
        -2.5962e-01,
        -6.7681e-01,
        -7.1163e-02,
        -9.5030e-01,
        -4.0385e-01,
        -9.4605e-01,
        -1.1227e00,
        -6.9064e-01,
        -1.1060e00,
        -1.0958e00,
        -8.7155e-01,
        -6.3248e-01,
        -9.0634e-01,
        -8.5295e-01,
        -6.4920e-01,
        -8.4231e-01,
        -8.9462e-01,
        -6.8456e-01,
        -3.9250e-01,
        -7.9879e-01,
        -7.1378e-01,
        -9.0493e-02,
        -4.1572e-01,
        9.9993e03,
    ]

    heights = [
        0.4022,
        0.1320,
        -0.5007,
        0.2278,
        -0.0570,
        0.1832,
        -0.8603,
        0.0759,
        0.3247,
        0.2826,
        0.2969,
        0.1764,
        -0.5244,
        -0.4788,
        -0.0916,
        0.6123,
        -0.3484,
        -0.1902,
        -0.1880,
        -0.6134,
        0.2112,
        -0.0740,
        -0.0905,
        -0.1619,
    ]

    assert len(subwords) == len(distances) and len(heights) == len(distances)

    t, dependency_graph, parent, height = build_tree_whole_words(
        subwords, distances, heights
    )
    t.pretty_print()
