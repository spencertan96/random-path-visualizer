from functools import reduce

## Does not fully substitute nodes (due to missing nodes from 13-17)
## Need manual renaming for 1-12 in adj_list and all coords

FILEPATH = "nodes.txt"
SAVEFILEPATH = "renamed_nodes.txt"
FIRST_NODE_NUM = 1

node_dict = dict()
relations = dict()
node_num = FIRST_NODE_NUM

def is_node_not_deleted(num):
    deleted_nums = []
    if num in deleted_nums:
        return False
    return True

with open(FILEPATH, 'r') as f:
    text = f.readline()
    while text:
        text = text.split(":")
        relations[int(text[0])] = node_num
        # for skipping nodes 13-18
        if node_num == 12:
            node_num += 7
        else:
            node_num += 1
        nodes = text[1].split(",")
        nodes = [int(x) for x in nodes]
        nodes = list(filter(is_node_not_deleted, nodes))
        node_dict[int(text[0])] = nodes
        text = f.readline()
print("Mapping dictionary:")
print(relations)

with open(SAVEFILEPATH, 'w') as f:
    for old_num,nodes in node_dict.items():
        node_str = reduce(lambda x,y: f"{x if x not in relations else relations[x]},{y if y not in relations else relations[y]}", nodes)
        f.write(f"{relations[old_num]}:{node_str}\n")
print("Output done!")

