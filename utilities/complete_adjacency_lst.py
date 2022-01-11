from functools import reduce

FILEPATH = "adj_list.txt"
SAVEFILEPATH = "adj_list_full.txt"

adj_dict = dict()

with open(FILEPATH, 'r') as f:
    text = f.readline()
    while text:
        text = text.split(":")
        key = int(text[0])
        nodes = text[1].split(",")
        nodes = [int(x) for x in nodes]
        if key not in adj_dict:
            adj_dict[key] = nodes
        else:
            adj_dict[key].extend(nodes)
        text = f.readline()
print("Initial adj dict[30]")
print(adj_dict[30])

# check each adjacency list, check if node num higher,
#  add itself to their adj list if not alr there
new_adj_dict = adj_dict.copy()

for num, nodes in adj_dict.items():
    for node in nodes:
        if node > num:
            # add to their adj list
            if num not in new_adj_dict[node]:
                new_adj_dict[node].append(num)
print("Filled adj_dict[30]:")
print(new_adj_dict[30])

#  sort all adj list at end
for nodes in new_adj_dict.values():
    nodes.sort()
print("Sorted adj_dict[30]:")
print(new_adj_dict[30])

with open(SAVEFILEPATH, 'w') as f:
    for k,v in new_adj_dict.items():
        node_str = reduce(lambda x,y: f"{x},{y}", v)
        f.write(f"{k}:{node_str}\n")
print("Output done!")