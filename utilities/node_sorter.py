FILEPATH = "nodes.txt"
SAVEFILEPATH = "sorted_nodes.txt"
FIRST_NODE_NUM = 18

nodes = []

with open(FILEPATH, 'r') as f:
    text = f.readline()
    while text:
        text = text.split(":")
        coords = text[1].split(",")
        nodes.append((int(coords[0]), int(coords[1])))
        text = f.readline()
print(nodes)

# sort nodes by x then y
nodes.sort(key=lambda node: node[1])
nodes.sort(key=lambda node: node[0])
print("After sorting")
print(nodes)

node_num = FIRST_NODE_NUM
with open(SAVEFILEPATH, 'w') as f:
    for node in nodes:
        f.write(f"{node_num}:{node[0]},{node[1]}\n")
        node_num += 1
print("Output done!")
