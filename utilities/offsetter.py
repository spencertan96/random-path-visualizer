FILEPATH = "nodes.txt"
offset = 1

output = ""
with open(FILEPATH, 'r') as f:
    text = f.readline()
    while text:
        text = text.split(":")
        output += f"{int(text[0]) + offset}:{text[1]}"
        text = f.readline()
print(output)