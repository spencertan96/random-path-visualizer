import os
import math
import random
import sys
import numpy as np
import cv2 as cv

# Adjustable options
EACH_MEMBER_ALL_NEIGHBOURS = True
GREEDY_PICKING = True
DEBUG = False
FILEPATH = './sample_input.txt'
SAVEFILEPATH = './sample_previous_path.txt'
IMGFILEPATH = 'sample_img.png'
DISPLAY_GRAPH = False
GRAPHFILEPATH = 'sample_graph.png'
MEMBER_NUM = 11

# Constants
ARROWTIP_RATIO = 15
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

# Input: 2-tuple (etc. (0, 10), (15, 20))
def get_distance_between(coords1, coords2):
    x1 = coords1[0]
    y1 = coords1[1]
    x2 = coords2[0]
    y2 = coords2[1]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def sort_by_dist(tuple):
    return tuple[1]

# One-way edges, only stores other node that current node connects to.
class Edge:
    def __init__(self, num, euclid_dist) -> None:
        self.node_num = num
        self.euclid_dist = euclid_dist
        pass

class Node:
    # Class-wide dictionary to store references to Nodes.
    node_dict = dict()

    def __init__(self, num, coordinates) -> None:
        self.num = num
        self.coords = coordinates
        # list of node numbers (not the actual node object itself) this node is connected to
        self.connected_nodes = []
        self.selected = False
        pass

    def __str__(self) -> str:
        # state = "Selected" if self.selected else "Not selected"
        # return self.name + " (" + state + ")"
        return f"N{self.num}({self.selected})"

    def __repr__(self) -> str:
        # state = "Selected" if self.selected else "Not selected"
        # return self.name + " (" + state + ")"
        return f"N{self.num}({self.selected})"

    def print_edges(self):
        nodes_to = f"N{self.num} is connected to: "
        for node in self.connected_nodes:
            nodes_to += str(node) + " "
        print(nodes_to)

    # Input: Integer number of node this is connected to.
    def add_edge(self, node_num):
        self.connected_nodes += [node_num]
        pass

    # Method to get next node to connect line to (for drawing).
    # Returns node number of selected node
    def get_next(self, goal_node):
        if GREEDY_PICKING:
            node_nums = self.connected_nodes
            goal_coords = goal_node.coords
            # Go through list of edges, calculate distance to goal for each edge
            dist_list = []
            for node_num in node_nums:
                # Make sure cannot point to other members if they are not the goal
                if node_num <= MEMBER_NUM and node_num != goal_node.num:
                    continue
                dist = get_distance_between(goal_coords, Node.node_dict[node_num].coords)
                dist_list += [(node_num, dist)]
            # Sort generated distances and pick shortest
            dist_list.sort(key=sort_by_dist)
            for tpl in dist_list:
                if Node.node_dict[tpl[0]].selected:
                    continue
                # Not selected yet
                Node.node_dict[tpl[0]].selected = True
                return tpl[0]
            # All nodes already selected, use closest
            return dist_list[0][0]
        else:
            # TODO: A* pathfinding, use Euclidean distance as heuristic
            return -1

# Each member has a name.
class Member:
    def __init__(self, name, node, neighbors = ()) -> None:
        self.name = name
        self.node = node
        self.coords = node.coords
        self.neighbors = neighbors
        self.selected = False
        pass

    def __str__(self) -> str:
        return self.name
        
    def __repr__(self) -> str:
        return self.name
    
    def add_neighbor(self, neighbor):
        self.neighbors = self.neighbors + (neighbor,)
        pass

    def print_neighbors(self):
        neighbour_str = ""
        for member in self.neighbors:
            neighbour_str += member.name + " "
        print(self.name, "'s neighbours are:", neighbour_str)
        pass

    # Method to get next neighbor in path.
    def get_next(self, prev_assignments):
        # select a random non-selected neighbor
        while len(self.neighbors) > 0:
            num_neighbors = len(self.neighbors)
            index = random.randint(0, num_neighbors - 1)
            ## Check prev assignments and re-select if repeated
            if prev_assignments:
                # get previous assignment
                prev = prev_assignments[self.name]
                if prev == self.neighbors[index].name:
                    print("Repeated neighbor", prev, "detected, re-selecting!")
                    # update neighbor list
                    intermediateLst = list(self.neighbors)
                    intermediateLst.pop(index)
                    self.neighbors = intermediateLst
                    continue

            if self.neighbors[index].selected:
                # ran out of choices
                if len(self.neighbors) == 1:
                    print("No neighbors available!")
                    return None
                # update neighbor list
                intermediateLst = list(self.neighbors)
                intermediateLst.pop(index)
                self.neighbors = intermediateLst
                continue
            else:
                break
        member = self.neighbors[index]
        member.selected = True
        # remove selected neighbor
        if DEBUG:
            print("Picked", member, "out of", num_neighbors, "choices")
        # print(self.neighbors)
        intermediateLst = list(self.neighbors)
        intermediateLst.pop(index)
        self.neighbors = intermediateLst
        return member

def initialize_members():
    ## Initialize graph nodes and edges
    # Load from file
    # Format: {MemberName}:{XCoordinate},{YCoordinate}
    #         {NodeNumber}:{XCoordinate},{YCoordinate}
    #         till empty line, then
    #         {NodeNumber}:{ConnectedNode},{ConnectedNode2}
    members_initialized = 0
    members_names = []
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as f:
            text = f.readline()
            if DEBUG:
                print(text)
            while text and text != "\n":
                # Ignore comments
                if text[0] == "#":
                    text = f.readline()
                    continue
                text = text.split(":")
                if members_initialized < MEMBER_NUM:
                    members_initialized += 1
                    node_num = members_initialized
                    members_names.append(text[0])
                else: 
                    node_num = int(text[0])
                coords = text[1].split(",")
                Node.node_dict[node_num] = Node(node_num, (int(coords[0]), int(coords[1])))
                text = f.readline()
            # Add edges
            text = f.readline()
            if DEBUG:
                print(text)
            while text and text != "\n":
                # Ignore comments
                if text[0] == "#":
                    text = f.readline()
                    continue
                text = text.split(":")
                node_num = int(text[0])
                nodes = text[1].split(",")
                for node in nodes:
                    Node.node_dict[node_num].add_edge(int(node))
                text = f.readline()
            # TODO: Validate edges (through a function?)
        if DEBUG:
            Node.node_dict[1].print_edges()
    else:
        print("Graph data file {FILEPATH}' not found!")
        # Initialize nodes of members
        # Default if no file loaded
        Node.node_dict[1] = Node(1, (602, 228))
        Node.node_dict[2] = Node(2, (534, 426))
        Node.node_dict[3] = Node(3, (174, 626))
        Node.node_dict[4] = Node(4, (423, 190))
        Node.node_dict[5] = Node(5, (269, 365))
        Node.node_dict[6] = Node(6, (521, 23))
        Node.node_dict[7] = Node(7, (374, 566))
        Node.node_dict[8] = Node(8, (600, 622))
        Node.node_dict[9] = Node(9, (165, 130))
        Node.node_dict[10] = Node(10, (60, 346))
        Node.node_dict[11] = Node(11, (300, 49))
        # Initialize connecting nodes
        Node.node_dict[12] = Node(12, (602, 228))
        # Initialize edges
        Node.node_dict[1].add_edge(12)
        print(Node.node_dict)

    # Names and corresponding nodes
    members = []
    for i in range(MEMBER_NUM):
        members.append(Member(members_names[i], Node.node_dict[i+1]))
    print("Members:", members)
    return members

# Returns a dictionary of member: previously assigned member, or an empty dictionary if no previous assignment saved.
def retrieve_prev_assignments():
    if not os.path.exists(SAVEFILEPATH):
        return dict()
    
    assignments = dict()
    with open(SAVEFILEPATH, 'r') as f:
        text = f.readline()
        text = text.rstrip()
        if DEBUG:
            print(text)
        while text and text[0] != "#" and text != "\n":
            pair = text.split("->")
            assignments[pair[0]] = pair[1]
            text = f.readline()
            text = text.rstrip()
            if DEBUG:
                print(text)
    return assignments

def initialize_member_neighbours(members):
    if not EACH_MEMBER_ALL_NEIGHBOURS:
        # Each member can only connect to another member close by
        # Hard-coded to work only for the sample graph
        members[1].add_neighbor(members[0])
        members[3].add_neighbor(members[0])
        members[5].add_neighbor(members[0])
        members[7].add_neighbor(members[0])
        members[0].add_neighbor(members[1])
        members[2].add_neighbor(members[1])
        members[3].add_neighbor(members[1])
        members[5].add_neighbor(members[1])
        members[6].add_neighbor(members[1])
        members[7].add_neighbor(members[1])
        members[4].add_neighbor(members[2])
        members[6].add_neighbor(members[2])
        members[8].add_neighbor(members[2])
        members[9].add_neighbor(members[2])
        members[0].add_neighbor(members[3])
        members[1].add_neighbor(members[3])
        members[4].add_neighbor(members[3])
        members[5].add_neighbor(members[3])
        members[6].add_neighbor(members[3])
        members[8].add_neighbor(members[3])
        members[9].add_neighbor(members[3])
        members[10].add_neighbor(members[3])
        members[1].add_neighbor(members[4])
        members[2].add_neighbor(members[4])
        members[3].add_neighbor(members[4])
        members[6].add_neighbor(members[4])
        members[7].add_neighbor(members[4])
        members[8].add_neighbor(members[4])
        members[9].add_neighbor(members[4])
        members[10].add_neighbor(members[4])
        members[0].add_neighbor(members[5])
        members[3].add_neighbor(members[5])
        members[7].add_neighbor(members[5])
        members[10].add_neighbor(members[5])
        members[0].add_neighbor(members[6])
        members[1].add_neighbor(members[6])
        members[2].add_neighbor(members[6])
        members[4].add_neighbor(members[6])
        members[7].add_neighbor(members[6])
        members[9].add_neighbor(members[6])
        members[0].add_neighbor(members[7])
        members[1].add_neighbor(members[7])
        members[2].add_neighbor(members[7])
        members[5].add_neighbor(members[7])
        members[6].add_neighbor(members[7])
        members[4].add_neighbor(members[8])
        members[9].add_neighbor(members[8])
        members[10].add_neighbor(members[8])
        members[2].add_neighbor(members[9])
        members[4].add_neighbor(members[9])
        members[8].add_neighbor(members[9])
        members[10].add_neighbor(members[9])
        members[3].add_neighbor(members[10])
        members[5].add_neighbor(members[10])
        members[8].add_neighbor(members[10])
        if DEBUG:
            members[0].print_neighbors()
            members[1].print_neighbors()
            members[2].print_neighbors()
            members[3].print_neighbors()
            members[4].print_neighbors()
            members[5].print_neighbors()
            members[6].print_neighbors()
            members[7].print_neighbors()
            members[8].print_neighbors()
            members[9].print_neighbors()
            members[10].print_neighbors()
    # Any neighbour
    else:
        for member in members:
            for neighbor in members:
                if member.name == neighbor.name:
                    continue
                member.add_neighbor(neighbor)
            # member.print_neighbors()

def pick_member(members, index):
    member = members[index]
    member.selected = True
    members.pop(index)
    return (member, members)

def draw_arrows(path, img):
    firstMember = True
    # Color variation
    color_r = 0
    color_g = 255
    color_b = 0
    red = False
    green_and_blue = False
    green = False

    # Add points to draw along
    for member in path:
        if DEBUG: 
            print(f"Drawing arrow to {member}...")
        if firstMember:
            prev = member
            pathStart = member.coords
            firstMember = False
            continue
        # Get waypoints from "prev" to "member"
        prev_node = prev.node
        if DEBUG: 
            print(f"First node: {prev_node.num}")
        node_num = prev_node.num
        nodes = [node_num]
        while node_num != member.node.num:
            node_num = prev_node.get_next(member.node)
            if DEBUG: 
                print(f"Next node: {node_num}")
            nodes += [node_num]
            prev_node = Node.node_dict[node_num]
        # Draw waypoints
        prev_node = nodes[0]
        for node in nodes:
            if node == nodes[0]:
                continue
            elif node != nodes[-1]:
                img = cv.line(img, Node.node_dict[prev_node].coords, Node.node_dict[node].coords,
                             (color_r, color_g, color_b), 3, -1, 0)
            else:
                # Get distance between coords
                dist = get_distance_between(Node.node_dict[prev_node].coords, Node.node_dict[node].coords)
                tip_size = ARROWTIP_RATIO/dist
                img = cv.arrowedLine(img, Node.node_dict[prev_node].coords, Node.node_dict[node].coords,
                                    (color_r, color_g, color_b), 3, -1, 0, tip_size)
            prev_node = node

        # Vary color from green to R+B to red to G+B then back to green
        if red:
            color_g = 0
            color_r = 255
            color_b -= 40
            if color_b == 0:
                # (255, 0, 0)
                green_and_blue = True
                red = False
        elif green_and_blue:
            color_r -= 40
            color_g += 40
            color_b += 40
            if color_r == 15:
                # (15, 240, 240)
                green = True
                green_and_blue = False
        elif green:
            color_r = 0
            color_g = 255
            color_b -= 40
            if color_b == 15:
                # (0, 255, 0)
                green = False
        else:
            color_r += 40
            color_g -= 40
            color_b += 40
            if color_g == 15:
                # (240, 15, 240)
                red = True
        prev = member
    chainEnd = prev.coords

    # Draw text "Start" and "End"
    displacementX = 20
    displacementY = 15
    startPos = (pathStart[0] + displacementX, pathStart[1] + displacementY)
    endPos = (chainEnd[0] + displacementX, chainEnd[1] + displacementY)
    cv.putText(img, "START", startPos, cv.FONT_HERSHEY_SIMPLEX, 1, RED, 3, -1)
    cv.putText(img, "END", endPos, cv.FONT_HERSHEY_SIMPLEX, 1, RED, 3, -1)

    return img

def save_assignments(path):
    with open(SAVEFILEPATH, 'w') as f:
        start = path[0]
        prev = path[0]
        for name in path:
            if name == prev:
                continue
            f.write(str(prev) + "->" + str(name) + "\n")
            prev = name
        f.write(str(prev) + "->" + str(start) + "\n")

def display_graph(img):
    graph_img = img.copy()
    # go through Node.node_dict, draw edges from key to each value
    for node in Node.node_dict.values():
        for node_to in node.connected_nodes:
            graph_img = cv.line(graph_img, Node.node_dict[node.num].coords, Node.node_dict[node_to].coords,
                    (0, 255, 0), 3, -1, 0)
    
    for node in Node.node_dict.values():
        # annotate each node with its num
        pos = (node.coords[0] + 3, node.coords[1] + 1)
        cv.putText(graph_img, str(node.num), pos, cv.FONT_HERSHEY_PLAIN, 0.9, BLUE, 1, -1)
        
    for node in Node.node_dict.values():
        # draw node as dot
        graph_img = cv.circle(graph_img, node.coords, radius=2, color=RED, thickness=-1)

    cv.imshow("Edges drawn", graph_img)
    return graph_img

def generate_path():
    global DISPLAY_GRAPH
    path = []
    members = initialize_members()
    prev_assignments = retrieve_prev_assignments()
    initialize_member_neighbours(members)
    # Pick random starting point
    # Randomly pick neighbour, continue with that neighbour
    random_start = random.randint(0, len(members) - 1)
    (member, members) = pick_member(members, random_start)
    print(f"Starting with: {member}")
    path.append(member)
    # Pick rest of the members
    while len(path) < MEMBER_NUM:
        next = member.get_next(prev_assignments)
        if next == None:
            # No solution, try for another solution
            # Reset everything
            path = []
            members = initialize_members()
            initialize_member_neighbours(members)
            # Pick random starting point
            # Randomly pick neighbour, continue with that neighbour
            random_start = random.randint(0, len(members) - 1)
            (member, members) = pick_member(members, random_start)
            print(f"Restarting with: {member}")
            path.append(member)
            continue
        path.append(next)
        members.remove(next)
        member = next
    print("Final path: ", path)

    ## Draw arrows on picture

    # Load image
    img = cv.imread(IMGFILEPATH)
    if img is None:
        sys.exit("Could not read the image.")
    
    # Display graph with annotated nodes and edges, can be used to save an image of the graph
    if DISPLAY_GRAPH:
        graph_img = display_graph(img)
    # Controls for image prompt, "x" to exit, "s" to save image, "c" to continue to display path
    while DISPLAY_GRAPH:
        k = cv.waitKey(0)
        if k == ord("x"):
            # exit
            return
        if k == ord("s"):
            cv.imwrite(GRAPHFILEPATH, graph_img)
            DISPLAY_GRAPH = False
        elif k == ord("c"):
            # continue to display path
            cv.destroyAllWindows()
            DISPLAY_GRAPH = False
            break

    # Draw arrows
    img = draw_arrows(path, img)

    # Show final image
    instructionPos = (10, 25)
    secondInstructionPos = (10, 50)
    thirdInstructionPos = (10, 75)
    cv.putText(img, "'R' to re-generate", instructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.putText(img, "'X' to close", secondInstructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.putText(img, "'S' to save current path", thirdInstructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.imshow("Generated Path", img)
    
    is_current_config_saved = False
    while True:
        if is_current_config_saved:
            saveNotifPos = (10, 530)
            cv.putText(img, "Path saved!", saveNotifPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
            cv.imshow("Generated Path", img)
            
        k = cv.waitKey(0)
        # If "x" key pressed, close image
        if k == ord("x"):
            break
        elif k == ord("s"):
            # Save assignments
            save_assignments(path)
            print("Current path saved!")
            is_current_config_saved = True
        elif k == ord("r"):
            is_current_config_saved = False
            generate_path()
            break

generate_path()
