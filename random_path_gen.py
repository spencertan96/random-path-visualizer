import os
import math
import random
import sys
import heapq
import numpy as np
import cv2 as cv
from enum import Enum, unique

@unique
class Pathfinding(Enum):
    GREEDY_PICKING = 0
    DIJKSTRA = 1
    DIJKSTRA_WITH_OVERLAP = 2

# Adjustable options
NO_IMAGE = False
FILEPATH = 'sample_input.txt'
SAVEFILEPATH = 'sample_previous_path.txt'
IMGFILEPATH = 'sample_img.png'
# Number of points in a bezier curve
# (The higher it is, the smoother curves will be at the expense of computation time)
NUM_POINTS = 10
# Only matters if NO_IMAGE is False
MEMBER_NUM = 11
# Initial pathfinding method
METHOD = Pathfinding.GREEDY_PICKING
# Not very useful option
EACH_MEMBER_ALL_NEIGHBOURS = True
# Developer options
DISPLAY_GRAPH = False
GRAPHFILEPATH = 'sample_graph.png'
DEBUG = False

# Constants
BEZIERIFY = False
NUM_METHODS = len(Pathfinding)
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

# Class meant to be used in relaxing of neighbours in Dijkstra's algorithm through a Priority Queue.
class DijkstraNode:
    # Class-wide dictionary to store references to DijkstraNodes, int -> DijkstraNode.
    node_dict = dict()

    def __init__(self, num, dist, prev) -> None:
        self.num = num
        self.dist = dist
        self.selected = False
        # Relevant values to be set during relaxing of neighbours
        self.prev = prev
        pass

    def __repr__(self) -> str:
        return f'N{self.num}: {self.dist}{self.selected}'

    def __lt__(self, other):
        if self.dist == other.dist:
            return self.num < other.num
        return self.dist < other.dist

class Node:
    # Class-wide dictionary to store references to Nodes, int -> Node.
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

    # Method to get next node to connect line to (for Greedy Picking).
    # Returns node number of selected node
    def get_next_greedy(self, goal_node):
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
    
    # TODO: A* pathfinding, use Euclidean distance as heuristic
    
    # Input: Node object that represents the goal node
    def get_greedy_path(self, goal_node):
        nodes = [self.num]
        node_num = self.num
        prev_node = self
        while node_num != goal_node.num:
            node_num = prev_node.get_next_greedy(goal_node)
            if DEBUG: 
                print(f"Next node: {node_num}")
            nodes += [node_num]
            prev_node = Node.node_dict[node_num]
        return nodes

    # Input: Node object that represents the goal node
    def get_dijkstra_path(self, goal_node):
        # Dijkstra's algorithm
        # Priority queue to be filled with DijkstraNodes with custom comparator (node_num, dist_to)
        # There will be duplicates of same node in PQ, only minimum dist will be relevant, rest will be ignored
        pq = []
        heapq.heapify(pq)
        prev_dict = dict()
        # Dict to track if node has been visited
        visited_dict = dict()
        for num in Node.node_dict.keys():
            if num == self.num:
                dn = DijkstraNode(num, 0, num)
                heapq.heappush(pq, dn)
                continue
            dn = DijkstraNode(num, sys.maxsize, -1)
            heapq.heappush(pq, dn)
        while pq:
            smallest_node = heapq.heappop(pq)
            # Ignore visited nodes/selected nodes
            if smallest_node.num in visited_dict or DijkstraNode.node_dict[smallest_node.num].selected:
                continue
            # Never select member nodes unless it is the source node or goal node
            if not smallest_node.num == self.num:
                if not smallest_node.num == goal_node.num:
                    if smallest_node.num <= MEMBER_NUM:
                        continue
            visited_dict[smallest_node.num] = True
            # Update previous node
            prev_dict[smallest_node.num] = smallest_node.prev
            if goal_node.num == smallest_node.num:
                # Goal reached, terminate
                break
            relax_neighbours(smallest_node, pq)
        # Check if goal reached
        if goal_node.num not in prev_dict or prev_dict[goal_node.num] == -1:
            print("Error! Goal cannot be reached! Restarting generator...")
            # Failsafe, restart
            generate_path()
            quit()
        # Get sequence from prev_dict
        seq = []
        num = goal_node.num
        if DEBUG:
            print(prev_dict)
        while num != self.num:
            seq.insert(0, num)
            if num != goal_node.num:
                # Don't mark goal node as selected or it will get skipped when relaxing
                # Only mark if overlap disallowed
                if METHOD != Pathfinding.DIJKSTRA_WITH_OVERLAP:
                    DijkstraNode.node_dict[num].selected = True
            num = prev_dict[num]
        seq.insert(0, self.num)
        return seq

    # Input: A path (list of node_nums)
    # Use points in path as control points to generate a bezier curve.
    # Get points from curve at segments of t.
    # Output: List of coordinates as tuples
    def get_bezier_path(self, path):
        seq = []
        t_steps = 1 / NUM_POINTS
        t = 0
        while t <= 1:
            arr = []
            arr2 = []
            # Initialize array with first set of LERPs, path needs to and is guaranteed to be > 2 nodes
            # path first contains node_nums
            for i in range(len(path)):
                if i == 0:
                    continue
                # add LERPS to arr
                val = lerp(Node.node_dict[path[i-1]].coords, Node.node_dict[path[i]].coords, t)
                arr.append(val)
            # Arr should now be filled with coordinates
            while len(arr) > 1:
                for i in range(len(arr)):
                    if i == 0:
                        continue
                    val = lerp(arr[i-1], arr[i], t)
                    arr2.append(val)
                arr = arr2.copy()
                arr2 = []
            rounded_coords = (round(arr[0][0]), round(arr[0][1]))
            seq.append(rounded_coords)
            t += t_steps
        return seq

# Takes in a DijkstraNode and PQ, calculates new distance and pushes into PQ
# Ignore member nodes!
def relax_neighbours(dnode, pq):
        node = Node.node_dict[dnode.num]
        for n in node.connected_nodes:
            new_dist = get_distance_between(Node.node_dict[n].coords, node.coords) + dnode.dist
            heapq.heappush(pq, DijkstraNode(n, new_dist, dnode.num))
            continue
        return

# Input: 2 coordinates and a t-value with which to LERP with.
# Output: A 2-tuple of coordinates
# Rounds to 2 d.p to preserve meaningful decimals, only rounded to int at end of function
def lerp(coords1, coords2, t):
    x = round(coords1[0] * (1 - t) + coords2[0] * t, 2)
    y = round(coords1[1] * (1 - t) + coords2[1] * t, 2)
    return (x, y)

# Each member has a name.
class Member:
    # Each member is tied to a node
    # Unless there is no image as no node required
    def __init__(self, name, node, neighbors = ()) -> None:
        self.name = name
        self.node = node if node else None
        self.coords = node.coords if node else None
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
                    if DEBUG:
                        print("Repeated neighbor", prev, "detected, re-selecting!")
                    # update neighbor list
                    if len(self.neighbors) == 1:
                        # only 1 neighbor left but is repeated
                        if DEBUG:
                            print("No eligible neighbor! Restarting path!")
                        return None
                    intermediateLst = list(self.neighbors)
                    intermediateLst.pop(index)
                    self.neighbors = intermediateLst
                    continue

            if self.neighbors[index].selected:
                # ran out of choices
                if len(self.neighbors) == 1:
                    if DEBUG:
                        print("No neighbors available! Restarting path!")
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

# Returns list of members that the path will be made out of
def initialize_members():
    members_names = read_simple_input_file(FILEPATH) if NO_IMAGE else read_input_file(FILEPATH) 

    # Names and corresponding nodes
    members = []
    for i in range(MEMBER_NUM):
        if NO_IMAGE:
            members.append(Member(members_names[i], None))
        else:
            members.append(Member(members_names[i], Node.node_dict[i+1]))
    print("Members:", members)
    return members

# Read input file, return list of members
# Automatically counts the number of members in the input file and sets MEMBER_NUM
def read_simple_input_file(filepath):
    global MEMBER_NUM
    # Only read one line, ignore extra lines
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            text = f.readline()
            if DEBUG:
                print(text)
            while text and text[0] == "#":
                # Ignore comments
                text = f.readline()
                continue
            names = text.split(",")
            MEMBER_NUM = len(names)
            return names
    else:
        print("Input file {FILEPATH}' not found!")
        quit()

# Read input file, populate Node.node_dict, return list of members
def read_input_file(filepath):
    # Load from file
    # Format: {MemberName}:{XCoordinate},{YCoordinate}
    #         {NodeNumber}:{XCoordinate},{YCoordinate}
    #         till empty line, then
    #         {NodeNumber}:{ConnectedNode},{ConnectedNode2}
    members_initialized = 0
    names = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
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
                    names.append(text[0])
                else: 
                    node_num = int(text[0])
                coords = text[1].split(",")
                Node.node_dict[node_num] = Node(node_num, (int(coords[0]), int(coords[1])))
                # Attributes don't really matter, only need to track whether used or not through .selected
                DijkstraNode.node_dict[node_num] = DijkstraNode(node_num, -1, -1)
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
            # TODO: Validate edges
        if DEBUG:
            Node.node_dict[1].print_edges()
    else:
        print("Input file {FILEPATH}' not found!")
        quit()
    return names

# Returns a dictionary of member: previously assigned member, or an empty dictionary if no previous assignment saved.
def retrieve_prev_assignments():
    if not os.path.exists(SAVEFILEPATH):
        return dict()
    
    assignments = dict()
    all_assignments = ""
    with open(SAVEFILEPATH, 'r') as f:
        text = f.readline()
        text = text.rstrip()
        all_assignments += text
        while text and text[0] != "#" and text != "\n":
            pair = text.split("->")
            assignments[pair[0]] = pair[1]
            text = f.readline()
            text = text.rstrip()
            all_assignments += ", " + text
    if DEBUG: 
        print(all_assignments[:-2])
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

def draw_points(nodeslst, img):
    # Color variation
    color_r = 0
    color_g = 255
    color_b = 0
    red = False
    green_and_blue = False
    green = False
    pathStart = None

    for lst in nodeslst:
        prev_node = lst[0]
        for node in lst:
            if node == lst[0]:
                if pathStart == None:
                    pathStart = Node.node_dict[node].coords
                continue
            elif node != lst[-1]:
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
        
    pathEnd = Node.node_dict[prev_node].coords

    # Draw text "Start" and "End"
    displacementX = 20
    displacementY = 15
    startPos = (pathStart[0] + displacementX, pathStart[1] + displacementY)
    endPos = (pathEnd[0] + displacementX, pathEnd[1] + displacementY)
    cv.putText(img, "START", startPos, cv.FONT_HERSHEY_SIMPLEX, 1, RED, 3, -1)
    cv.putText(img, "END", endPos, cv.FONT_HERSHEY_SIMPLEX, 1, RED, 3, -1)

    # Display instructions
    instructionPos = (10, 25)
    secondInstructionPos = (10, 50)
    thirdInstructionPos = (10, 75)
    fourthInstructionPos = (10, 100)
    cv.putText(img, "'Z'/'X' to switch algorithms", instructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.putText(img, "'S' to save current path", secondInstructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.putText(img, "'R' to re-generate", thirdInstructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
    cv.putText(img, "'Q' to quit", fourthInstructionPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)

    return img

# Return a list of images with different paths based on the different pathfinding methods
def draw_arrows(path, img):
    new_img = img.copy()
    # List of lists of nodes, where each list is to be an arrow
    nodes = []
    firstMember = True

    # Add points to draw along
    for member in path:
        if DEBUG: 
            print(f"Drawing arrow to {member}...")
        if firstMember:
            prev = member
            firstMember = False
            continue
        # Get waypoints from "prev" to "member"
        prev_node = prev.node
        if DEBUG: 
            print(f"First node: {prev_node.num}")

        if METHOD == Pathfinding.GREEDY_PICKING:
            path = prev_node.get_greedy_path(member.node)
        elif METHOD == Pathfinding.DIJKSTRA or METHOD == Pathfinding.DIJKSTRA_WITH_OVERLAP:
            path = prev_node.get_dijkstra_path(member.node)

        # Bezierify-ing lines
        if BEZIERIFY:
            # List of coordinates that do not correspond to any nodes
            coords_lst = prev_node.get_bezier_path(path)
            # Create temporary nodes that point to coords in Node.node_dict
            num_nodes = len(Node.node_dict)
            new_num = num_nodes + 1
            new_nodes = []
            for coords in coords_lst:
                Node.node_dict[new_num] = Node(new_num, (coords[0], coords[1]))
                new_nodes.append(new_num)
                new_num += 1
            nodes.append(new_nodes)
        else:
            nodes.append(path)

        prev = member
    if DEBUG:
        bezier_str = " - Bezier-ified" if BEZIERIFY else ""
        print(f"{METHOD}{bezier_str}'s path: {nodes}")

    new_img = draw_points(nodes, new_img)
    
    return new_img

# Un-select all nodes
def reset_node_status():
    if METHOD == Pathfinding.GREEDY_PICKING:
        # Reset Node.node_dict
        for num, node in Node.node_dict.items():
            node.selected = False
    else:
        # Reset DijkstraNode.node_dict
        for num, node in DijkstraNode.node_dict.items():
            node.selected = False

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
    global METHOD
    global BEZIERIFY
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

    ## Stop if no input image
    is_current_config_saved = False
    while NO_IMAGE:
        instruction = "Type 'q' to quit, 'r' to generate a new path\n" if is_current_config_saved else "Type 'q' to quit, 's' to save current path', 'r' to generate a new path\n"
        char = input(instruction).lower()
        # If "x" key pressed, close image
        if char[0] == "q":
            return
        elif char[0] == "s":
            if is_current_config_saved:
                print("Path already saved!")
            else:
                # Save assignments
                save_assignments(path)
                print("Current path saved!")
                is_current_config_saved = True
        elif char[0] == "r":
            is_current_config_saved = False
            generate_path()
            return

    ## Draw arrows on picture

    # Load image
    img = cv.imread(IMGFILEPATH)
    if img is None:
        sys.exit("Could not read the image.")
    
    # Display graph with annotated nodes and edges, can be used to save an image of the graph
    if DISPLAY_GRAPH:
        graph_img = display_graph(img)
    # Controls for image prompt, "q" to quit, "s" to save image, "c" to continue to display path
    while DISPLAY_GRAPH:
        k = cv.waitKey(0)
        if k == ord("q"):
            # exit
            return
        elif k == ord("s"):
            cv.imwrite(GRAPHFILEPATH, graph_img)
            print(f"Graph saved in {GRAPHFILEPATH}")
        elif k == ord("c"):
            # continue to display path
            cv.destroyAllWindows()
            DISPLAY_GRAPH = False
            break

    # Draw arrows
    image_to_show = draw_arrows(path, img)
    
    is_current_config_saved = False
    while True:
        if is_current_config_saved:
            saveNotifPos = (10, 530)
            cv.putText(image_to_show, "Path saved!", saveNotifPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
        cv.imshow("Generated Path", image_to_show)
            
        k = cv.waitKey(0)
        # If "x" key pressed, close image
        if k == ord("q"):
            break
        elif k == ord("s"):
            # Save assignments
            save_assignments(path)
            print("Current path saved!")
            is_current_config_saved = True
        elif k == ord("x"):
            # Automatically remove Bezierification
            BEZIERIFY = False
            # Use next method
            curr_method = METHOD.value
            next_method = curr_method + 1 if curr_method < NUM_METHODS - 1 else 0
            reset_node_status()
            METHOD = Pathfinding(next_method)
            print(f"Now using: {METHOD}")
            # re-draw with new METHOD
            image_to_show = draw_arrows(path, img)
        elif k == ord("z"):
            # Automatically remove Bezierification
            BEZIERIFY = False
            # Use previous method
            curr_method = METHOD.value
            next_method = curr_method - 1 if curr_method != 0 else NUM_METHODS - 1
            reset_node_status()
            METHOD = Pathfinding(next_method)
            print(f"Now using: {METHOD}")
            # re-draw with new METHOD
            image_to_show = draw_arrows(path, img)
        elif k == ord("b"):
            # Toggle Bezier-ification of current paths
            BEZIERIFY = not BEZIERIFY
            display_str = f"Converting {METHOD} path into Bezier curves!" if BEZIERIFY else f"Now using: {METHOD} without Bezier curves!"
            print(display_str)
            reset_node_status()
            # re-draw 
            image_to_show = draw_arrows(path, img)
        elif k == ord("r"):
            is_current_config_saved = False
            generate_path()
            break

generate_path()
