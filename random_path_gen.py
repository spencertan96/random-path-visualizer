import os
import math
import random
import sys
import heapq
import numpy as np
import cv2 as cv
from datetime import datetime
from enum import Enum, unique

@unique
class Pathfinding(Enum):
    GREEDY_PICKING = 0
    DIJKSTRA = 1
    DIJKSTRA_WITH_OVERLAP = 2

@unique
class MemberType(Enum):
    CIRCLE = 0
    SQUARE = 1

# Adjustable options
NO_IMAGE = False
FILEPATH = 'sample_input.txt'
SAVEFILEPATH = 'sample_previous_path.txt'
IMGFILEPATH = 'sample_img.png'
# Folder to save images for whatsapp bot (any argument to activate)
IMG_FOLDER_NAME = 'path_images'
IMG_NAME = 'random_path_gen_img'
# FILEPATH = 'showcase_input.txt'
# SAVEFILEPATH = 'showcase_previous_path.txt'
# IMGFILEPATH = 'sample_map.jpg'
IMG_DIMENSIONS = [(1000,750), (1200,900), (1400,1050), (1600,1200)]
CURR_DIMENSION_INDEX = 0
# Only matters if NO_IMAGE is False
MEMBER_NUM = 11
# MEMBER_NUM = 18
# Number of points in a bezier curve
# (The higher it is, the smoother curves will be at the expense of computation time)
NUM_POINTS = 15
# Not very useful option
EACH_MEMBER_ALL_NEIGHBOURS = True
# Developer options
DISPLAY_GRAPH = False
GRAPHFILEPATH = 'sample_graph.png'
DEBUG = False
# Initial pathfinding method
METHOD = Pathfinding.GREEDY_PICKING

# Constants
BEZIERIFY = False
ADJUST_CONTROL_POINTS = False
NUM_METHODS = len(Pathfinding)
ARROWTIP_RATIO = 15
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
MEMBERS = []
# to deal with floating point inaccuracies
EPSILON = 0.0000001

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
            generate_path(False)
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
    # Split nodes into groups of 3 to make quadratic bezier curves and connect them to make path.
    # Get points from curve at segments of t.
    # Output: List of coordinates as tuples
    def get_bezier_path(self, path):
        seq = []
        paths = []
        for i in range(0, len(path), 3):
            next_index = i + 3
            next_path_len = len(path[next_index:])
            if next_path_len <= 2:
                # combine with current path and end
                paths.append(path[i:])
                break
            else:
                paths.append(path[i:i+3])
        for path in paths:
            # get bezier curve for each path
            seq.extend(get_bezier_points(path))
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

# Input: Control points for bezier curve
# Output: List of points that make up bezier curve (NUM_POINTS points in curve)
def get_bezier_points(ctrl_pts):
    seq = []
    t_steps = 1 / (NUM_POINTS - 1)
    t = 0
    while t <= 1 + EPSILON:
        arr = []
        arr2 = []
        # Initialize array with first set of LERPs, path needs to and is guaranteed to be > 2 nodes
        # path first contains node_nums
        for i in range(len(ctrl_pts)):
            if i == 0:
                continue
            # add LERPS to arr
            val = lerp(Node.node_dict[ctrl_pts[i-1]].coords, Node.node_dict[ctrl_pts[i]].coords, t)
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

# Each member has a name.
class Member:
    # Each member is tied to a node
    # Unless there is no image as no node required
    def __init__(self, name, type, size, is_excluded, node, neighbors = ()) -> None:
        self.name = name
        self.type = type
        self.size = size
        self.is_excluded = is_excluded
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

            if self.neighbors[index].selected or self.neighbors[index].is_excluded:
                # ran out of choices
                if len(self.neighbors) == 1:
                    if DEBUG:
                        print(f"No neighbors available for {self.name}! Restarting path!")
                    return None
                # update neighbor list
                intermediateLst = list(self.neighbors)
                intermediateLst.pop(index)
                self.neighbors = intermediateLst
                continue
            else:
                break
        # member should never have no neighbors, here to catch just in case
        if len(self.neighbors) == 0:
            print(f"{self.name} has no more unselected neighbours!")
            return None
        member = self.neighbors[index]
        member.selected = True
        # remove selected neighbor
        # print(self.neighbors)
        intermediateLst = list(self.neighbors)
        intermediateLst.pop(index)
        self.neighbors = intermediateLst
        return member

# Returns list of members that the path will be made out of
def initialize_members():
    members = read_simple_input_file(FILEPATH) if NO_IMAGE else read_input_file(FILEPATH)
    display_str = "["
    for member in members:
        is_excluded = " - Excluded" if member.is_excluded else ""
        display_str += member.name + is_excluded + ", "
    display_str = display_str[:-2] + "]" 
    print("Members:", display_str)
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
            members = []
            for name in names:
                members.append(Member(name, None))
            return members
    else:
        raise AssertionError(f"Input file {FILEPATH}' not found!")

# Read input file, populate Node.node_dict, return list of members
def read_input_file(filepath):
    # Load from file
    # Format: {MemberName}:{XCoordinate},{YCoordinate}:(C/S}{Size})
    #         {NodeNumber}:{XCoordinate},{YCoordinate}
    #         till empty line, then
    #         {NodeNumber}:{ConnectedNode},{ConnectedNode2}
    members_initialized = 0
    members = []
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
                    name = text[0]
                    coords = text[1].split(",")
                    if text[2][0].lower() == "s":
                        type = MemberType.SQUARE
                    elif text[2][0].lower() == "c":
                        type = MemberType.CIRCLE
                    else:
                        raise AssertionError(f"Input Error! Not 'C' or 'S' for {name}'s representation type in '{FILEPATH}'!")
                    size = int(text[2][1:])
                    Node.node_dict[node_num] = Node(node_num, (int(coords[0]), int(coords[1])))
                    if len(text) <= 3:
                        members.append(Member(name, type, size, False, Node.node_dict[node_num]))
                    else:
                        members.append(Member(name, type, size, True, Node.node_dict[node_num]))
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
        raise AssertionError(f"Input file {FILEPATH}' not found!")
    return members

# Returns a dictionary of member: previously assigned member, or an empty dictionary if no previous assignment saved.
def retrieve_prev_assignments():
    if not os.path.exists(SAVEFILEPATH):
        print("No previous assignment found!")
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
            if member.is_excluded:
                continue
            for neighbor in members:
                if member.name == neighbor.name or neighbor.is_excluded:
                    continue
                member.add_neighbor(neighbor)
            # member.print_neighbors()

# Input: List of all members
# Output: List of members without excluded ones
def remove_excluded_members(members):
    output = []
    for member in members:
        if member.is_excluded:
            continue
        output.append(member)
    return output

def pick_member(members):
    random_start = random.randint(0, len(members) - 1)
    member = members[random_start]
    member.selected = True
    members.pop(random_start)
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

        if METHOD == Pathfinding.GREEDY_PICKING:
            node_path = prev_node.get_greedy_path(member.node)
        elif METHOD == Pathfinding.DIJKSTRA or METHOD == Pathfinding.DIJKSTRA_WITH_OVERLAP:
            node_path = prev_node.get_dijkstra_path(member.node)

        # Bezierify-ing lines
        if BEZIERIFY:
            # List of coordinates that do not correspond to any nodes
            coords_lst = prev_node.get_bezier_path(node_path)
            if ADJUST_CONTROL_POINTS:
                if DEBUG:
                    print(f"Checking if path {prev} to {member} is invalid...")
                coords_lst = validate_path(coords_lst, prev, member, node_path)
                if DEBUG:
                    print(f"Correction done from {prev} to {member}!")
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
            nodes.append(node_path)

        prev = member
    if DEBUG:
        bezier_str = " - Bezier-ified" if BEZIERIFY else ""
        print(f"{METHOD}{bezier_str}'s path: {nodes}")

    new_img = draw_points(nodes, new_img)
    
    return new_img

def display_instructions(img):
    # Display instructions
    x_pos = 6
    y_pos = 25
    line_spacing = 25
    font_size = 0.6
    instr_pos = []
    for i in range(0, 6):
        instr_pos.append((x_pos, y_pos + line_spacing * i))
    cv.putText(img, "'Z'/'X' to switch algorithms", instr_pos[0], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    cv.putText(img, "'B' to toggle Bezier curves", instr_pos[1], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    cv.putText(img, "'A' to toggle line adjustment", instr_pos[2], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    cv.putText(img, "'S' to save current path", instr_pos[3], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    cv.putText(img, "'R' to re-generate", instr_pos[4], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    cv.putText(img, "'Q' to quit", instr_pos[5], cv.FONT_HERSHEY_SIMPLEX, font_size, RED, 2, -1)
    return img

# Check each line segment and make sure no intersections with any other members in image.
# Re-bezierifies path. Creates temporary nodes with new coords.
# Input: list of coordinates, members involved in path, line path (list of node numbers)
# Output: Integer that cooresponds to control point to be shifted, if 0 returned, no intersection
def validate_path(coords_lst, member_from, member_to, path):
    new_coords_lst = []
    i = 0
    # Decreases by 20% each adjustment on the same node
    adjustment_factor = 30
    # Reduce adjustment factor only when same member intersecting
    previous_intersecting_mem = None
    while i < len(coords_lst):
        if i == 0:
            new_coords_lst.append(coords_lst[i])
            i += 1
            continue
        p0 = new_coords_lst[i-1]
        p1 = coords_lst[i]
        
        (intersecting_member, closest_point) = test_intersection(p0, p1, member_from, member_to)
        # find closest node to intersecting circle, adjust and re-generate bezier curve
        # Circle: Adjusts using adjust_factor * (circle.center to node) vector
        # Square: Adjusts by shifting towards vector: adjust_factor * norm(square.center to closest square.corner)
        if intersecting_member:
            # Adjustment factor
            if previous_intersecting_mem and previous_intersecting_mem == intersecting_member:
                adjustment_factor = 0.8 * adjustment_factor if adjustment_factor > 2.5 else 2
            else:
                previous_intersecting_mem = intersecting_member
                adjustment_factor = 30

            if intersecting_member.type == MemberType.CIRCLE:
                (nodenum_to_adjust, node_index) = find_closest_node(path, closest_point)
                node_coords = Node.node_dict[nodenum_to_adjust].coords
                centerToPt = (node_coords[0] - closest_point[0], node_coords[1] - closest_point[1])
                adjust_vec = normalize(centerToPt)
                new_pos = (node_coords[0] + round(adjustment_factor * adjust_vec[0]), node_coords[1] + round(adjustment_factor * adjust_vec[1]))
            elif intersecting_member.type == MemberType.SQUARE:
                (nodenum_to_adjust, node_index) = find_closest_node(path, closest_point)
                node_coords = Node.node_dict[nodenum_to_adjust].coords
                corner = find_closest_corner(intersecting_member, node_coords)
                # node_to_corner = normalize((corner[0] - node_coords[0], corner[1] - node_coords[1]))
                # center_to_corner = normalize((corner[0] - intersecting_member.coords[0], corner[1] - intersecting_member.coords[1]))
                center_to_node = normalize((node_coords[0] - intersecting_member.coords[0], node_coords[1] - intersecting_member.coords[1]))
                # OUTWARD_FACTOR = 2
                # adjust_vec = (node_to_corner[0] + OUTWARD_FACTOR * center_to_corner[0], node_to_corner[1] + OUTWARD_FACTOR * center_to_corner[1])
                adjust_vec = center_to_node
                new_pos = (node_coords[0] + round(adjustment_factor * adjust_vec[0]), node_coords[1] + round(adjustment_factor * adjust_vec[1]))
                # print(f"Moving N{nodenum_to_adjust} by {adjustment_factor:.1f}x({adjust_vec[0]:.3f}, {adjust_vec[1]:.3f}) from {member_from} to {member_to} avoiding {intersecting_member}")
            
            if DEBUG:
                print(f"Adjusting node: {nodenum_to_adjust} by {adjustment_factor} x {adjust_vec}")
            # create temp node to hold new_pos
            num_nodes = len(Node.node_dict)
            new_num = num_nodes + 1
            new_node = Node(new_num, new_pos)
            Node.node_dict[new_num] = new_node
            path.pop(node_index)
            path.insert(node_index, new_node.num)
            
            # Reset params
            coords_lst = member_from.node.get_bezier_path(path)
            new_coords_lst = []
            i = 0
            continue 
        new_coords_lst.append(p1)
        i += 1
    return new_coords_lst

# Each member is approximated to a circle.
# Input: 2 points of line segment to be tested for intersections
# Output: Intersecting member if any, else None
def test_intersection(p0, p1, member_from, member_to):
    for member in MEMBERS:
        # Ignore member_from and member_to
        if member.name == member_from.name or member.name == member_to.name:
            continue
        center = member.coords

        if member.type == MemberType.CIRCLE:
            # LERP between 2 points
            # x = (1-t)*x0 + t*x1
            # y = (1-t)*y0 + t*y1
            # Find intersection between line and circle
            # If 0 <= t <= 1, has intersection
            # (x - h)^2 + (y - k)^2 = r^2
            # Subst. x and y into eqn, solve for t
            p0x_sq = p0[0] * p0[0]
            p1x_sq = p1[0] * p1[0]
            p0p1x = p0[0] * p1[0]
            p0y_sq = p0[1] * p0[1]
            p1y_sq = p1[1] * p1[1]
            p0p1y = p0[1] * p1[1]
            a = p0x_sq + p1x_sq - 2 * p0p1x + p0y_sq + p1y_sq - 2 * p0p1y
            b = -2 * p0x_sq + 2 * p0p1x + 2 * p0[0] * center[0] - 2 * p1[0] * center[0] - 2 * p0y_sq + 2 * p0p1y + 2 * p0[1] * center[1] - 2 * p1[1] * center[1]
            c = p0x_sq + center[0] * center[0] - 2 * p0[0] * center[0] + p0y_sq + center[1] * center[1] - 2 * p0[1] * center[1] - member.size * member.size
            if a == 0:
                # linear equation
                if b == 0:
                    t = - c
                else:
                    t = c / b
                if t >= 0 and t <= 1:
                    if DEBUG: 
                        print(f"Discrim.: {discrim}, Intersect {member.name} from {member_from.name} to {member_to.name}")
                    return (member, center)
                # no intersection
                continue
            discrim = b * b - 4 * a * c
            if discrim > 0:
                sqrt_discrim = math.sqrt(discrim)
                t1 = (- b - sqrt_discrim) / (2 * a)
                t2 = (- b + sqrt_discrim) / (2 * a)
                if (t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1):
                    if DEBUG: 
                        print(f"Discrim.: {discrim}, Intersect {member.name} from {member_from.name} to {member_to.name}")
                    return (member, center)
            elif discrim == 0:
                t1 = (- b) / (2 * a)
                if t1 >= 0 and t1 <= 1:
                    if DEBUG:
                        print(f"Discrim.: {discrim}, Intersect {member.name} from {member_from.name} to {member_to.name}")
                    return (member, center)
            # no intersection
            continue
        elif member.type == MemberType.SQUARE:
            # square intersection test
            if point_in_square(p0, center, member.size):
                intersection = get_square_intersection(p0, p1, member)
                return (member, intersection)
            else:
                if point_in_square(p1, center, member.size):
                    intersection = get_square_intersection(p1, p0, member)
                    return (member, intersection)
            continue
        else:
            raise AssertionError(f"Error! Invalid member representation type! Check input file {FILEPATH}!")
    return (None, None)

# Start and end nodes are ignored so that the line does not shift!
# Input: List of node numbers in path, coords to be compared with.
# Output: Node number that is closest to given coords and its index in path
def find_closest_node(path, coords):
    closest_node = None
    min_index = -1
    index = 0
    min_dist = sys.maxsize
    for node_num in path:
        if index == 0 or index == len(path) - 1:
            index += 1
            continue
        dist = get_distance_between(Node.node_dict[node_num].coords, coords)
        if dist < min_dist:
            min_dist = dist
            closest_node = node_num
            min_index = index
        index += 1
    if closest_node == None:
        # Can technically be because path contains only 2 nodes but 2 nodes between members should not intersect any other members.
        raise AssertionError(f"No closest node found (path might be empty!)\nNode path:{path}")
    return (closest_node, min_index)

# Input: Point to check, center of square, width of square
# Output: Boolean
def point_in_square(pt, center, width):
    # make width even
    width = width + 1 if width % 2 == 1 else width

    left_x = center[0] - width/2
    right_x = center[0] + width/2
    top_y = center[1] - width/2
    bottom_y = center[1] + width/2
    if pt[0] > left_x and pt[0] < right_x:
        if pt[1] > top_y and pt[1] < bottom_y:
            return True 
    return False

# Find coordinates of intersection between line segment (p0, p1) and square member.
# p0 is guaranteed to be in the square.
# Input: 2 points and the intersecting square
# Output: Coords of intersection
def get_square_intersection(p0, p1, sq):
    # checks to determine which side of square intersected, 0 = top, 1 = right, 2 = bottom, 3 = left
    # Use parametric equation to find next coordinate (x or y)
    # make width even
    width = sq.size + 1 if sq.size % 2 == 1 else sq.size

    left_x = sq.coords[0] - width/2
    right_x = sq.coords[0] + width/2
    top_y = sq.coords[1] - width/2
    bottom_y = sq.coords[1] + width/2

    # handle edge cases
    if p1[0] == p0[0]:
        # vertical line, top or bottom
        if bottom_y > p0[1] and bottom_y > p1[1]:
            side = 0
            return (p0[0], top_y)
        else:
            side = 2
            return (p0[0], bottom_y)
    elif p1[1] == p0[1]:
        # horizontal line, right or left
        if right_x > p0[0] and right_x / 2 > p1[0]:
            side = 3
            return (left_x, p0[1])
        else:
            side = 1
            return (right_x, p0[1])
    
    grad = (p1[1] - p0[1]) / (p1[0] - p0[0])
    if p0[0] - p1[0] > 0: # p1 | p0 |
        if grad > 0:
            # bottom/left
            x = ((bottom_y - p1[1]) / grad) + p1[0]
            if x > left_x:
                side = 2
            else:
                side = 3
        else:
            # top/left
            x = ((top_y - p1[1]) / grad) + p1[0]
            if x > left_x:
                side = 0
            else:
                side = 3
    else: # | p0 | p1
        if grad > 0:
            # top/right
            x = ((top_y - p1[1]) / grad) + p1[0]
            if x < right_x:
                side = 0
            else:
                side = 1
        else:
            # bottom/right
            x = ((bottom_y - p1[1]) / grad) + p1[0]
            if x < right_x:
                side = 2
            else:
                side = 1
    if side == 0:
        y = top_y
        x = ((y - p1[1]) / grad) + p1[0]
        return (x, y)
    elif side == 1:
        x = right_x
        y = grad * (x - p1[0]) + p1[1]
        return (x, y)
    elif side == 2:
        y = bottom_y
        x = ((y - p1[1]) / grad) + p1[0]
        return (x, y)
    elif side == 3:
        x = left_x
        y = grad * (x - p1[0]) + p1[1]
        return (x, y)

# Input: Square, point to check against
# Output: Coords of closest corner
def find_closest_corner(sq, pt):
    # make width even
    width = sq.size + 1 if sq.size % 2 == 1 else sq.size
    left_x = sq.coords[0] - width/2
    right_x = sq.coords[0] + width/2
    top_y = sq.coords[1] - width/2
    bottom_y = sq.coords[1] + width/2
    corners = [(left_x, bottom_y), (left_x, top_y), (right_x, top_y), (right_x, bottom_y)]
    min_dist = sys.maxsize
    closest = (0, 0)
    for corner in corners:
        dist = get_distance_between(pt, corner)
        if dist < min_dist:
            min_dist = dist
            closest = corner
    return closest

# Input: 2D-vector to normalise
# Output: Normalized vector
def normalize(vec):
    magnitude = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])            
    return (vec[0] / magnitude, vec[1] / magnitude)

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

    # draw outlines of members
    for member in MEMBERS:
        if member.type == MemberType.CIRCLE:
            graph_img = cv.circle(graph_img, member.coords, member.size, RED, 2)
        elif member.type == MemberType.SQUARE:
            top_left = (round(member.coords[0] - member.size / 2), round(member.coords[1] - member.size / 2))
            bottom_right = (round(member.coords[0] + member.size / 2), round(member.coords[1] + member.size / 2))
            graph_img = cv.rectangle(graph_img, top_left, bottom_right, RED, 2)
        else:
            raise AssertionError(f"Error! Invalid member representation type! Check input file {FILEPATH}!")
    
    for node in Node.node_dict.values():
        # annotate each node with its num
        pos = (node.coords[0] + 3, node.coords[1] + 1)
        cv.putText(graph_img, str(node.num), pos, cv.FONT_HERSHEY_PLAIN, 0.9, BLUE, 1, -1)
        
    for node in Node.node_dict.values():
        # draw node as dot
        graph_img = cv.circle(graph_img, node.coords, radius=2, color=RED, thickness=-1)

    # resize img
    graph_img = cv.resize(graph_img, IMG_DIMENSIONS[CURR_DIMENSION_INDEX], interpolation = cv.INTER_AREA)
    cv.imshow("Graph Structure", graph_img)
    return graph_img

def generate_path(shouldSaveImg):
    global MEMBERS
    global DISPLAY_GRAPH
    global METHOD
    global BEZIERIFY
    global ADJUST_CONTROL_POINTS
    global CURR_DIMENSION_INDEX
    path = []
    members = initialize_members()
    MEMBERS = members.copy()
    members = remove_excluded_members(members)
    prev_assignments = retrieve_prev_assignments()
    initialize_member_neighbours(members)
    # Pick random starting point
    # Randomly pick neighbour, continue with that neighbour
    (member, members) = pick_member(members)
    if DEBUG:
        print(f"Starting with: {member}")
    path.append(member)
    # Pick rest of the members
    while members:
        next = member.get_next(prev_assignments)
        if next == None:
            if DEBUG:
                print(f"Failed path:{path}")
            # No solution, try for another solution
            # Reset everything
            path = []
            members = initialize_members()
            members = remove_excluded_members(members)
            initialize_member_neighbours(members)
            # Pick random starting point
            # Randomly pick neighbour, continue with that neighbour
            (member, members) = pick_member(members)
            print(f"Restarting with: {member}")
            path.append(member)
            continue
        path.append(next)
        members.remove(next)
        member = next
    # For debugging, specify a fixed path using member indices
    # path = []
    # indices = [8, 4, 1, 10, 6, 7, 9, 3, 5, 2, 12, 11]
    # for index in indices:
    #     path.append(MEMBERS[index - 1])
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
            generate_path(shouldSaveImg)
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
    # Platform-dependent, might only work on Windows
    KEYCODE_UP = 2490368
    KEYCODE_DOWN = 2621440

    isImageSaved = False

    while True:
        if is_current_config_saved:
            saveNotifPos = (10, 530)
            cv.putText(image_to_show, "Path saved!", saveNotifPos, cv.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, -1)
        # resize img
        resized_img = cv.resize(image_to_show, IMG_DIMENSIONS[CURR_DIMENSION_INDEX], interpolation = cv.INTER_AREA)
        resized_img = display_instructions(resized_img)
        # if saving image, don't need to show
        if not shouldSaveImg:
            cv.imshow("Generated Path", resized_img)

        if (shouldSaveImg and not isImageSaved):
            isImageSaved = True
            os.makedirs(IMG_FOLDER_NAME, exist_ok=True)
            
            # use date + time in imgName
            currTimestamp = datetime.now()
            dateTimeStr = currTimestamp.strftime("%Y%m%d-%H%M%S")

            imgSavePath = os.path.join(IMG_FOLDER_NAME, IMG_NAME + "_" + dateTimeStr + ".png")
            cv.imwrite(imgSavePath, resized_img)
            
        k = cv.waitKeyEx(0)
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
            # Reset ADJUST_CONTROL_POINTS
            ADJUST_CONTROL_POINTS = False
            # Toggle Bezier-ification of current paths
            BEZIERIFY = not BEZIERIFY
            display_str = f"Converting {METHOD} path into Bezier curves!" if BEZIERIFY else f"Now using: {METHOD} without Bezier curves!"
            print(display_str)
            reset_node_status()
            # re-draw 
            image_to_show = draw_arrows(path, img)
        elif k == ord("a"):
            if not BEZIERIFY:
                print("Not currently using Bezier curves! Nothing to adjust!")
                continue
            # Toggle auto-adjustment of Bezier control points
            ADJUST_CONTROL_POINTS = not ADJUST_CONTROL_POINTS
            display_str = f"Adjusting Bezier curves to not intersect any members!" if ADJUST_CONTROL_POINTS else f"Removing Bezier curve adjustments!"
            print(display_str)
            reset_node_status()
            # re-draw 
            image_to_show = draw_arrows(path, img)
            print("Adjustment done! Drawing new image now...")
        # For developer use, keypress to save the final result as an image 
        # elif k == ord("f"):
        #     # Save image of final result
        #     RESULTFILEPATH = "sample_result.png"
        #     cv.imwrite(RESULTFILEPATH, image_to_show)
        #     print(f"Graph saved in {RESULTFILEPATH}")
        elif k == KEYCODE_UP:
            if CURR_DIMENSION_INDEX == len(IMG_DIMENSIONS):
                print(f"Already at largest window size!")
            else:
                CURR_DIMENSION_INDEX += 1
                print(f"Now using {CURR_DIMENSION_INDEX}-th window size!")
        elif k == KEYCODE_DOWN:
            if CURR_DIMENSION_INDEX == 0:
                print(f"Already at smallest window size!")
            else:
                CURR_DIMENSION_INDEX -= 1
                print(f"Now using {CURR_DIMENSION_INDEX}-th window size!")
        elif k == ord("r"):
            is_current_config_saved = False
            generate_path(shouldSaveImg)
            break

if __name__ == '__main__':
    if (len(sys.argv) <= 0):
        generate_path(False)
    else:
        generate_path(True)
