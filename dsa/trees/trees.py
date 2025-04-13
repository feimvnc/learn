"""
Node: or key, or payload
Edge: connect two nodes
Root: only node having no incoming edges
Path: ordered list of nodes that are connected by edges
Children: set of nodes that having incoming edges from the same node 
Parent: a node to which it connects with outgoing edges
Sibling: nodes with same parent
Subtree: a set of nodes and edges comprised of a parent and all the descendants of that parent
Leaf node: node without children
Level: number of path from root to node n
Height: maximum level of any node in the tree


"""

# Representing a tree

class Node: 
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def insert_left(self, child):
        if not self.left:  # no existing node
            self.left = child
        else:
            # push existing node down one level in the tree
            child.left = self.left  
            self.left = child

    def insert_right(self, child):
        if self.right is None:
            self.right = child
        else:
            child.right = self.right
            self.right = child