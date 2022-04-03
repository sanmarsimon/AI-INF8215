####################################################################
# 		* Simon, Sanmar (1938126)
# 		* Harti, Gali
####################################################################

from __future__ import annotations

from math import log, sqrt
from typing import List, Tuple

from MCTSBoard import MCTSBoard


class Node:

    def __init__(self, player: int = 0, action: Tuple[str, int, int] = None,
                 following_shortest_path: bool = False,
                 board: MCTSBoard = None, U: int = 0, N: int = 0):
        """Node constructor

        Args:
            player (int, optional): The player. Defaults to 0.
            action (Tuple[str, int, int], optional): The action done at the node. Defaults to None.
            following_shortest_path (bool, optional): A boolean indicating whether the action is folling the shortest path or not. Defaults to False.
            board (MCTSBoard, optional): The new board after the action is done. Defaults to None.
            U (int, optional): Number of win following the node. Defaults to 0.
            N (int, optional): Number of simulation involving the node. Defaults to 0.
        """

        self.player = player
        self.action = action
        self.following_shortest_path = following_shortest_path
        self.board = board
        self.U = U
        self.N = N
        self.children: List[Node]
        self.children = []
        self.parent = None

    def addChild(self, child: Node):

        child.parent = self
        self.children.append(child)

    def get_uct_value(self) -> float:
        """Returns UCT value of the node.
        If the node has not been simulated infinity is returned to make sure
        to simulate the node at the next iteration
        """

        N = self.N
        if N == 0:
            return float('inf')
        else:
            N_parent = self.parent.N
            U = self.U
            return (U / N) + sqrt(2) * sqrt(log(N_parent) / N)
