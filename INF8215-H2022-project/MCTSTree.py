####################################################################
# 		* Simon, Sanmar (1938126)
# 		* Harti, Gali
####################################################################

import random
from typing import List, Set, Tuple

from MCTSBoard import MCTSBoard
from Node import Node
from quoridor import NoPath


class StateTree:

    def __init__(self, player: int = 0, initial_board: MCTSBoard = None):
        opponent = 1 - player
        self.root = Node(player=opponent, board=initial_board, action=None,
                             U=0, N=0)

    def selectNode(self) -> Node:
        """
        If there are multiple nodes giving the highest UCT,
        the node going into the direction of the shortest path is prioritized
        """
        node = self.root

        while len(node.children) != 0 and node.N > 0:
            all_ucts = list(
                map(lambda child: child.get_uct_value(), node.children))
            max_uct = max(all_ucts)
            nodes_with_max_uct = [node.children[i] for i in
                                  range(len(node.children)) if
                                  all_ucts[i] == max_uct]
            pawns_nodes_with_max_uct, walls_nodes_with_max_uct = self.separate_pawns_walls_nodes(
                nodes_with_max_uct)
            nodes_with_max_uct_following_shortest_path = [n for n in pawns_nodes_with_max_uct if n.following_shortest_path]

            if len(nodes_with_max_uct_following_shortest_path) >= 1:
                node = random.choice(nodes_with_max_uct_following_shortest_path)
            elif len(pawns_nodes_with_max_uct):
                node = random.choice(pawns_nodes_with_max_uct)
            else:
                node = random.choice(walls_nodes_with_max_uct)

        return node

    def expand(self, node: Node) -> Node:
        """
        Expand a specific node.

        Returns:
            Node: The random leaf node after the expanding the most promising node
        """
        current_board = node.board
        if current_board.is_finished():
            return node

        player = 1 - node.player
        opponent = node.player

        player_shortest_path = None

        try:
            player_shortest_path = current_board.get_shortest_path(player)
            has_shortest_path = True
        except NoPath:
            has_shortest_path = False

        #######################################################################
        # If no walls are remaining, we only add the pawn move
        # that bring us to the shortest path
        #######################################################################
        if current_board.nb_walls[player] == 0 and has_shortest_path:
            action = 'P', player_shortest_path[0][0], player_shortest_path[0][1]
            new_board = current_board.clone()
            new_board.play_action_with_no_check(action, player)
            new_node = Node(player=player, action=action,
                                following_shortest_path=True, board=new_board)
            node.addChild(new_node)
            return new_node

        #######################################################################
        # We add the pawn moves
        #######################################################################
        for action in current_board.get_legal_pawn_moves(player):
            new_board = current_board.clone()
            new_board.play_action_with_no_check(action, player)
            following_shortest_path = has_shortest_path and (
                player_shortest_path[0][0], player_shortest_path[0][1]) == (
                                          action[1], action[2])
            node.addChild(Node(player=player, action=action,
                                    following_shortest_path=following_shortest_path,
                                    board=new_board))

        #######################################################################
        # Only take walls that are close to the opponent or that touch
        # other walls
        #######################################################################

        # Get walls that touch the opponent
        oppo_y, oppo_x = current_board.pawns[opponent]
        walls_close_to_pawns = self.get_walls_close_pawns(oppo_y, oppo_x)

        # Get walls attached to other walls
        walls_close_to_walls = self.getAdjacentWalls(current_board)

        # We add nodes corresponding to walls
        all_walls = walls_close_to_pawns.union(walls_close_to_walls)
        for is_horizontal_wall, wall_y, wall_x in all_walls:
            if not current_board.is_wall_possible_here((wall_y, wall_x),
                                                       is_horizontal_wall):
                continue
            new_board = current_board.clone()
            new_board.add_wall_with_no_check((wall_y, wall_x),
                                             is_horizontal_wall, player)

            action = 'WH' if is_horizontal_wall else 'WV', wall_y, wall_x
            new_board.play_action_with_no_check(action, player)
            node.addChild(
                Node(player=player, action=action, board=new_board))

        return random.choice(node.children)

    def simulate(self, node: Node) -> int:
        """
        We simulate the node. We check if a node is better than another one by comparing
        the length of the players shortest path to victory and the length
        of his opponent shortest path to victory

        Args:
            node (Node): The node to simulate

        Returns:
            int: 1 if player won and 0 if player lost

        """
        board = node.board

        player = 1 - node.player
        opponent = node.player

        player_shortest_path = board.min_steps_before_victory_safe(
            player=player)
        opponent_shortest_path = board.min_steps_before_victory_safe(
            player=opponent)

        initial_player = 1 - self.root.player
        if initial_player == player:
            return int(player_shortest_path <= opponent_shortest_path)
        else:
            return int(opponent_shortest_path <= player_shortest_path)
    
    def backPropagate(self, node: Node, simulation_result: int) -> None:
        while node:
            node.U += simulation_result
            node.N += 1
            node = node.parent
    
    def getAdjacentWalls(self, current_board: MCTSBoard) -> Set[Tuple[bool, int, int]]:
        """
        Get all the walls adjacent to other walls
        Args:
            current_board (MCTSBoard): The board from which to retrieve the walls

        Returns:
            Set[Tuple[bool, int, int]]: The list of walls
        """
        walls_close_to_walls = set()

        for wall_y, wall_x in current_board.horiz_walls:
            walls_close_to_walls.add((True, wall_y, wall_x - 2))
            walls_close_to_walls.add((True, wall_y, wall_x + 2))
            for y in range(-1, 2):
                for x in range(-1, 2):
                    walls_close_to_walls.add((False, wall_y - y, wall_x - x))
        for wall_y, wall_x in current_board.verti_walls:
            walls_close_to_walls.add((False, wall_y - 2, wall_x))
            walls_close_to_walls.add((False, wall_y + 2, wall_x))
            for y in range(-1, 2):
                for x in range(-1, 2):
                    walls_close_to_walls.add((True, wall_y - y, wall_x - x))

        return walls_close_to_walls

    def get_walls_close_pawns(self, oppo_y, oppo_x) -> Set[Tuple[bool, int, int]]:
        """
        Get all the walls close to a specific pawn position
        Args:
            oppo_y: The opponent y position
            oppo_x: The opponent x position

        Returns:
            Set[Tuple[bool, int, int]]: The list of walls

        """
        walls_close_to_pawns = set()
        oppo_y -= 1
        walls_close_to_pawns.add((True, oppo_y, oppo_x - 1))
        walls_close_to_pawns.add((True, oppo_y, oppo_x))
        walls_close_to_pawns.add((True, oppo_y + 1, oppo_x - 1))
        walls_close_to_pawns.add((True, oppo_y + 1, oppo_x))
        oppo_y += 1
        oppo_x -= 1
        walls_close_to_pawns.add((False, oppo_y, oppo_x))
        walls_close_to_pawns.add((False, oppo_y - 1, oppo_x))
        walls_close_to_pawns.add((False, oppo_y, oppo_x + 1))
        walls_close_to_pawns.add((False, oppo_y - 1, oppo_x + 1))
        return walls_close_to_pawns

    def get_best_child_action(self) -> Tuple[str, int, int]:
        """
        We pick the best action by taking the node that has been simulated the most. If a few nodes
        haves been simulated the same number of time, we pick the action that give us the most gains.

        Returns:
            Tuple[str, int, int]: The best action to perform
        """
        player = 1 - self.root.player
        opponent = self.root.player

        max_N = max(list(map(lambda child: child.N, self.root.children)))
        nodes_with_max_N = [node for node in self.root.children if
                            node.N == max_N]
        gains = list(map(
            lambda node: self.get_node_gain(node),
            nodes_with_max_N
        ))
        max_gain = max(gains)
        nodes_with_max_gains = [nodes_with_max_N[i] for i in range(len(nodes_with_max_N)) if gains[i] == max_gain]

        # In case of equality, we should move the pawn first
        pawns_nodes, walls_nodes = self.separate_pawns_walls_nodes(
            nodes_with_max_gains)

        # If there is no action that involves adding walls, we pick pawn moves
        if len(walls_nodes) == 0 or self.root.board.nb_walls[player] == 0:
            return random.choice(pawns_nodes).action

        # If there is no action that involves moving paws, we add walls
        if len(pawns_nodes) == 0:
            return random.choice(walls_nodes).action

        # In case of equality between pawn_moves we take the actions that lead us to the shortest
        # path
        random_wall_node = random.choice(walls_nodes)
        random_pawn_node = random.choice(pawns_nodes)
        if random_wall_node.board.min_steps_before_victory_safe(player) \
            >= random_wall_node.board.min_steps_before_victory_safe(opponent):
            return random_wall_node.action

        return random_pawn_node.action

    def separate_pawns_walls_nodes(self, nodes: List[Node]) -> Tuple[
        List[Node], List[Node]]:
        """
        We separate the list of actions into two: pawn moves actions and adding walls action
        """
        pawns_nodes = []
        walls_nodes = []
        for node in nodes:
            if node.action[0] == 'P':
                pawns_nodes.append(node)
            else:
                walls_nodes.append(node)
        return pawns_nodes, walls_nodes

    def get_node_gain(self, node: Node) -> int:
        """
        We calculate the gain by returning the difference between the length of
        the players shortest path to victory and the length of his opponent shortest path to victory
        """
        root = self.root
        player = 1 - root.player
        opponent = root.player
        if node.action[0] == "P":
            return root.board.min_steps_before_victory_safe(player) \
                   - node.board.min_steps_before_victory_safe(player)
        else:
            opponentGain = node.board.min_steps_before_victory_safe(opponent) \
                           - root.board.min_steps_before_victory_safe(opponent)
            playerGain = node.board.min_steps_before_victory_safe(player) \
                         - root.board.min_steps_before_victory_safe(player)
            return opponentGain - playerGain
