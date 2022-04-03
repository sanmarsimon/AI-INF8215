# ####################################################################
# # 		* Simon, Sanmar (1938126)
# # 		* Harti, Ghali (1953494)
# ####################################################################
#
# import random
# from typing import List, Set, Tuple
#
# from CustomBoard import CustomBoard
# from Node import Node
# from quoridor import NoPath
#
#
# class Tree:
#
#     def __init__(self, player: int = 0, initial_board: CustomBoard = None):
#         opponent = 1 - player
#         self.root = Node(player=opponent, board=initial_board, action=None,
#                              U=0, N=0)
#
#     def getInterestingNode(self) -> Node:
#         """
#         If there are multiple nodes giving the highest UCT,
#         the node going into the direction of the shortest path is prioritized
#         """
#         node = self.root
#
#         while len(node.children) != 0 and node.N > 0:
#             all_ucts = list(map(lambda child: child.get_uct_value(), node.children))
#             max_uct = max(all_ucts)
#             nodes_max_uct = [node.children[i] for i in range(len(node.children)) if all_ucts[i] == max_uct]
#
#             pawns_nodes_max_uct, walls_nodes_max_uct = self.separate_pawns_walls_nodes(nodes_max_uct)
#             nodes_max_uct_after_shortest_path = [n for n in pawns_nodes_max_uct if n.following_shortest_path]
#
#             if len(nodes_max_uct_after_shortest_path) > 0:
#                 node = random.choice(nodes_max_uct_after_shortest_path)
#             elif len(pawns_nodes_max_uct):
#                 node = random.choice(pawns_nodes_max_uct)
#             else:
#                 node = random.choice(walls_nodes_max_uct)
#
#         return node
#
#     @staticmethod
#     def getPlayersFromNode(nodePlayer):
#         return 1 - nodePlayer, nodePlayer
#
#     def expand(self, node: Node):
#         current_board = node.board
#         if current_board.is_finished():
#             return node
#
#         player, opponent = self.getPlayersFromNode(node.player)
#
#         player_shortest_path = None
#
#         try:
#             player_shortest_path = current_board.get_shortest_path(player)
#             has_shortest_path = True
#         except NoPath:
#             has_shortest_path = False
#
#         if current_board.nb_walls[player] == 0 and has_shortest_path:
#             action = 'P', player_shortest_path[0][0], player_shortest_path[0][1]
#             new_board = current_board.clone()
#             new_board.play_action_with_no_check(action, player)
#             new_node = Node(player=player, action=action, following_shortest_path=True, board=new_board)
#             node.addChild(new_node)
#             return new_node
#
#         for action in current_board.get_legal_pawn_moves(player):
#             new_board = current_board.clone()
#             new_board.play_action_with_no_check(action, player)
#             following_shortest_path = has_shortest_path and (
#                 player_shortest_path[0][0], player_shortest_path[0][1]) == (
#                                           action[1], action[2])
#             node.addChild(Node(player=player, action=action, following_shortest_path=following_shortest_path, board=new_board))
#
#         all_walls = self.getInterestingWalls(current_board, current_board.pawns[opponent])
#         for is_horizontal_wall, wall_y, wall_x in all_walls:
#             if not current_board.is_wall_possible_here((wall_y, wall_x),is_horizontal_wall):
#                 continue
#             new_board = current_board.clone()
#             new_board.add_wall_with_no_check((wall_y, wall_x), is_horizontal_wall, player)
#             action = 'WH' if is_horizontal_wall else 'WV', wall_y, wall_x
#             new_board.play_action_with_no_check(action, player)
#             node.addChild(
#                 Node(player=player, action=action, board=new_board))
#
#         return random.choice(node.children)
#
#     def simulate(self, node: Node):
#         board = node.board
#         player, opponent = self.getPlayersFromNode(node.player)
#
#         player_shortest_path = board.min_steps_before_victory_safe(
#             player=player)
#         opponent_shortest_path = board.min_steps_before_victory_safe(
#             player=opponent)
#
#         initial_player = 1 - self.root.player
#         if initial_player == player:
#             return int(player_shortest_path <= opponent_shortest_path)
#         else:
#             return int(opponent_shortest_path <= player_shortest_path)
#
#     def backPropagate(self, node: Node, simulation_result: int):
#         while node:
#             node.U += simulation_result
#             node.N += 1
#             node = node.parent
#
#     def getInterestingWalls(self, current_board, opponent_pos):
#         # Set of walls of interest to return
#         interesting_walls = set()
#
#         # Adding all the walls that are close to pawns
#         for wall_y, wall_x in current_board.horiz_walls:
#             interesting_walls.add((True, wall_y, wall_x - 2))
#             interesting_walls.add((True, wall_y, wall_x + 2))
#             for y in range(-1, 2):
#                 for x in range(-1, 2):
#                     interesting_walls.add((False, wall_y - y, wall_x - x))
#
#         for wall_y, wall_x in current_board.verti_walls:
#             interesting_walls.add((False, wall_y + 2, wall_x))
#             interesting_walls.add((False, wall_y - 2, wall_x))
#             for y in range(-1, 2):
#                 for x in range(-1, 2):
#                     interesting_walls.add((True, wall_y - y, wall_x - x))
#
#         # Adding all the walls that are close to game walls
#         opponent_y, opponent_x = opponent_pos
#         opponent_y -= 1
#         interesting_walls.add((True, opponent_y, opponent_x - 1))
#         interesting_walls.add((True, opponent_y, opponent_x))
#         interesting_walls.add((True, opponent_y + 1, opponent_x - 1))
#         interesting_walls.add((True, opponent_y + 1, opponent_x))
#         opponent_y += 1
#         opponent_x -= 1
#         interesting_walls.add((False, opponent_y, opponent_x))
#         interesting_walls.add((False, opponent_y, opponent_x + 1))
#         interesting_walls.add((False, opponent_y - 1, opponent_x))
#         interesting_walls.add((False, opponent_y - 1, opponent_x + 1))
#
#         return interesting_walls
#
#     def get_best_child_action(self):
#         player, opponent = self.getPlayersFromNode(self.root.player)
#         nodes_max_N = [node for node in self.root.children if node.N == max(list(map(lambda child: child.N, self.root.children)))]
#         gains = list(map(lambda node: self.get_node_gain(node),nodes_max_N))
#         nodes_max_gains = [nodes_max_N[i] for i in range(len(nodes_max_N)) if gains[i] == max(gains)]
#
#         pawns_nodes, walls_nodes = self.separate_pawns_walls_nodes(nodes_max_gains)
#
#         # Case : No action with wall moving
#         if len(walls_nodes) == 0 or self.root.board.nb_walls[player] == 0:
#             return self.get_random_node(pawns_nodes).action
#
#         # Case : No action with pawn moving
#         if len(pawns_nodes) == 0:
#             return self.get_random_node(walls_nodes).action
#
#         # Case : Equality
#         random_wall_node = self.get_random_node(walls_nodes)
#         random_pawn_node = self.get_random_node(pawns_nodes)
#         if random_wall_node.board.min_steps_before_victory_safe(player) \
#             >= random_wall_node.board.min_steps_before_victory_safe(opponent):
#             return random_wall_node.action
#
#         return random_pawn_node.action
#
#     @staticmethod
#     def get_random_node(nodes):
#         return random.choice(nodes)
#
#     def separate_pawns_walls_nodes(self, nodes):
#         pawns_nodes = []
#         walls_nodes = []
#         for node in nodes:
#             if node.action[0] == 'P':
#                 pawns_nodes.append(node)
#             else:
#                 walls_nodes.append(node)
#         return pawns_nodes, walls_nodes
#
#     def get_node_gain(self, node: Node) -> int:
#         player, opponent = self.getPlayersFromNode(self.root.player)
#         if node.action[0] == "P":
#             return self.root.board.min_steps_before_victory_safe(player) - node.board.min_steps_before_victory_safe(player)
#         else:
#             opponentGain = node.board.min_steps_before_victory_safe(opponent) - self.root.board.min_steps_before_victory_safe(opponent)
#             playerGain = node.board.min_steps_before_victory_safe(player) - self.root.board.min_steps_before_victory_safe(player)
#             return opponentGain - playerGain
