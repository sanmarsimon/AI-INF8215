####################################################################
# 		* Simon, Sanmar (1938126)
# 		* Harti, Ghali (1953494)
####################################################################
"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import annotations
import traceback
from time import time
from Tree import Tree
from CustomBoard import CustomBoard
from quoridor import *


class MyAgent(Agent):
    """My Quoridor agent."""

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """

        if time_left is None:
            time_left = float('inf')

        print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')

        initial_board = CustomBoard(percepts)

        try:
            tree = Tree(player=player, initial_board=initial_board)

            nb_iterations_left = self.get_nb_iteration_left(initial_board,player)
            maximum_time_to_spend = self.get_maximum_time_to_spend(step,time_left)

            if maximum_time_to_spend == 0:
                shortest_path = initial_board.get_shortest_path(player)
                return 'P', shortest_path[0][0], shortest_path[0][1]

            start_time = time()
            while True:
                print(f"Iteration remaining {nb_iterations_left}")

                promisingNode = tree.getInterestingNode()

                tree.expand(promisingNode)

                simulation_result = tree.simulate(promisingNode)

                tree.backPropagate(promisingNode, simulation_result)
                nb_iterations_left -= 1
                if nb_iterations_left == 0:
                    break
                elapsed_time = time() - start_time
                if elapsed_time >= maximum_time_to_spend:
                    break

            # Phase 5 - Simulation ended, chose the best action to do
            best_child_node_action = tree.get_best_child_action()
            return best_child_node_action

        except:
            print(traceback.format_exc())
            print(initial_board)
            # In case of unexpected failure, do a random action
            return random.choice(initial_board.get_actions(player))

    def get_maximum_time_to_spend(self, step, time_left):
        """
        This functions estimates the time to spend for a play
        Args:
            step: Current step number in game
            time_left: Time left in the game

        Returns:
            Estimate of the time to spend for the next action
        """
        MAXIMUM_STEPS_IN_GAME = 40
        player_action_no = (step + 1) // 2
        if player_action_no < 6:
            return player_action_no
        elif player_action_no < 27:
            return (time_left - 60) / (27 - player_action_no)
        elif player_action_no < MAXIMUM_STEPS_IN_GAME:
            return (time_left - 2) / (MAXIMUM_STEPS_IN_GAME - player_action_no)
        else:
            return 0

    def get_nb_iteration_left(self, initial_board, player):
        """
        This function estimates the maximum number of iterations for each round
        Args:
            initial_board: Current board
            player: Current playing player number

        Returns: Number of iterations left to play

        """
        MAXIMUM_STEPS_IN_GAME = 550
        nb_iterations_left = MAXIMUM_STEPS_IN_GAME
        if initial_board.nb_walls[player] == 0:
            nb_iterations_left = 1
        return nb_iterations_left


if __name__ == "__main__":
    agent_main(MyAgent())
