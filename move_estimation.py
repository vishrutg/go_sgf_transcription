from collections import defaultdict
from dataclasses import dataclass
import cv2

from enum import Enum
import numpy as np


class Stone(Enum):
    BLACK = 0
    EMPTY = 1
    WHITE = 2
def stone_value_to_type(x):
    if x == Stone.BLACK.value:
        return Stone.BLACK
    if x == Stone.WHITE.value:
        return Stone.WHITE
    if x == Stone.EMPTY.value:
        return Stone.EMPTY
def opposite_stone(stone):
    if stone == Stone.BLACK:
        return Stone.WHITE
    if stone == Stone.WHITE:
        return Stone.BLACK
    return None

class Action:
    def __init__(self, stone_type, prob, coord1=None,coord2=None, add_handicap_stones=[], no_action=False, is_pass=False):
        self.coord1=coord1
        self.coord2=coord2
        self.stone_type=stone_type
        self.is_pass = is_pass
        self.add_handicap_stones = add_handicap_stones
        self.no_action = no_action
        self.prob = prob

        possible_action_categories = [
            coord1 is not None and coord2 is not None,
            is_pass,
            len(add_handicap_stones) > 0,
            no_action
        ]
        assert sum(possible_action_categories) == 1
            
    def __hash__(self):
        if self.coord1 is not None and self.coord2 is not None:
            return int(self.coord1*self.coord2*(19**self.stone_type.value))
        else:
            return -1
    def __eq__(self, other):
        if self.coord1 != other.coord1:
            return False
        if self.coord2 != other.coord2:
            return False
        if self.stone_type != other.stone_type:
            return False
        if self.is_pass != other.is_pass:
            return False
        if self.no_action != other.no_action:
            return False
        for coord1, coord2 in self.add_handicap_stones:
            found = False
            for o_coord1, o_coord2 in other.add_handicap_stones:
                if o_coord1 == coord1 and o_coord2 == coord2:
                    found = True
            if not found:
                return False
        return True
    
    def __repr__(self):
        if self.coord1 is not None and self.coord2 is not None:
            args = {
                "coord1":self.coord1,
                "coord2":self.coord2,
            }
        elif self.is_pass:
            args = {
                "is_pass": self.is_pass,
            }
        elif self.add_handicap_stones:
            handicap_stones = ",".join([str(x) for x in self.add_handicap_stones])
            args = {
                "add_handicap_stones": handicap_stones,                
            }
        elif self.no_action:
            args = {
                "no_action": self.no_action
            }
        args["prob"] = self.prob
        args["stone_type"] = self.stone_type
        args = [f"{k}={v}" for k,v in args.items()]        
        return f"Action({','.join(args)})"
    def __str__(self):
        return self.__repr__()
    

class State:
    def __init__(self, board_state, last_to_play, black_captures=0, white_captures=0, ko_location=None, handicap_stones=[]):
        assert board_state.dtype == np.uint8
        self.board_state = board_state
        # do this so that it can be a hash function
        self.board_state.flags.writeable=False
        self.last_to_play = last_to_play
        self.black_captures = black_captures
        self.white_captures = white_captures
        self.ko_location = ko_location
        self.handicap_stones = handicap_stones
        self.observation_prob = {}
        self.prior_prob = {}
        self._valid_actions = None
        
    def __hash__(self):
        return hash(self.board_state.tobytes())        

    def __eq__(self, other):
        if np.any(self.board_state != other.board_state):
            return False
        if self.last_to_play != other.last_to_play:
            return False
        if self.black_captures != other.black_captures:
            return False
        if self.white_captures != other.white_captures:
            return False
        if np.any(self.ko_location != other.ko_location):
            return False
        for coord1, coord2 in self.handicap_stones:
            found = False
            for o_coord1, o_coord2 in other.handicap_stones:
                if o_coord1 == coord1 and o_coord2 == coord2:
                    found = True
            if not found:
                return False
        if self.can_add_handicap != other.can_add_handicap:
            return False
        return True


    @staticmethod
    def _get_group(board_state, coord1, coord2):
        group = {(coord1, coord2)}
        color = stone_value_to_type(board_state[coord1, coord2])
        if color == Stone.EMPTY:
            return None, None
        
        while True:
            updated_group = State._get_group_helper(board_state, group, color)
            if len(updated_group) == len(group):
                break
            group = updated_group


        liberties = State._get_stone_group_adjacencies(board_state, group)
        num_liberties = 0
        for (i, j) in liberties:
            assert board_state[i, j] != color
            if board_state[i,j] == Stone.EMPTY.value:
                num_liberties += 1

        return group, num_liberties

    @staticmethod
    def _get_group_helper(board_state, group, color):
        assert group
        assert color != Stone.EMPTY.value
        adjacencies = State._get_stone_group_adjacencies(board_state, group)
        for (coord1,coord2) in adjacencies:
            if board_state[coord1, coord2] ==  color.value:
                group.add((coord1,coord2))
        return group
                
    @staticmethod
    def _get_stone_group_adjacencies(board_state, group):
        # only return the immediate adjacencies in a set
        adjacencies = set()
        board_size = board_state.shape[0]
        for (coord1, coord2) in group:
            if coord1+1 < board_size and not (coord1+1,coord2) in group:
                adjacencies.add((coord1+1,coord2))
            if coord1-1 >= 0 and not (coord1-1,coord2) in group:
                adjacencies.add((coord1-1,coord2))
            if coord2+1 < board_size and not (coord1,coord2+1) in group:
                adjacencies.add((coord1,coord2+1))
            if coord2-1 >= 0 and not (coord1,coord2-1) in group:
                adjacencies.add((coord1,coord2-1))
        return adjacencies

    @staticmethod
    def dead_stones(board_state, last_action):
        if last_action.coord1 is None or last_action.coord2 is None:
            return set()
        adjacencies = State._get_stone_group_adjacencies(board_state, [(last_action.coord1, last_action.coord2)])
        dead_stones = set()
        for adj in adjacencies:
            if board_state[adj] == opposite_stone(last_action.stone_type).value:
                group, num_liberties = State._get_group(board_state, *adj)
                if num_liberties == 0:
                    dead_stones = dead_stones.union(group)
        return dead_stones


    @property
    def valid_actions(self):
        if self._valid_actions is None:
            self._valid_actions = self.generate_all_valid_actions()
        return self._valid_actions

    @property
    def can_add_handicap(self):
        return np.all(self.board_state != Stone.WHITE)
    

    def probability_of_observation(self, frame_index, observation_probability):
        board_size = self.board_state.shape[0]
        board_state = np.expand_dims(self.board_state, 2)
        assert observation_probability.shape == (board_size, board_size, 3)
        prob = np.take_along_axis(observation_probability, board_state, axis=2)
        assert prob.shape == (board_size, board_size,1)
        prob = np.nanprod(prob)
        self.observation_prob[frame_index] = prob
        return prob

    def posterior_prob(self, frame_index):
        assert frame_index in self.prior_prob
        assert frame_index in self.observation_prob
        return self.prior_prob[frame_index] * self.observation_prob[frame_index]

    
    def generate_all_valid_handicap_actions(self):
        if not self.can_add_handicap:
            return []
        
        handicap_actions = []
        board_size=self.board_state.shape[0]
        if board_size == 19:
            handicap_indices = [3,9,15]
        if board_size == 9:
            handicap_indices = [2,4,6]
        handicap_indices = np.stack(np.meshgrid(handicap_indices, handicap_indices), axis=2)
        assert handicap_indices.shape == (3,3,2)

        # eliminate actions that place handicap stone on existing handicap stone
        def valid_handicap_action(board_state, handicap_stones):
            for coord1, coord2 in handicap_stones:
                if board_state[coord1, coord2] != Stone.EMPTY.value:
                    return []
            action = Action(
                stone_type=Stone.BLACK,
                prob=1.0/16,
                add_handicap_stones = handicap_stones
            )
            return [action]

        A = handicap_indices[0,0,:]
        B = handicap_indices[2,2,:]
        C = handicap_indices[2,0,:]
        D = handicap_indices[0,2,:]
        E = handicap_indices[1,1,:]
        F = handicap_indices[0,1,:]
        G = handicap_indices[2,1,:]
        H = handicap_indices[1,0,:]
        I = handicap_indices[1,2,:]

        # 2 possibilities for 2 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B])
        handicap_actions += valid_handicap_action(self.board_state, [C, D])

        # 4 possibilities for 3 handicap 
        handicap_actions += valid_handicap_action(self.board_state,  [A, B, C])
        handicap_actions += valid_handicap_action(self.board_state, [A, B, D])
        handicap_actions += valid_handicap_action(self.board_state, [A, C, D])
        handicap_actions += valid_handicap_action(self.board_state, [B, C, D])

        # 1 possibilities for 4 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D])

        # 1 possibilities for 5 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, E])

        # 2 possibilities for 6 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, F, G])
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, H, I])

        # 2 possibilities for 7 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, E, F, G])
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, E, H, I])

        # 1 possibilities for 8 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, F, G, H, I])

        # 1 possibilities for 9 handicap 
        handicap_actions += valid_handicap_action(self.board_state, [A, B, C, D, E, F, G, H, I])
        return handicap_actions

    def generate_all_valid_non_handicap_actions(self):
        turn = opposite_stone(self.last_to_play)
        possible_actions = self.board_state == Stone.EMPTY.value
        if self.ko_location is not None:
            possible_actions[self.ko_location] = False
        possible_actions = np.transpose(np.nonzero(possible_actions))

        non_suicidal_possible_actions = []
        new_board_state = np.copy(self.board_state)
        for empty_point in possible_actions:
            new_board_state[empty_point[0], empty_point[1]] = turn.value
            _, liberties = self._get_group(new_board_state, empty_point[0], empty_point[1])
            if liberties > 0:
                non_suicidal_possible_actions.append(empty_point)
            new_board_state[empty_point] = Stone.EMPTY.value
        possible_actions = non_suicidal_possible_actions

        if len(possible_actions) > 0:
            prob = 1 / len(possible_actions)
            actions = [
                Action(coord1=empty_point[0], coord2=empty_point[1], stone_type=turn, prob=prob)
                for empty_point in possible_actions
            ]
            actions.append(Action(stone_type=turn, is_pass=True, prob = 1e-10))
        else:
            actions = [Action(stone_type=turn, is_pass=True, prob = 1)]
            
        actions.append(Action(stone_type=turn, no_action=True, prob=0.1))
        return actions

    def generate_all_valid_actions(self):
        return self.generate_all_valid_non_handicap_actions() + self.generate_all_valid_handicap_actions()

    
    def is_valid_action(self, action):
        if action in self.valid_actions:
            return True
        return False
    
    def generate_new_board_state(self, action):
        if action.no_action:
            return self
        if action.is_pass:
            return State(
                self.board_state,
                last_to_play=action.stone_type,
                black_captures = self.black_captures,
                white_captures = self.white_captures,
                ko_location=None,
                handicap_stones = self.handicap_stones,
            )
        new_board_state = np.copy(self.board_state)

        if len(action.add_handicap_stones) > 0:
            assert self.can_add_handicap
            for (coord1, coord2) in action.add_handicap_stones:
                new_board_state[coord1, coord2] = Stone.BLACK.value
            return State(
                new_board_state,
                last_to_play=action.stone_type,
                black_captures = self.black_captures,
                white_captures = self.white_captures,
                ko_location=None,
                handicap_stones=self.handicap_stones + action.add_handicap_stones,
            )
            
            
        new_board_state[action.coord1, action.coord2] = action.stone_type.value
        dead_stones = self.dead_stones(new_board_state, action)
        if len(dead_stones) > 0:
            for dead_stone in dead_stones:
                new_board_state[dead_stone[0], dead_stone[1]] = Stone.EMPTY.value
        ko_location = None
        if len(dead_stones) == 1:
            group, num_liberties = self._get_group(new_board_state, action.coord1, action.coord2)
            if len(group) == 1 and num_liberties == 1:
                ko_location = dead_stones[0]                
        if action.stone_type == Stone.BLACK:
            black_captures = self.black_captures + len(dead_stones)
            white_captures = self.white_captures
        if action.stone_type == Stone.WHITE:
            white_captures = self.white_captures + len(dead_stones)
            black_captures = self.black_captures 
        return State(
            new_board_state,
            last_to_play = action.stone_type,
            black_captures = black_captures,
            white_captures = white_captures,
            handicap_stones=self.handicap_stones,
            ko_location=ko_location
        )
        
    def generate_non_handicap_actions_toward_keyframe(self, target_board_state):
        turn = opposite_stone(self.last_to_play)
        need_stone = np.logical_and(target_board_state == turn.value, self.board_state == Stone.EMPTY.value)
        need_stone = np.stack(np.nonzero(need_stone),axis=1)
        assert need_stone.shape[1] == 2
        possible_pts = {tuple(pt) for pt in need_stone}
        
        remove_stone = np.logical_and(target_board_state == Stone.EMPTY.value, self.board_state == turn.value)
        remove_stone = np.stack(np.nonzero(remove_stone),axis=1)
        assert remove_stone.shape[1] == 2
        for coord1, coord2 in remove_stone:
            group,_ = self._get_group(self.board_state, coord1, coord2)
            assert group is not None
            adjacencies = self._get_stone_group_adjacencies(self.board_state, group)
            possible_pts = possible_pts.union({tuple(pt) for pt in adjacencies if target_board_state[pt] != self.last_to_play.value})

        if len(possible_pts) > 0:
            prob = 1.0/len(possible_pts)
            possible_actions = {
                Action(
                    coord1=coord1,
                    coord2=coord2,
                    stone_type=turn,
                prob=prob
                )
                for coord1, coord2 in possible_pts
            }
            return list(possible_actions)
        else:
            return []

    def generate_non_handicap_actions_toward_non_keyframe(self, board_prob1, board_prob2):
        turn = opposite_stone(self.last_to_play)
        board_state0 = self.board_state

        board_state0 == Stone.EMPTY,  
        
        
    def filter_impossible_actions(self, observation_probabilities, threshold, two_moves=False):
        # turn = opposite_stone(self.last_to_play)

        # handicap_actions = self.generate_all_valid_handicap_actions()

        # keyframe_board_state = np.argmax(observation_probabilities[-1], axis=2)
        # keyframe_actions = self.generate_non_handicap_actions_toward(target_board_state=keyframe_board_state)
        
        # other_actions = [
        #     # Action(stone_type=turn, is_pass=True, prob = 1e-10),
        #     Action(stone_type=turn, no_action=True, prob=0.1)
        # ]
        # return handicap_actions + keyframe_actions + other_actions
        
        possible_actions = []
        for action in self.valid_actions:
            if action.coord1 is not None and action.coord2 is not None:
                prob = observation_probabilities[0][action.coord1, action.coord2, action.stone_type.value]
                if not np.isnan(prob) and prob > threshold:
                    possible_actions.append(action)
                elif np.isnan(prob) and not two_moves:
                    # if board is obstructed look a move ahead
                    prob = observation_probabilities[1][action.coord1, action.coord2, action.stone_type.value]
                    if not np.isnan(prob) and prob > threshold:
                        possible_actions.append(action)
                    elif not np.isnan(prob) and prob < threshold:
                        # check if was immediately captured
                        test_board_state = np.copy(self.board_state)
                        test_board_state[action.coord1, action.coord2] = action.stone_type.value
                        group, libs = self._get_group(test_board_state, action.coord1, action.coord2)
                        if libs == 1:
                            possible_actions.append(action)
                    elif np.isnan(prob):
                        possible_actions.append(action)
                elif two_moves:
                    possible_actions.append(action)
            elif len(action.add_handicap_stones) > 0:
                assert self.can_add_handicap
                assert action.stone_type == Stone.BLACK
                prob = [
                    observation_probabilities[0][coord1, coord2, action.stone_type.value]
                    for coord1, coord2 in action.add_handicap_stones
                ]
                prob = [x > threshold for x in prob if not np.isnan(x)]
                if np.all(prob):
                    possible_actions.append(action)
            elif action.is_pass:
                pass
                # possible_actions.append(action)
            elif action.no_action:
                possible_actions.append(action)
            else:
                raise NotImplementedError
        return possible_actions
            
            

                        


@dataclass
class Edge:
    src_index: int
    dst_index: int
    src_frame_index: int
    dst_frame_index: int
    action: Action
    def assert_valid(self):
        assert src_frame_index + 1 == dst_frame_index or src_frame_index == dst_frame_index
    def __repr__(self):
        return f"{self.src_index}@{self.src_frame_index} -> {self.dst_index}@{self.dst_frame_index}"
    def __str__(self):
        return self.__repr__()

class IndexedStateList:
    def __init__(self):
        self.state_list = []
        self.state_lookup = {}
    def __getitem__(self, index):
        return self.state_list[index]
    def get_index(self, state):
        return self.state_lookup[state]
    def add_state_if_not_exists(self, state):
        if state in self.state_lookup:
            index = self.state_lookup[state]
        else:
            index = len(self.state_list)
            self.state_lookup[state] = index
            self.state_list.append(state)
        return index
    def subset(self, keep_indices_set):
        new_indices_to_old_indices = sorted(set(keep_indices_set))
        old_indices_to_new_indices = {old_index: new_index for new_index, old_index in enumerate(new_indices_to_old_indices)}
        self.state_list = [self.state_list[old_index] for old_index in new_indices_to_old_indices]
        self.state_lookup = {state: index for index, state in enumerate(self.state_list)}
        return old_indices_to_new_indices

class EdgeSequence:
    def __init__(self):
        self.edges = []        
    def get_src_state_index(self, src_frame_index=None):
        if len(self.edges) == 0:
            return None
        if src_frame_index is None:
            return self.edges[0].src_index
        else:
            for edge in self.edges:
                if edge.src_frame_index== src_frame_index:
                    return edge.src_index
            return None
    def get_dst_state_index(self, dst_frame_index=None):
        if len(self.edges) == 0:
            return None
        if dst_frame_index is None:
            return self.edges[-1].dst_index
        else:
            for edge in reversed(self.edges):
                if edge.dst_frame_index== dst_frame_index:
                    return edge.dst_index
            return None
    def get_src_frame_index(self):
        if len(self.edges) > 0:
            return self.edges[0].src_frame_index
        else:
            return None
    def get_dst_frame_index(self):
        if len(self.edges) > 0:
            return self.edges[-1].dst_frame_index
        else:
            return None
    def append(self, edge):
        if len(self.edges) > 0:
            assert self.get_dst_state_index() == edge.src_index, f"{self.get_dst_state_index()} == {edge.src_index}"
            assert self.get_dst_frame_index() == edge.src_frame_index
        self.edges.append(edge)
    def assert_valid(self):
        if len(self.edges) > 0:
            for i in range(len(self.edges)-1):
                assert self.edges[i].dst_index == self.edges[i+1].src_index 
                assert self.edges[i].dst_frame_index == self.edges[i+1].src_frame_index
    def remap_state_indices(self, old_indices_to_new_indices):
        for edge in self.edges:
            edge.src_index = old_indices_to_new_indices[edge.src_index]
            edge.dst_index = old_indices_to_new_indices[edge.dst_index]
    def get_state_indices_set(self):
        state_indices = {edge.dst_index for edge in self.edges}
        state_indices.add(self.edges[0].src_index)
        return state_indices
    def connect(self, other):
        if len(self.edges) > 0 and len(other.edges) > 0:
            assert self.get_dst_state_index() == other.get_src_state_index()
            assert self.get_dst_frame_index() == other.get_src_frame_index()
            for edge in other.edges:
                self.append(edge)
        elif len(self.edges) == 0:
            self.edges = other.edges
        
    def __repr__(self):
        if len(self.edges) > 0:
            states = [f"{edge.src_index}@{edge.src_frame_index}" for edge in self.edges]
            states.append(f"{self.edges[-1].dst_index}@{self.edges[-1].dst_frame_index}")
            return " -> ".join(states)
        else:
            return ""
    def __str__(self):
        return self.__repr__()
    def is_empty(self):
        return len(self.edges) == 0
    
class GameSolver:
    def __init__(self, observation_probabilities, board_size = 19, observation_probability_threshold=0.5, num_candidates=16):
        self.observation_probabilities = np.array([observation_probabilities[i] for i in observation_probabilities])
        self.observation_probability_threshold=observation_probability_threshold
        self.num_candidates = num_candidates

        start_board_state = np.full([board_size, board_size], fill_value = Stone.EMPTY.value, dtype=np.uint8)
        self.start_state = State(
            start_board_state,
            last_to_play=Stone.WHITE,
            black_captures=0,
            white_captures=0,
            ko_location=None,
            handicap_stones=[],
        )
        self.start_state.observation_prob[-1] = 1
        self.start_state.prior_prob[-1] = 1

        self.state_list = IndexedStateList()
        self.state_list.add_state_if_not_exists(self.start_state)
        self.game_sequence = {}
        self.edges_lookup = defaultdict(lambda:[])

        self.keyframes = {-1: True}
        self.keyframe_estimate = {-1: self.start_state.board_state}
        self.next_keyframe = {-1: -1}
        self.num_frames = len(observation_probabilities)
        for frame_index in range(self.num_frames):
            observation_probabilities = self.observation_probabilities[frame_index]
            if np.any(np.isnan(observation_probabilities)):
                self.keyframes[frame_index] = False
                continue
            board_state_estimate = np.argmax(observation_probabilities, axis=2)
            prob = np.take_along_axis(observation_probabilities, np.expand_dims(board_state_estimate,2), axis=2)
            obvious = np.all(prob > self.observation_probability_threshold)
            not_start = np.any(board_state_estimate != Stone.EMPTY.value)
            if frame_index == self.num_frames-1:
                self.keyframes[frame_index] = True
            else:
                self.keyframes[frame_index] = obvious # and not_start
            if self.keyframes[frame_index]:
                self.keyframe_estimate[frame_index] = board_state_estimate.astype(np.uint8)

        last_keyframe = self.num_frames-1
        for frame_index in reversed(range(self.num_frames)):
            if self.is_keyframe(frame_index):
                last_keyframe = frame_index
            self.next_keyframe[frame_index] = last_keyframe
            
                
        temp = np.array([frame_index - np.sum(v != Stone.EMPTY.value) for frame_index,v in self.keyframe_estimate.items()])
        temp2 = np.concat([np.array([0]),temp]) - np.concat([temp,np.array([0])])
        for i,frame_index in enumerate(sorted(self.keyframe_estimate.keys())):
            print(frame_index, np.sum(self.keyframe_estimate[frame_index] != Stone.EMPTY.value).item(), temp[i], temp2[i])

    def clear_memory(self):
        state_indices = [
            edge_sequence.get_state_indices_set()
            for fame_index, edge_sequence in self.game_sequence.items() if edge_sequence is not None
        ]
        keep_indices_set = set().union(*state_indices)
        old_indices_to_new_indices = self.state_list.subset(keep_indices_set=keep_indices_set)
        for frame_index, edge_sequence in self.game_sequence.items():
            if edge_sequence is not None:
                edge_sequence.remap_state_indices(old_indices_to_new_indices=old_indices_to_new_indices)
       

    def is_keyframe(self, frame_index):
        return self.keyframes[frame_index]
    
    def matches_keyframe(self, state_index, keyframe_index):
        assert self.is_keyframe(keyframe_index)
        board_state_estimate = self.keyframe_estimate[keyframe_index]
        return np.all(self.state_list[state_index].board_state == board_state_estimate)
    
    def add_or_update_state_index(self, src_index, src_frame_index, action, dst_frame_index):
        new_state = self.state_list[src_index].generate_new_board_state(action)
        dst_index = self.state_list.add_state_if_not_exists(new_state)
        
        # update priors
        if dst_frame_index not in self.state_list[dst_index].prior_prob:
            self.state_list[dst_index].prior_prob[dst_frame_index] = 0
        additional_prior_prob = self.state_list[src_index].prior_prob[src_frame_index] * action.prob
        self.state_list[dst_index].prior_prob[dst_frame_index] += additional_prior_prob
        
        return dst_index

    def generate_descendant_states(self, src_index, src_frame_index, dst_frame_index):
        dst_keyframe_index = self.next_keyframe[dst_frame_index]
        
        possible_actions = self.state_list[src_index].filter_impossible_actions(
            self.observation_probabilities[dst_frame_index:(dst_keyframe_index+1)], self.observation_probability_threshold, two_moves=src_frame_index==dst_frame_index
        )
        descendant_indices = {
            action: self.add_or_update_state_index(src_index, src_frame_index, action, dst_frame_index)
            for action in possible_actions
        }
        for action, dst_index in descendant_indices.items():
            if dst_frame_index not in self.state_list[dst_index].observation_prob:
                self.state_list[dst_index].probability_of_observation(dst_frame_index, self.observation_probabilities[dst_frame_index])
            assert src_frame_index <= dst_frame_index
            edge = Edge(src_index=src_index, dst_index=dst_index, action=action, src_frame_index=src_frame_index, dst_frame_index=dst_frame_index)
            self.edges_lookup[dst_index].append(edge)

        descendant_indices = {v for k,v in descendant_indices.items()}
        return descendant_indices
    
    def create_possible_states(self, start_state_indices, start_frame_index, end_frame_index, additional_moves=0, debug=False):
        candidate_state_indices = start_state_indices
        for frame_index in range(start_frame_index+1, end_frame_index+1):
            new_descendant_state_indices = [
                self.generate_descendant_states(
                    state_index,
                    src_frame_index=frame_index-1,
                    dst_frame_index=frame_index
                )
                for state_index in candidate_state_indices
            ]
            descendant_state_indices = set().union(*new_descendant_state_indices)

            if additional_moves > 0 and frame_index == end_frame_index:
                new_descendant_state_indices = descendant_state_indices
                for i in range(additional_moves):
                    out = [
                        self.generate_descendant_states(
                            state_index,
                            src_frame_index=frame_index,
                            dst_frame_index=frame_index
                        )
                        for state_index in new_descendant_state_indices
                    ]
                    new_descendant_state_indices = new_descendant_state_indices.union(*out)
                    descendant_state_indices.union(new_descendant_state_indices)
                
            # prune large number of descendants into fewer candidates
            descendant_posteriors = {idx: self.state_list[idx].posterior_prob(frame_index) for idx in descendant_state_indices}
            candidate_state_indices = [k for k, v in sorted(descendant_posteriors.items(), key=lambda item: -item[1])][:self.num_candidates]
            if debug:                
                debug_img = self.display_candidate_images(candidate_state_indices, frame_index)
                cv2.imshow("original", self.debug_imgs[frame_index])
                cv2.imshow("candidates", debug_img)
                stone_placements = board_estimation.BoardEstimator.instantaneous_probabilities_to_placements(self.observation_probabilities[frame_index])
                instantaneous_estimate = self.debug_draw_board(stone_placements, self.observation_probabilities[frame_index])
                cv2.imshow("instantaneous", instantaneous_estimate)
                cv2.waitKey(0)

        return candidate_state_indices
        
    def find_best_path(self, start_frame_index, start_state_index, end_frame_index, end_state_index):
        if start_frame_index > end_frame_index:
            return None, 0
        if start_frame_index == end_frame_index and start_state_index == end_state_index:
            score = self.state_list[end_state_index].posterior_prob(end_frame_index)
            return EdgeSequence(), score

        best_path = None
        best_score = 0
        for edge in self.edges_lookup[end_state_index]:            
            if edge.dst_frame_index != end_frame_index:
                continue
            if edge.src_frame_index == edge.dst_frame_index and edge.src_index == edge.dst_index:
                continue
            print(edge, "end_frame_index", end_frame_index, "end_state_index", end_state_index)
            assert edge.src_frame_index <= edge.dst_frame_index
            assert edge.src_frame_index <= end_frame_index
            path, score = self.find_best_path(
                start_frame_index=start_frame_index,
                start_state_index=start_state_index,                
                end_frame_index=edge.src_frame_index,
                end_state_index=edge.src_index
            )
            if path is not None and score >= best_score:
                best_score = score
                best_path = path
                best_path.append(edge)

        if best_path is not None:
            assert not best_path.is_empty()
            best_path.assert_valid()
        score = self.state_list[end_state_index].posterior_prob(end_frame_index)
        return best_path, score


    def _solve_partial_graph(self, start_frame_index, end_frame_index, additional_moves=0, debug=False):
        if debug:
            print("solving partial graph: ", start_frame_index, end_frame_index)

        assert self.is_keyframe(start_frame_index)
        assert self.is_keyframe(end_frame_index)

        if start_frame_index != -1:
            start_board_state = self.keyframe_estimate[start_frame_index]
            if self.game_sequence[start_frame_index] is not None:
                start_state_indices = [self.game_sequence[start_frame_index].get_dst_state_index()]
            else:
                start_states = [
                    State(
                        start_board_state,
                        last_to_play=Stone.WHITE
                    ),
                    
                    State(
                        start_board_state,
                        last_to_play=Stone.BLACK
                    )
                ]           
                start_state_indices = [self.state_list.add_state_if_not_exists(state) for state in start_states]
                for start_state_index in start_state_indices:
                    self.state_list[start_state_index].observation_prob[start_frame_index] = 1
                    self.state_list[start_state_index].prior_prob[start_frame_index] = 1
        else:
            start_state_indices = [0]
        
        end_state_candidate_indices = self.create_possible_states(
            start_state_indices=start_state_indices,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            additional_moves=additional_moves,
            debug=debug
        )

        best_path = None
        best_score = 0
        for end_state_candidate_index in end_state_candidate_indices:
            if not self.matches_keyframe(end_state_candidate_index, end_frame_index):
                continue
            for start_state_index in start_state_indices:
                path, score = self.find_best_path(
                    start_frame_index=start_frame_index,
                    start_state_index=start_state_index,
                    end_frame_index=end_frame_index,
                    end_state_index=end_state_candidate_index
                )
                if path is not None and score >= best_score:
                    best_score = score
                    best_path = path
        if best_path is not None:
            best_path.assert_valid()
        return best_path

    def solve_partial_graph(self, start_frame_index, end_frame_index, debug=False):
        best_sequence = self._solve_partial_graph(
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            debug=debug
        )
        if debug:
            print("0 extra move success", best_sequence is not None)
        if best_sequence is None:
            best_sequence = self._solve_partial_graph(
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                additional_moves=2,
                debug=debug
            )
            if debug:
                print("2 extra move success", best_sequence is not None)

        self.game_sequence[end_frame_index] = best_sequence
        self.clear_memory()
        return best_sequence is not None
            
    
    def solve(self, debug=False):
        keyframes = {
            frame_index for frame_index in range(self.num_frames)
            if self.is_keyframe(frame_index)
        }
        keyframes.add(-1)
        keyframes.add(self.num_frames-1)
        keyframes = sorted(keyframes)
        if debug:
            print(keyframes)
        
        for i in range(len(keyframes)-1):
            start_frame_index = keyframes[i]
            end_frame_index = keyframes[i+1]
            success = self.solve_partial_graph(
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                debug = debug
            )


    def to_sgf(self):

        def player_string(action):
            if action.stone_type == Stone.BLACK:
                return "B"
            if action.stone_type == Stone.WHITE:
                return "W"
            return NotImplementedError
        def coord_letter(x):
            return "abcdefghijklmnopqrstuvwxyz"[x]
        action_list = [self.game_sequence[index-1].action for index in range(len(self.game_sequence))]
        handicap_stones = [action.add_handicap_stones for action in action_list if len(action.add_handicap_stones) > 0]
        handicap_string = [f"[{coord_letter(x[0])}{coord_letter(x[1])}]" for x_action in handicap_stones for x in x_action]
        move_list = [f"{player_string(action)}{coord_letter(action.coord1)}{coord_letter(action.coord2)}"
                     for action in action_list if action.coord1 is not None and action.coord2 is not None]
        move_string = "\n(;".join(move_list)+''.join([")"]*(len(move_list)))

        header = [
            "(FF[4]",
            "CA[UTF-8]",
            "GM[1]",
        ]
        if len(handicap_string) > 0:
            header.append(f"AB{''.join(handicap_string)}")

        header = '\n'.join(header)
        sgf = f"{header}\n{move_string}"
        return sgf

    def display_candidate_images(self, candidate_state_indices, frame_index):
        assert len(candidate_state_indices) > 0
        grid_size=int(np.sqrt(self.num_candidates))
        candidate_images = []
        for i, index in enumerate(candidate_state_indices):
            board_state = self.state_list[index].board_state
            # board_estimator.draw_board(camera, lattice, board_state)
            img = self.debug_draw_board(board_state, self.observation_probabilities[frame_index])
            score1 = np.log(self.state_list[index].observation_prob[frame_index])
            score2 = np.log(self.state_list[index].prior_prob[frame_index])
            img = cv2.putText(img, f"obs:{str(round(score1,4))}" ,(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
            img = cv2.putText(img, f"prior:{str(round(score2,4))}" ,(250,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
            candidate_images.append(img)
        if len(candidate_images) < grid_size**2:
            candidate_images += [np.zeros_like(candidate_images[0])]*(grid_size**2-len(candidate_images))

        h = candidate_images[0].shape[0]
        w = candidate_images[0].shape[1]
        candidate_images = np.reshape(np.stack(candidate_images), [grid_size,grid_size,h, w, 3])
        candidate_images = np.transpose(candidate_images, [0,2,1,3,4])
        candidate_images = np.reshape(candidate_images, [grid_size*h, grid_size*w, 3])
        return candidate_images


if __name__ == '__main__':
    import argparse
    import image_utils
    import lattice_estimation
    import board_estimation
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--board_size', type=int, default=19)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened()
    test_camera = image_utils.Camera(
        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frames-1)
    _,test_img = cap.read()
    assert test_img is not None

    lattice = lattice_estimation.estimate_video_lattice(
        board_size=args.board_size,
        camera=test_camera,
        img_generator=image_utils.video_generator(args.video_path),
        num_x_bins=3,
        num_y_bins=3,
        num_z_bins=3,
        num_x_rot_bins=3,
        num_y_rot_bins=3,
        num_z_rot_bins=3,
        z_rot_search_range_degrees = 2,
        z_search_range_mm = 10,
        x_search_range_mm = 25,
        y_search_range_mm = 25,
    )
    
    board_estimator=board_estimation.BoardEstimator(
        board_size=args.board_size,
        circle_sampling_k=8,
        circle_sampling_r=0.25,
    )
    board_estimator.estimate_parameters(
        imgs = image_utils.video_generator(args.video_path),
        lattices=[lattice]*num_frames,
        camera=test_camera,
        max_imgs=num_frames//4
    )
    

    # debug_board_state = np.array(
    #     [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1],
    #      [1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,0,1,1],
    #      [1,1,1,2,1,2,1,2,2,1,2,1,1,1,1,0,1,1,1],
    #      [1,1,2,1,1,1,0,0,0,2,2,1,1,1,1,1,1,1,1],
    #      [1,1,0,0,1,0,1,1,0,0,2,0,0,1,1,1,0,0,1],
    #      [1,1,1,1,1,1,1,0,2,2,0,2,1,1,0,1,1,2,1],
    #      [1,1,1,1,1,1,1,1,1,1,0,2,1,1,1,1,2,1,1],
    #      [1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1],
    #      [1,2,0,0,1,1,1,2,1,2,1,2,1,1,1,1,1,1,1],
    #      [2,1,2,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],
    #      [1,0,2,0,0,1,1,1,1,1,1,1,1,1,1,2,2,1,1],
    #      [1,0,2,0,2,0,1,1,1,1,1,1,1,1,1,2,0,1,1],
    #      [1,2,0,0,2,2,0,0,1,0,1,1,1,1,1,0,0,1,1],
    #      [2,2,0,1,0,1,0,2,0,1,1,1,0,1,1,1,1,1,1],
    #      [0,0,1,0,2,2,0,2,0,2,1,1,1,1,1,0,1,1,1],
    #      [1,2,0,2,0,2,1,2,2,1,2,1,0,1,1,1,0,1,1],
    #      [1,1,1,2,0,2,1,1,1,1,1,1,2,1,2,1,1,1,1],
    #      [1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    # ).astype(np.uint8)
    # debug_state = State(board_state=debug_board_state, last_to_play=Stone.BLACK)
    # debug_img = board_estimator.draw_board(test_camera, lattice, stone_placements=debug_state.board_state)
    # cv2.imshow("debug", debug_img)
    # cv2.waitKey(0)
    
    # print("dead stones", debug_state.dead_stones(debug_state.board_state, last_action=Action(coord1=18, coord2=4, stone_type=Stone.WHITE, prob=1)))


    
    observation_probabilities = {}
    for frame_index, img in enumerate(image_utils.video_generator(args.video_path)):
        detected_lines = image_utils.find_lines_in_img(img, lattice_size=args.board_size, debug=False)
        filtered_det_lines_img = lattice.filtered_detected_lines_img_by_camera_extrinsics(
            test_camera, lattice.initial_camera_extrinsics, detected_lines
        )
        obstruction_map = lattice.estimate_obstruction_map(test_camera, filtered_det_lines_img)
        observation_probabilities_i = board_estimator.get_instantaneous_board_state_probabilities(
            lattice=lattice,
            camera=test_camera,
            img=img,
            obstruction_map=obstruction_map
        )
        observation_probabilities[frame_index] = observation_probabilities_i

    
    graph = GameSolver(observation_probabilities, board_size = args.board_size, observation_probability_threshold=0.6, num_candidates=16)
    graph.debug_imgs = {frame_index: img for frame_index, img in enumerate(image_utils.video_generator(args.video_path))}
    graph.debug_draw_board = lambda board_state, board_prob: board_estimator.draw_board(test_camera, lattice, stone_placements=board_state, stone_probabilities=board_prob)
    graph.solve(debug=True)
