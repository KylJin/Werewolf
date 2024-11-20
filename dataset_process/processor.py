import os
import json
from tqdm import tqdm
import numpy as np
from utils import get_embeddings

SPEAKING_STRATEGY = {
    "honest_evidence": 0,
    "deceptive_evidence": 1,
    "honest_accusation": 2,
    "deceptive_accusation": 3,
    "honest_defense": 4,
    "deceptive_defense": 5
}
TEAM_VILLAGE = ["Insomniac", "Robber", "Seer", "Troublemaker", "Villager"]
TEAM_WEREWOLF = ["Werewolf"]


class DataProcessor(object):
    def __init__(self, embedding_model="gemini"):
        self.embedding_model = embedding_model
        # all transitions
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        # one game history
        self.messages = None
        self.game_info = None
        self.day_start_idx = 0  # include
        self.day_end_idx = 0  # not include
    
    def process_dataset(self, dir_path):
        """
        Process all game histories in dir_path
        args:
            dir_path: the path of all game histories
        """
        self.clear()
        for file_name in tqdm(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            self.process_history(file_path)
    
    def process_history(self, file_path):
        """
        Process one game history.
        args:
            file_path: the path of the game history
        """
        with open(file_path, mode='r') as f:
            history = json.load(f)
        self.messages, self.game_info = history["messages"], history["evaluation"]
        
        # find the day phase start and end index
        for idx, message in enumerate(self.messages):
            if message["agent_name"] == "Moderator":
                if "Night phase ends." in message["content"]:
                    self.day_start_idx = idx + 1
                if "Day phase ends." in message["content"]:
                    self.day_end_idx = idx
                    break
        
        # get all player names
        player_names = list(self.game_info["roles_ground_truth"].keys())
        # gather all players' trajectory
        for player in player_names:
            obs, acts, rews, terms, _ = self._get_transitions(player)
            self.observations.append(obs)
            self.actions.append(acts)
            self.rewards.append(rews)
            self.terminals.append(terms)
    
    def _get_transitions(self, player):
        """
        Extract the transitions of player's trajectory in given game.
        args:
            player: the name of player
        returns:
            observations, actions, rewards, terminals, original_rewards (all numpy.array)
        """
        # get visible messages in night phase
        visible_messages = ""
        for message in self.messages[:self.day_start_idx]:
            if message["visible_to"] == "all" or player in message["visible_to"]:
                visible_messages = f"{visible_messages}\n[{message['agent_name']}]: {message['content']}"
        
        # extract observations and actions
        observations = []
        actions = []
        for message in self.messages[self.day_start_idx: self.day_end_idx]:
            if message["agent_name"] == player:
                obs = f"<Game history>:{visible_messages}\n<My thought and belief>: {message['belief']}".strip()
                observations.append(get_embeddings(obs, backend=self.embedding_model))
                actions.append(SPEAKING_STRATEGY[message["strategy"]])
            visible_messages = f"{visible_messages}\n[{message['agent_name']}]: {message['content']}"
        
        observations = np.array(observations)
        actions = np.array(actions)
        # get rewards
        rew, ori_rew = self._calc_reward(player)
        rewards = rew * np.ones(actions.shape[0])
        ori_rewards = ori_rew * np.ones(actions.shape[0])
        # set terminals
        terminals = np.zeros(actions.shape[0])
        terminals[-1] = 1

        return observations, actions, rewards, terminals, ori_rewards
    
    def _calc_reward(self, player):
        """
        Calculate the final reward for player in given game.
        args:
            player_name: the name of player
        returns:
            adjusted reward, original reward
        """
        winner = self.game_info["winner"]
        roles_ground_truth = self.game_info["roles_ground_truth"]
        num_player = float(len(roles_ground_truth))
        voting = self.game_info["voting_result"]
        
        if winner == "Draw":
            return -voting[player] / num_player, 0
        elif (roles_ground_truth[player] in TEAM_VILLAGE and "Village" in winner) or \
            (roles_ground_truth[player] in TEAM_WEREWOLF and "Werewolf" in winner):  # win the game
            return 1 - voting[player] / num_player, 1
        else:  # lose the game
            return -1 - voting[player] / num_player, -1
    
    def clear(self):
        """
        Clear data caches in the buffer.
        """
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminals.clear()
    
    def get_dataset(self):
        """
        Get transitions in the buffer.
        """
        observations = np.concatenate(self.observations)
        actions = np.concatenate(self.actions)
        rewards = np.concatenate(self.rewards)
        terminals = np.concatenate(self.terminals)
        return observations, actions, rewards, terminals
    
    def save_dataset(self, save_path):
        """
        Save transitions in the buffer in a d3rlpy way.
        args:
            save_path: the path of saved dataset
        """
        observations = np.concatenate(self.observations)
        actions = np.concatenate(self.actions)
        rewards = np.concatenate(self.rewards)
        terminals = np.concatenate(self.terminals)

        import d3rlpy
        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )

        with open(save_path, "w+b") as f:
            dataset.dump(f)


if __name__ == "__main__":
    processor = DataProcessor(embedding_model="openai")
    processor.process_dataset(dir_path="./gpt4_dataset")
    processor.save_dataset(save_path="./processed_dataset.h5")
