from support_functions import *
import numpy as np
import dill
import itertools
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import copy
from pathlib import Path
import subprocess
from datetime import datetime
import configparser

config = configparser.RawConfigParser() 
config.read_file(open(r'q_learning.config'))

alpha = config.get('train.qlearning', 'alpha')
gamma = config.get('train.qlearning', 'gamma')
epsilon = config.get('train.qlearning', 'epsilon')
z = config.get('train.qlearning', 'z')
n = config.get('board_size', 'n')
tree_search_file_name = config.get('tree_search', 'file_name')
trained_player = config.get('train.qlearning', 'file_name')


start = datetime.now()

# Force the INFO messages to be printed to the console
logging.basicConfig(level=logging.DEBUG)

# Loading our tree opponent from the first article
logging.info('Checking if pkl files are generated...')
if not Path(tree_search_file_name).exists():
    subprocess.call('generate_tree.py')
else:
    logging.info(f'{tree_search_file_name} found...')

logging.info('Loading tree...')
with open(tree_search_file_name, 'rb') as f:
    tree = dill.load(f)

# Using memoization to speed up the tree
logging.info('Precomputing tree moves...')
precompute_tree_moves(tree)
# Create the board and player
tictactoe = Tictoe(int(n))
player_tree = Player(1, tree, alpha = float(alpha),
                            gamma = float(gamma),
                            epsilon = float(epsilon)) 

logging.info('Starting the training loop...')
no_episodes = int(z)
for ep_idx in tqdm(range(no_episodes)):
    while not tictactoe.is_endstate():
        tictactoe = player_tree.make_move(tictactoe)
        tictactoe = player_tree.make_computer_move(tictactoe)
        player_tree.update_qtable()
    tictactoe.reset_board()

logging.info('Saving the model...')
with open(trained_player, 'wb') as f:
    dill.dump(player_tree, f)

logging.info('Training finished...')
logging.info('done (%s)...' % (datetime.now() - start))