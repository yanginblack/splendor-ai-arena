# Splendor-AI-Arena

> This project serves as the final project for CS221, creating an environment where AI agents, developed using various algorithms, can play the game Splendor against each other.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [About](#about)
- [Code Structure](#code-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Game Overview](#game-overview)
  - [Game.py](#gamepy)
    - [Main Functions](#main-functions)
    - [State Definition](#state-definition)
    - [Action Definition](#action-definition)
    - [Rule Modifications](#rule-modifications)
    - [Discarding Gems](#discarding-gems)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About

Splendor-AI-Arena provides a simulation environment for developing and testing AI agents playing the board game Splendor. It allows agents developed using various algorithms to compete, enabling analysis and comparison of different AI strategies.

## Code Structure

- **BaseGame.py**: A template for game environments, defining the required API. New games should inherit from this class and implement the necessary methods to ensure compatibility with the training system.
- **play.py**: The entry point for starting games. It defines the players, number of games, display settings, and other configurations.
- **GameConfig.py**: Central configuration file for specifying which game to play.
- **PlayGround.py**: Manages game execution, handling turns and recording results.
- **Game Folder**: Contains game-specific logic, including:
- **GamePlayer.py**: Defines player behavior.
- **Game.py**: Implements the core game logic.

## Getting Started

### Prerequisites
- Python

### Play local

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/splendor-ai-arena.git
2. Enter the repo
   ```bash
   cd splendor-ai-arena
4. Play the game
   ```bash
   python play.py


## Game Overview
### Game.py
Game.py is the game environment, define the state dimentions and data structure, action data structure, initiate state, providing functions to interact with state, and provide display function to output human readable prints of the state.
####Main Functions
- getNextState: Takes the current player ID and state, executes the action, and returns the next state and player ID.
- getValidMoves: Takes the current player ID and state, returning a list indicating valid actions (1 for valid, 0 for invalid).
- canonicalForm: Transforms the state into a canonical form relative to the current player, simplifying training and gameplay.
- translateAction: Converts actions from the canonical form back to the specific player's perspective.
- display: Displays the state in a human-readable format.
#### State Defination
State(len = 300): 
- (0 - 131) 12 cards read to be purchased, 4 cards for level1, 4 cards for level2, 4 cards for level3. Each card has 11 digits: first 5 are the required gems of the card, in the order of white, red, green, blue and brown. The next 5 represent the number of gems this card provides. e.g. Card provide a red gem, it should be [0 1 0 0 0].The last digit is the card's points.
e.g. [0 1 0 2 2 0 0 0 0 0 1 0] is the card required 1 red gem, 2 blue gems, 2 brown gems. Provides 1 brown gem. No points.
- (132 - 155) 4 nobles, 6 digits for 1 noble. e.g. [1 1 1 1 1 3] is a noble that requires 1 of each gem, provides 3 points.
- (156 - 161) public remaining gems: white, red, green, blue, brown and gold.
- (162 - 207) player1 info 46 digits
- 162 - 167: player 1 gems: white, red, green, blue, brown and gold.
- 168 - 172: player 1 permanent gems: white, red, green, blue, and brown.
- 173: player 1 points
- 174 - 206: player 1 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
- 207: player 1 acquired cards.
- 208 - 253: player 2 info
- 208 - 213: player 2 gems: white, red, green, blue, brown and gold.
- 214 - 218: player 2 permanent gems: white, red, green, blue, and brown.
- 219: player 2 points
- 220 - 252: player 2 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
- 253: player 2 acquired cards.
- 254 - 299: player 3 info
- 254 - 259: player 3 gems: white, red, green, blue, brown and gold.
- 260 - 264: player 3 permanent gems: white, red, green, blue, and brown.
- 265: player 3 points
- 266 - 298: player 3 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
- 299: player 3 acquired cards.
#### Action Defination
Action (len = 48):    
- 0 - 14: 12+3 cards purchase, 4 cards per row, row 0 is level 1, row 1 is level 2, row 2 is level 3. (0-11), 3 reserved card purchase(12-14), 
- 15 - 26: 12 cards reserve, rows can columns are the same as cards purchase, (15-26), 5 taking two gems of (white, red, green, blue and brown) (27-31), 
- 27 - 41: C53 to take 3 in different color(32-36), 5 for discarding gem (white, red, green, blue and brown) (37-41), 1 for pass (42)
- 42 - 47: for discarding gems when the any player has more than 10 gems in their round, all other players only have 1 action: pass, and that player only have five discarding actions until this player has 10 gems. 
#### Rule Modification
If a player meets the requirements for multiple nobles in a turn, they automatically acquire all eligible nobles instead of selecting one.
#### Discarding Gems
When any player has more than 10 gems, the game will hold immediately, all the other players only have one valid action: 47 (Holding), and the player with more than 10 gems have five valid actions (42, 43, 44, 45, 56) (only valid if they have this type of gem) to discard. 
The main place for agents to play the game. Input to each agent is the play method with state as input, output should be action (number inside action space)
Since Splendor has three different positions, the algroithm in Player.py can focus on the first player actions. All the state input to player play function will be canonical, which is the transformed state that treating current player as first player, which can allow us train one AI and let it play all three positions.
