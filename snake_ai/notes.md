# Reinforcement Learning

Reinforcement Learning is teaching a software agent (decision maker) how to behave in an environment such it maximizes the cumulative reward. Agent learns by taking actions (making decisions).

In our snake game, the agent (AI player) will learn to play snake game (environment) by taking actions (making move) and learning from them.

Each action leads to a state(current snapshot of environment) and a reward. The reward tells the agent how good the previous action was. The goal of agent is to maximize the reward for the long term.

## Reward
Let's define the reward for the game.
- If snake eats food, reward is +10
- If game gets over, reward is -10
- For every other action, reward is 0

## Code organization

We will have a Game class, representing the envionment. This class will have a `play_step` function which is called by `Agent` class to take action.
```python
reward, is_game_over, score = play_step(action)
```

Similarly, we will have a `Model` class which learns `Policy` it maps the current state (current snapshot of the environment) to the best possible action agent can take. During training, model will learn this policy by taking actions and collecting rewards. It will eventually learn what actions are best in current state.
```python
action = model.predict
```

`Agent` class will represent the AI agent, which will take action in the game (environment). The agent will take action based on current policy dictated by the model.

## Action
In our simple game world, we can take left, right, up and down actions. However this may not be the best approach. Instead better actions are:
- straight: the snake continues to move in the direction it is moving right now.
- turn left: the snake makes a left turn
- turn right: the snake makes a right turn.

```python
[1, 0, 0] -> straight
[0, 1, 0] -> right turn
[0, 0, 1] -> left turn
```
## States
Based on the game environment, we can describe the state using these 11 variables.
- danger straight: If snake continuous in straight direction, the snake dies.
- danger left: If the snake makes a left turn, the snake dies.
- danger right: If the snake makes a right turn, the snake dies.
- direction left: The snake is going in the left direction.
- direction right: The snake is going in the right direction.
- direction up: The snake is going in up direction.
- direction down: The snake is going in the down direction.
- food left: Food is on the left direction of the snake.
- food right: Food is on the right direction of the snake.
- food top: Food is on the top direction of the snake.
- food down: Food is on the down direction of the snake.

So a state of `[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]` says:
- There is no danger currently for snake's next move.
- The snake is going in the left direction.
- The food is on the left and top of the snake.

## Model
Our model maps the current state to an action. The neural network will have input layer of size 11 (current representation of state) and output layer will have size of 3, predicting a logit for each 3 action. We choose the max of the logit to determine the next action.

### Model training
We are using Deep-Q learning. Q stands for quality of the action. So these steps define how model will be trained.
1. Init Q Value (initialization of the model)
2. Choose action (`model.predict(state)` or a random action for exploration)
3. Perform action
4. Measure reward
5. Update Q vaule (+ train the model)
Repeat step 2 to 4, till the model learns.

So what is our loss function to update the Q value. We want the model to learn best action for the given state. So during training we want to penalize the model for taking a wrong action and we want to encourage a good action. We also want to discount for future rewards. We will leverage Bellman's equation for this.

$$Q_{new}(s, a) = Q(s, a) + \alpha[R(s, a) + \gamma(max(Q'(s', a')) - Q(s, a))]$$

With this, we can just use mean squared error as our loss

$$loss = (Q_{new} - Q)^2$$

In terms of how this materializes in the code:
```python
q = model.predict(state0)
q_new = R + gamma * max(Q(state_1))
```


