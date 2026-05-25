import gymnasium as gym
import numpy as np
import random
import time

env = gym.make("CartPole-v1", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPISODES = 500

epsilon = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

DISCRETE_OS_SIZE = [20, 20, 20, 20]

high = env.observation_space.high
low = env.observation_space.low


high[1] = 5
high[3] = 5

low[1] = -5
low[3] = -5

discrete_os_win_size = (
    high - low
) 



q_table = np.random.uniform(
    low=-2,
    high=0,
    size=(DISCRETE_OS_SIZE + [env.action_space.n])
)


def get_discrete_state(state):

    discrete_state = (
        (state - low) / discrete_os_win_size
    )

    discrete_state = np.clip(
        discrete_state,
        0,
        np.array(DISCRETE_OS_SIZE) - 1
    )

    return tuple(discrete_state.astype(int))



for episode in range(EPISODES):

    state, info = env.reset()

    discrete_state = get_discrete_state(state)

    done = False

    total_reward = 0

    while not done:

        env.render()

        time.sleep(0.02)

       
        if random.random() > epsilon:
            action = np.argmax(
                q_table[discrete_state]
            )
        else:
            action = env.action_space.sample()

       

        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_state)

        total_reward += reward

       
        if not done:

            max_future_q = np.max(
                q_table[new_discrete_state]
            )

            current_q = q_table[
                discrete_state + (action,)
            ]

            new_q = (
                (1 - LEARNING_RATE) * current_q
                + LEARNING_RATE * (
                    reward
                    + DISCOUNT * max_future_q
                )
            )

            q_table[
                discrete_state + (action,)
            ] = new_q

        discrete_state = new_discrete_state

    

    epsilon = max(
        MIN_EPSILON,
        epsilon * EPSILON_DECAY
    )

   

    print(
        f"Episode: {episode + 1}, "
        f"Reward: {total_reward}, "
        f"Epsilon: {epsilon:.3f}"
    )




env.close()
