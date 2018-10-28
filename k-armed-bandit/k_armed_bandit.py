import numpy as np
from typing import Tuple, List

np.random.seed()


def initialize_action_value_function(k: int) -> List[int]:
    q_real = np.random.randint(10 * k, size=k)
    return q_real


def get_action_reward(
        q_real_action: int, std: float=1.0) -> float:
    return np.random.normal(q_real_action, std)


def get_greedy_action(Q: List[int]) -> int:
    greedy_actions = [np.argmax(Q)]
    last_index = greedy_actions[-1]
    if last_index != len(Q) - 1:
        next_max = last_index + np.argmax(Q[last_index + 1:]) + 1
        while Q[last_index] == Q[next_max]:
            greedy_actions.append(next_max)
            last_index = greedy_actions[-1]
            if last_index == len(Q) - 1:
                break
            next_max = last_index + np.argmax(Q[last_index + 1:]) + 1
    chosen_action = np.random.choice(greedy_actions)
    return chosen_action


def get_solution_stats(
        Q: List[float], q_real: List[int], cumulative_reward: float,
        iteration_no: int=100) -> Tuple[float, float]:
    mse = ((np.array(q_real) - np.array(Q)) ** 2).mean(axis=0)
    average_reward = cumulative_reward / iteration_no / np.max(q_real)
    return mse, average_reward


def stationary_epsilon_greedy_solution(
        k: int, q_real: list, iteration_no: int=100,
        epsilon: float=0.1, standard_deviation: float=1.0) -> tuple:
    performance_mse = []
    performance_average_reward = []
    Q = [0] * k
    N = [0] * k
    cumulative_reward = 0
    for i in range(iteration_no):
        probability = np.random.rand()
        if probability > epsilon:
            action = get_greedy_action(Q)
        else:
            action = np.random.randint(k)

        reward = get_action_reward(q_real[action], standard_deviation)
        cumulative_reward += reward
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        
        mse, average_reward = get_solution_stats(Q, q_real, cumulative_reward, iteration_no=(i + 1))
        performance_mse.append(mse)
        performance_average_reward.append(average_reward)
    return performance_mse, performance_average_reward


def print_stats(
        q_real, N, iteration_no, cumulative_reward, policy=None, H=None, Q=None,
        alpha=None, epsilon=None, c=None, init_value=None):
    if alpha:
        additional_factor = alpha
    elif epsilon:
        additional_factor = epsilon
    elif c:
        additional_factor = c
    elif init_value:
        additional_factor = init_value
    else:
        additional_factor = 0
    k = len(q_real)
    best_possible_action = np.argmax(q_real)
    percentage_best_possible_action = N[best_possible_action] / iteration_no * 100
    print("Percentage of taking best posible action in %d itearation: %5.2f %%, additional_factor: %5.2f\n"
          % (iteration_no, percentage_best_possible_action, additional_factor))
    print("Average reward: %8.2f\n" % (cumulative_reward / iteration_no))
    if H is not None:
        for i in range(k):
            print("H[%d] = %5.2f\tq_real[%d] = %5.2f\tpolicy[%d] = %5.2f\tN[%d] = %5d"
                  % (i, H[i], i, q_real[i], i, policy[i], i, N[i]))
    if Q:
        for i in range(k):
            print("Q[%d] = %5.2f\tq_real[%d] = %5.2f\tN[%d] = %5.2f"
                  % (i, Q[i], i, q_real[i], i, N[i]))
