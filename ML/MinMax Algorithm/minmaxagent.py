from math import inf
class MinMaxAgent:
    def __init__(self, player, difficulty=0):
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        self.setDifficulty(difficulty)

    def setDifficulty(self, difficulty):
        self.difficulty = difficulty

    def choose_action(self, state):
        best_score = -inf
        best_action = None
        for action in state.available_actions():
            next_state = state.play(action, self.player)
            score = self.minimax(next_state, self.difficulty, -inf, inf,False)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def minimax(self, state, depth, alpha, beta, is_maximizing):
        if state.is_winner(self.player):
            return 1
        elif state.is_winner(self.opponent):
            return -1
        elif state.is_draw():
            return 0

        if depth == 0:
            return 0

        if is_maximizing:
            max_score = -inf
            for action in state.available_actions():
                next_state = state.play(action, self.player)
                score = self.minimax(next_state, depth - 1, alpha, beta,False)
                max_score = max(score, max_score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = inf
            for action in state.available_actions():
                next_state = state.play(action, self.opponent)
                score = self.minimax(next_state, depth - 1, alpha, beta, True)
                min_score = min(score, min_score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score
