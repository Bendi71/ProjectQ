class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Initial empty board
        self.available_letters = ['X', 'O']

    def print_board(self):
        print('-------------')
        for i in range(3):
            print('| ' + ' | '.join(self.board[i * 3:i * 3 + 3]) + ' |')
            print('-------------')

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def is_winner(self, player):
        # Check rows, columns, and diagonals for a win
        win_states = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                      [0, 3, 6], [1, 4, 7], [2, 5, 8],
                      [0, 4, 8], [2, 4, 6]]
        for state in win_states:
            if all(self.board[i] == player for i in state):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def play(self, action, player):
        next_state = TicTacToe()
        next_state.board = self.board.copy()
        next_state.board[action] = player
        return next_state

class Connect4:
    def __init__(self):
        self.board = [' ']*42
        self.available_letters = ['X', 'O']
    def print_board(self):
        print('---------------------------------')
        for i in range(6):
            print(' | ' + ' | '.join(self.board[i*7:i*7+7]) + ' | ')
            print('---------------------------------')
        print()

    def available_actions(self):
        actions = []
        for col in range(7):
            for row in range(5, -1, -1):
                if self.board[row*7+col] == ' ':
                    actions.append(row*7+col)
                    break
        return actions

    def is_winner(self, player):
        # Check horizontal locations for win
        for row in range(6):
            for col in range(4):
                if all(self.board[row * 7 + col + i] == player for i in range(4)):
                    return True
        # Check vertical locations for win
        for col in range(7):
            for row in range(3):
                if all(self.board[(row + i) * 7 + col] == player for i in range(4)):
                    return True
        # Check positively sloped diagonals
        for row in range(3):
            for col in range(4):
                if all(self.board[(row + i) * 7 + col + i] == player for i in range(4)):
                    return True
        # Check negatively sloped diagonals
        for row in range(3):
            for col in range(4):
                if all(self.board[(row + 3 - i) * 7 + col + i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def play(self, action, player):
        next_state = Connect4()
        next_state.board = self.board.copy()
        next_state.board[action] = player
        return next_state
