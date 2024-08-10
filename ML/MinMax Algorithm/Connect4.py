from TicTacToe import MinMaxAgent
from tables import Connect4

def main():
    game = Connect4()
    player_letter = input('Choose your letter (X or O): ')
    if player_letter not in game.available_letters:
        print('Invalid letter')
        return
    difficulty = int(input('Choose difficulty level 0-16: '))
    if player_letter == 'X':
        agent = MinMaxAgent('O', difficulty)
    else:
        agent = MinMaxAgent('X', difficulty)

    player = 'X'
    while True:
        if player == player_letter:
            game.print_board()
            print(f'Your turn! Available actions: {game.available_actions()}')
            action = int(input('Enter your action: '))
            game = game.play(action, player)
        else:
            action = agent.choose_action(game)
            game = game.play(action, player)
        if game.is_winner(player):
            game.print_board()
            print(f'{player} wins!')
            break
        elif game.is_draw():
            game.print_board()
            print('Draw!')
            break
        player = 'X' if player == 'O' else 'O'

if __name__ == '__main__':
    main()
    ex = input('Press Enter to exit or any key to play again: ')
    while ex != '':
        main()
        ex = input('Press Enter to exit or any key to play again: ')