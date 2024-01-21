from aiPlayer import player, aiPlayer

class FourinarowGame:
    # TODO: create an abstract class "player" and have two such players be
    # parameters of a game. aiPlayer and humanPlayer inherit the abstract class
    def __init__(self, p1="P1", p2="P2"):
        if isinstance(p1, str):
            self.__p1 = player(p1)
            self.__p2 = player(p2)
            self.__p1Id = p1
            self.__p2Id = p2
        elif isinstance(p1, aiPlayer):
            self.__p1 = p1
            self.__p2 = p2
            self.__p1Id = p1.get_id()
            self.__p2Id = p2.get_id()
        self.__inturn = self.__p1
        self.__gameboard = []
        for i in range(7):
            self.__gameboard.append([])
            self.__gameboard[i] = ([None] * 6)


    def playTurn(self, col):
        """
        Plays a turn for the current player.
        Returns None if invalid move.
        """
        row = 0
        for tile in self.__gameboard[col]:
            if tile == None:
                if self.__inturn == self.__p1:
                    self.__gameboard[col][row] = self.__p1Id
                    self.__inturn = self.__p2
                else:
                    self.__gameboard[col][row] = self.__p2Id
                    self.__inturn = self.__p1
                return row
            else:
                row += 1
                continue
        return None
    

    def getResults(self):
        """
        See if anyone has won or if there are no tiles left (draw)
        return: the winner of the game ("P1", "P2"), or None in case of draw
        """

        # Pystyrivi
        for i in range(7):
            for j in range(3):
                if self.__gameboard[i][j] == self.__gameboard[i][j+1] ==\
                        self.__gameboard[i][j+2] == self.__gameboard[i][j+3] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"

        # Vaakarivi
        for j in range(6):
            for i in range(4):
                if self.__gameboard[i][j] == self.__gameboard[i+1][j] ==\
                        self.__gameboard[i+2][j] == self.__gameboard[i+3][j] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"

        # Vino rivi
        for j in range(3):
            for i in range(4):
                if self.__gameboard[i][j] == self.__gameboard[i+1][j+1] ==\
                        self.__gameboard[i+2][j+2] ==\
                        self.__gameboard[i+3][j+3] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"

        for j in range(3, 6):
            for i in range(4):
                if self.__gameboard[i][j] ==\
                        self.__gameboard[i+1][j-1] ==\
                        self.__gameboard[i+2][j-2] ==\
                        self.__gameboard[i+3][j-3] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"

        for j in range(3):
            for i in range(3, 7):
                if self.__gameboard[i][j] ==\
                        self.__gameboard[i-1][j+1] ==\
                        self.__gameboard[i-2][j+2] ==\
                        self.__gameboard[i-3][j+3] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"

        for j in range(3, 6):
            for i in range(3, 7):
                if self.__gameboard[i][j] ==\
                        self.__gameboard[i-1][j-1] ==\
                        self.__gameboard[i-2][j-2] ==\
                        self.__gameboard[i-3][j-3] !=\
                        None:
                    return f"{self.__gameboard[i][j]} has won!"
        
        
        for i in range(7):
            if self.__gameboard[i][5] == None:
                return False
        # Draw
        return "Draw"
    

    def getGameBoard(self):
        return self.__gameboard
    

    def getInturn(self):
        return self.__inturn
