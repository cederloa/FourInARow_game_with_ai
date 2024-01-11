class FourinarowGame:
    # TODO: create an abstract class "player" and have two such players be
    # parameters of a game. aiPlayer and humanPlayer inherit the abstract class
    def __init__(self, p1Id="P1", p2Id="P2"):
        self.__p1Id = p1Id
        self.__p2Id = p2Id
        self.__inturn = self.__p1Id
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
                if self.__inturn == self.__p1Id:
                    self.__gameboard[col][row] = self.__p1Id
                    self.__inturn = self.__p2Id
                else:
                    self.__gameboard[col][row] = self.__p2Id
                    self.__inturn = self.__p1Id
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
