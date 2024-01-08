from tkinter import *
from functools import partial
from fourinarowGame import FourinarowGame

class Gui_4iar:

    def __init__(self):
        self.__window = Tk()
        self.__window.title("Four in a row")
        self.__red = PhotoImage(file="red_dot.GIF")
        self.__yellow = PhotoImage(file="yellow_dot.GIF")
        self.__gray = PhotoImage(file="harmaa.GIF")
        self.__window.geometry("800x750")
        
        self.__colorcode = {"P1 has won!": "red", "P2 has won!": "yellow",
                            "Draw": "grey"}

        self.setupBoardLabels()
        self.setupButtons()
        self.startNewGame()

    
    def setupBoardLabels(self):
        # Playing grid
        self.__columns = []
        for i in range(7):
            self.__columns.append([])
            for _ in range(6):
                self.__columns[i].append(Label(self.__window, padx=51, pady=43))
        
        # Setting the labels on a grid, column indexing (j) starts from the
        # bottom
        for i in range(7):
            for j in range(6):
                self.__columns[i][j].grid(row=7 - j, column=i)

        # Info label
        self.__infoLabel = Label(self.__window)
        self.__infoLabel.grid(row=0, column=0, columnspan=2)
        self.__turnLabel = Label(self.__window)
        self.__turnLabel.grid(row=0, column=2, columnspan=4)


    def setupButtons(self):
        self.__colButtons = []
        for i in range(7):
            drop = partial(self.drop, i)
            self.__colButtons.append(Button(self.__window,
                                             text=f"   {i + 1}   ",
                                             command=drop))
            self.__colButtons[i].grid(row=8, column=i)

        Button(self.__window, text="New Game", command=self.startNewGame) \
            .grid(row=0, column=7, sticky=W + E)
        Button(self.__window, text="quit", command=self.__window.destroy) \
            .grid(row=1, column=7, sticky=W + E)


    def startNewGame(self):
        """ Reset the game board and labels """
        self.game = FourinarowGame()
        self.clear_board()
        self.show_turn()
        for i in range(7):
            self.__colButtons[i].configure(state=NORMAL)
        self.__infoLabel.configure(text="", bg="SystemButtonFace")

    
    def clear_board(self):
        for i in range(7):
            for label in self.__columns[i]:
                label.configure(image=self.__gray)


    def show_turn(self):
        self.__turnLabel.configure(text=f"{self.game.getInturn()}'s turn")


    def drop(self, col):
        """ Drop a token to the bottom of the column. Update gui."""
        row = self.game.playTurn(col)
        if row == None:
            self.__infoLabel.configure(text="Column full, pick another")
        else:
            self.__infoLabel.configure(text="")
            self.show_turn()
            self.updateGuiBoard(col, row)

        if self.game.getResults():
            self.__infoLabel.configure(text=self.game.getResults(),
                                    bg=self.__colorcode[self.game.getResults()])
            self.__turnLabel.configure(text="")
            self.disableButtons()


    def updateGuiBoard(self, col, row):
        game2gui = {None: self.__gray, "P1": self.__red, "P2": self.__yellow}
        self.__columns[col][row].configure(image=
                                game2gui[self.game.getGameBoard()[col][row]])
        
    
    def disableButtons(self):
        for i in range(7):
            self.__colButtons[i].configure(state=DISABLED)


    def getGame(self):
        return self.game


    def start(self):
        self.__window.mainloop()


def main():
    gui = Gui_4iar()
    gui.start()

if __name__ == "__main__":
    main()
