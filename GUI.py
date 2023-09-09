"""
imageReader: Read the image and load the model to recognize the sudoku board
PIL: Read the original image of sudoku and display it in GUI
filedialog: Widget that ask the user to choose a sudoku image to open
messagebox: Widget that pop up to show certain information
Algorithm: The backtracking algorithm to solve the recognized or inputted sudoku
"""

from tkinter import *
import tkinter.filedialog as fd, tkinter.messagebox as ms, tkinter.ttk
from imageReader import imageReader
from Algorithm import solveSudoku, checkBoard
from PIL import Image, ImageTk


class GUI:

    def __init__(self):
        self.__window = Tk()
        self.__window.title("Sudoku Solver")
        self.__window.geometry("700x550")   # Based on your preference

        self.__frame = Frame(self.__window)
        self.__frame.pack()

        self.__buttonFrame = Frame(self.__frame)
        self.__buttonFrame.grid(row=0, column=1)

        self.__imgPath = "" # Store the path of sudoku image that user wish to open
        self.__entries = [] # To store the entries object
        self.__board = [[0 for x in range(9)] for y in range(9)]    # To store the sudoku board but initialize it with all zero first

        tkinter.ttk.Style().configure("TButton", font=("Aerial", 12))
        openButton = tkinter.ttk.Button(self.__buttonFrame, text="Open", command=self.__openFile).grid(row=0, column=0)
        solveButton = tkinter.ttk.Button(self.__buttonFrame, text="Solve", command=self.__solve).grid(row=1, column=0)
        clearButton = tkinter.ttk.Button(self.__buttonFrame, text="Clear", command=self.__cleanEntries).grid(row=2, column=0)

        self.__canvas = Canvas(self.__frame, height=520, width=495)
        self.__createRow()
        self.__createCol()
        self.__createEntry()
        self.__canvas.grid(row=0, column=0)

        '''
        Keyboard event such as Up, Down, Left, Right to change the entry that focused
        '''
        self.__window.bind("<Up>", lambda e: self.__moveCursor(e, "up"))
        self.__window.bind("<Down>", lambda e: self.__moveCursor(e, "down"))
        self.__window.bind("<Left>", lambda e: self.__moveCursor(e, "left"))
        self.__window.bind("<Right>", lambda e: self.__moveCursor(e, "right"))

        self.__window.mainloop()

    def __initBoard(self):
        """
        Initialize the board after input or recognize a sudoku board
        :return: None
        """
        Entries = self.__entries[0]
        m = 1
        for i in range(9):
            for j in range(9):
                if Entries.get():
                    self.__board[i][j] = int(Entries.get())
                if m <= 80:
                    Entries = self.__entries[m]
                    m += 1

    def __configEntries(self):
        """
        Show the result of the recognition or the solution of the sudoku if exists
        :return: None
        """
        self.__cleanEntries()
        E = self.__entries[0]
        m = 1
        for i in range(9):
            for j in range(9):
                if self.__board[i][j]:
                    E.insert(1, self.__board[i][j])
                if m <= 80:
                    E = self.__entries[m]
                    m += 1

    def __solve(self):
        """
        Solve the sudoku and show the solution if exists else an error message will pop out
        :return: None
        """
        self.__cleanBoard()
        self.__initBoard()

        print("Before attempt : ")
        for r in range(len(self.__board)):
            print(self.__board[r])

        checkresult = checkBoard(self.__board)
        print(checkresult)
        if checkresult[0]:
            """
            If there is an 0 exists in the board, this implies that no solution found
            for the current sudoku
            """
            solveSudoku(self.__board)
            for r in range(len(self.__board)):
                if 0 in self.__board[r]:
                    ms.showerror("Error", "No solution found !")
                    return
            self.__configEntries()
            print("Solution : ")
            for r in range(len(self.__board)):
                print(self.__board[r])
        else:
            ms.showerror("Invalid", f"This is not a valid sudoku board since {checkresult[1]} repeated "
                                    f"at row {checkresult[2] + 1}, column {checkresult[3] + 1}!")
            return


    def __cleanEntries(self):
        """
        Clean the entries
        We do not clean the board since we modify it in place
        :return: None
        """
        for entry in self.__entries:
            entry.delete(0, END)

    def __cleanBoard(self):
        """
        Clean the board everytime before we try to solve it
        :return:
        """
        for i in range(len(self.__board)):
            for j in range(len(self.__board)):
                self.__board[i][j] = 0

    def __createEntry(self):
        """
        Create the entries for input or show the sudoku in the canvas created
        :return: None
        """
        p, q = 43.5, 63.5
        cnt = 0
        for i in range(9):
            for j in range(9):
                entry = Entry(self.__canvas, font='BOLD', name=f"{cnt}", highlightthickness=1, highlightcolor="black")
                entry.place(x=p, y=q, height=30, width=35)
                self.__entries.append(entry)
                p += 48.0
                cnt += 1
            q += 40
            p = 43.5

    def __createRow(self):
        """
        Create line to separate the entries created row by row
        :return: None
        """

        x1, y1 = 40, 60
        x2, y2 = 40, 420
        for m in range(10):
            if (m % 3 == 0):
                self.__canvas.create_line(x1, y1, x2, y2, width=3)
            else:
                self.__canvas.create_line(x1, y1, x2, y2, width=1)
            x1 += 48
            x2 += 48

    def __createCol(self):
        """
        Create line to separate the entries created column by column
        :return: None
        """
        i, j = 40, 60
        p, q = 473, 60
        for m in range(10):
            if m % 3 == 0:
                self.__canvas.create_line(i, j, p, q, width=3)
            else:
                self.__canvas.create_line(i, j, p, q, width=1)
            j += 40
            q += 40

    def __openFile(self):
        """
        Ask the user to open the sudoku image and display it in GUI
        :return: None
        """
        self.__imgPath = fd.askopenfilename()
        img = Image.open(self.__imgPath)
        img = img.resize((520, 495))    # This is based on your GUI size
        img = ImageTk.PhotoImage(img)
        label = Label(self.__frame)
        label.grid(row=0, column=0)
        label.img = img
        label["image"] = img
        self.__window.geometry("1200x550")  # Based on your preference

        self.__canvas.grid(row=0, column=1)
        self.__buttonFrame.grid(row=0, column=2)
        print(self.__imgPath)
        self.__board = imageReader(self.__imgPath)
        for r in self.__board:
            print(r)
        print()
        self.__configEntries()

    def __moveCursor(self, e:Event, d):
        """
        Keyboard event such as Up, Down, Left, Right to change the entry that focused
        :param e: A tkinter event
        :param d: Direction for the cursor to move
        :return:
        """
        cur = int(self.__window.focus_get().winfo_name())
        try:
            if d == "up":
                self.__entries[cur - 9].focus_set()
            elif d == "down":
                self.__entries[cur + 9].focus_set()
            elif d == "left" and cur != 0:
                self.__entries[cur - 1].focus_set()
            elif d == "right":
                self.__entries[cur + 1].focus_set()
        except IndexError:
            pass

if __name__ == '__main__':
    GUI()