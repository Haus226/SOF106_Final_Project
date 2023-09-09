def find_empty(board):
    """
    Find if there is any empty place in our sudoku board
    :param board: The board that carry the information of sudoku
    :return: None
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return i, j  # row, col
    return None

def valid(bo, num, pos):
    """

    :param bo:
    :param num:
    :param pos:
    :return:
    """
    # Check the row
    for i in range(len(bo)):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False, num, pos[0], i

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False, num, i, pos[1]

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False, num, i, j
    return True,

def checkBoard(bo):
    """
    Check all position using valid function
    :param bo: The sudoku board
    :return:
    """
    row, col = 0, 0
    while row < 9:
        while col < 9:
            if bo[row][col]:
                result = valid(bo, bo[row][col], (row, col))
                if not result[0]:
                    return result
            col += 1
        col = 0
        row += 1
    return True, # To maintain the returned type : tuple

def solveSudoku(bo):

    find = find_empty(bo)
    if not find:
        return True
    for i in range(1, 10):
        if valid(bo, i, find)[0]:
            bo[find[0]][find[1]] = i
            if solveSudoku(bo):
                return True
            bo[find[0]][find[1]] = 0
    return False



if __name__ == '__main__':
    i = solveSudoku([[0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 9, 0, 5, 0, 6, 0, 8, 0],
    [0, 0, 8, 4, 0, 1, 2, 0, 0],
    [0, 5, 9, 0, 0, 0, 8, 4, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 6],
    [0, 2, 3, 0, 0, 0, 5, 7, 0],
    [0, 0, 5, 3, 0, 7, 4, 0, 0],
    [0, 1, 0, 6, 0, 8, 0, 9, 4],
    [0, 0, 0, 0, 1, 0, 0, 0, 0]])
    print(i)

