"""
Custom tkinter display to show previous log actions

Last Updated: Summer Term, 2021
Author: Michael Li
"""

import tkinter as tk
import tkinter.font as tkFont


class Table:
    """
    The Table class is used to create a new tkinter widget to show the previously
    logged actions on the picamera
    """

    def __init__(self, root, lst):
        """

        :param root: (tk.Frame) The root tk window for displays
        :param lst: (array[str]) List of recorded actions
        """
        logFont = tkFont.Font(family='Courier', size=13)

        widths = [18, 61]

        entry = tk.Label(root, width=widths[0], font=logFont,
                         text="Timestamp")
        entry.grid(row=0, column=0)

        entry = tk.Label(root, width=widths[1], font=logFont,
                         text="Action")
        entry.grid(row=0, column=1)

        numRows = len(lst)
        numCols = 2

        for i in range(21):
            for j in range(numCols):
                entry = tk.Entry(root, width=widths[j], font=logFont)
                entry.grid(row=i + 1, column=j)

                if i >= numRows:
                    entry.insert(tk.END, "")
                else:
                    entry.insert(tk.END, lst[i][j])