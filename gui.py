from tkinter import Tk, Label, Button, IntVar
import server
import socket
import sys
from _thread import *


class MyFirstGUI:
    def __init__(self, master, size, inputs):
        self.master = master
        self.marioX = IntVar()
        self.isAlive = IntVar()
        self.inputs = []
        self.size = size
        master.title("SuperNEAT")
        self.label = Label(master, textvariable=self.marioX).grid(row=0)
        #self.label.pack()
        self.label = Label(master, textvariable=self.isAlive).grid(row=1)
        #self.label.pack()
        for i in range(size):
            self.inputs.append([])
            for j in range(size):
                self.inputs[i].append(IntVar())
                self.inputs[i][j].set(inputs[i][j])
                self.label = Label(master, textvariable=self.inputs[i][j]).grid(row=i+2, column=j)
                #self.label.pack()



    def update(self):
        self.marioX.set(server.GeneticAlgo.marioX)
        self.isAlive.set(server.GeneticAlgo.isAlive)
        for i in range(self.size):
            for j in range(self.size):
                self.inputs[i][j].set(server.GeneticAlgo.inputs[i][j])
        self.master.after(100, lambda: self.update())
        
def main():
    start_new_thread(server.connect,())
    size = server.size
    server.GeneticAlgo.inputs = []
    for i in range(size):
        server.GeneticAlgo.inputs.append([])
        for j in range(size):
            server.GeneticAlgo.inputs[i].append(0)
    root = Tk()
    my_gui = MyFirstGUI(root, size, server.GeneticAlgo.inputs)
    root.after(100, lambda: my_gui.update())
    root.mainloop()

main()