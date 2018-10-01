import GeneticAlgo

import socket
import sys
from _thread import *

size = 13

#Function for handling connections. This will be used to create threads
def clientthread(conn):
    #Sending message to connected client
    #infinite loop so that function do not terminate and thread do not end.
    while True:
        
        #Receiving from client
        data = conn.recv(1024)
        
        
        if not data: 
            break
            
        readData(data)
        
    #came out of loop
    conn.close()


def readData(str1):
    str1 = str(str1)
    str1 = str1[2:-1]
    available = False;
    inputs = [None] * size
    q = 0;
    specialValue = "";
    for i in range(size):
        inputs[i] = [None] * size
        p = 0
        while p < size:
            if (str1[q] == "-"):
                specialValue = "-";
                p-= 1;
            else:
                inputs[i][p] = int(specialValue + str1[q]);
                specialValue = "";
            q+= 1;
            p+= 1
    marioX = int(str1[q:(len(str1) - 1)]);
    isAlive = int(str1[len(str1) - 1:len(str1)]);
    if (isAlive == 1):
        alive = False;
    
    else:
        alive = True;   
    if (inputs[0] != None):
        available = True;   
    q = 0;
    #Outputs of this function: inputs, marioX, isAlive
    GeneticAlgo.inputs = inputs
    GeneticAlgo.marioX = marioX
    GeneticAlgo.isAlive = isAlive
    #print(inputs)
    #print(marioX)
    #print(isAlive)
    #Draw board here from inputs
    
    
def connect():
    print("running connect")
    HOST = ''   # Symbolic name meaning all available interfaces
    PORT = 10309 # Arbitrary non-privileged port
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    
    #Bind socket to local host and port
    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        sys.exit()
        
    print('Socket bind complete')
    
    #Start listening on socket
    s.listen(10)
    print('Socket now listening')
    
    
    
    #now keep talking with the client
    while 1:
        #wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
        
        #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
        start_new_thread(clientthread ,(conn,))
        GeneticAlgo.Run()
    
    s.close()
    
    
    #main()
    
