# 19 July 2014
# 08 Dec 2016 - updated for Python3

# in case any of this upsets Python purists it has been converted from an equivalent JRuby program

# this is designed to work with ... ArduinoPC2.ino ...

# the purpose of this program and the associated Arduino program is to demonstrate a system for sending 
#   and receiving data between a PC and an Arduino.

# The key functions are:
#    sendToArduino(str) which sends the given string to the Arduino. The string may 
#                       contain characters with any of the values 0 to 255
#
#    recvFromArduino()  which returns an array. 
#                         The first element contains the number of bytes that the Arduino said it included in
#                             message. This can be used to check that the full message was received.
#                         The second element contains the message as a string


# the overall process followed by the demo program is as follows
#   open the serial connection to the Arduino - which causes the Arduino to reset
#   wait for a message from the Arduino to give it time to reset
#   loop through a series of test messages
#      send a message and display it on the PC screen
#      wait for a reply and display it on the PC

# to facilitate debugging the Arduino code this program interprets any message from the Arduino
#    with the message length set to 0 as a debug message which is displayed on the PC screen

# the message to be sent to the Arduino starts with < and ends with >
#    the message content comprises a string, an integer and a float
#    the numbers are sent as their ascii equivalents
#    for example <LED1,200,0.2>
#    this means set the flash interval for LED1 to 200 millisecs
#      and move the servo to 20% of its range

# receiving a message from the Arduino involves
#    waiting until the startMarker is detected
#    saving all subsequent bytes until the end marker is detected

# NOTES
#       this program does not include any timeouts to deal with delays in communication
#
#       for simplicity the program does NOT search for the comm port - the user must modify the
#         code to include the correct reference.
#         search for the lines 
#               serPort = "/dev/ttyS80"
#               baudRate = 9600
#               ser = serial.Serial(serPort, baudRate)
#


#=====================================

#  Function Definitions

#=====================================

def sendToArduino(sendStr):
    ser.write(sendStr.encode('utf-8')) # change for Python3
    

#======================================

def recvFromArduino():
    global startMarker, endMarker
    
    ck = ""
    x = "z" # any value that is not an end- or startMarker
    byteCount = -1 # to allow for the fact that the last increment will be one too many
    
    # wait for the start character
    while  ord(x) != startMarker: 
        x = ser.read()
    
    # save data until the end marker is found
    while ord(x) != endMarker:
        if ord(x) != startMarker:
            ck = ck + x.decode("utf-8") # change for Python3
            byteCount += 1
        x = ser.read()
    
    return(ck)


#============================

def waitForArduino():

    # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded
    
    global startMarker, endMarker
    
    msg = ""
    while msg.find("Arduino is ready") == -1:

        while ser.inWaiting() == 0:
            pass
        
        msg = recvFromArduino()

        print (msg)
        print ()
        
#======================================

def runTest(td):
    numLoops = len(td)
    waitingForReply = False

    n = 0
    while True:
        if waitingForReply == False:
            if n < numLoops:
                teststr = "<" + td[n] + ">"
                sendToArduino(teststr)
                print ("Sent from PC -- LOOP NUM " + str(n) + " TEST STR " + teststr)
            waitingForReply = True

        if waitingForReply == True:

            while ser.inWaiting() == 0:
                pass
            
            dataRecvd = recvFromArduino()
            print ("Reply Received  " + dataRecvd)
            n += 1
            waitingForReply = False

            print ("===========")

        time.sleep(.2)

        if (n == numLoops):
            break
    time.sleep(20)


#======================================

# THE DEMO PROGRAM STARTS HERE

#======================================

import serial
import time
import argparse

parser = argparse.ArgumentParser(description='Send painting instructions to Arduino robot arm')

parser.add_argument('--instructions_file', default='arduino_actions/actions_all.csv', type=str, help='CSV Brush stroke insturction file')

args = parser.parse_args()

# NOTE the user must ensure that the serial port and baudrate are correct
# serPort = "/dev/ttyS80"
serPort = "COM4"
baudRate = 9600
ser = serial.Serial(serPort, baudRate)
ser.set_buffer_size(rx_size = 13000, tx_size = 13000)
ser.flushInput()
print ("Serial port " + serPort + " opened  Baudrate " + str(baudRate))

startMarker = 60
endMarker = 62

waitForArduino()

stroke_file = open(args.instructions_file, 'r') 

lines = stroke_file.readlines()
stroke_file.close()
lines_cleaned = []
for line in lines: 
    if len(line.strip()) > 0:
        lines_cleaned.append(line.strip());


runTest(lines_cleaned)


ser.close