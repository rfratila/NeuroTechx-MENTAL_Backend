# NeuroTechx-MENTAL_Backend
This is where we will store all of the code produced. Please try to keep it documented so that we can all learn from one another.

# To run the plugin 
When the openbci is plugged in and on GPIO 6 mode (bluetooth dongle) and the board is on BLE mode, go to /dev on the system and grep for the tty.usb* = this will tell you the device ID for your board 

Then to run in training mode run the following command (replace * with the actual ID), replace with your own name
#### python user.py -p /dev/tty.usbserial* --add abhi person Abhishek train