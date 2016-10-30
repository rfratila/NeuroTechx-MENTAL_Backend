 
* default set of instructions to send to the board on startup via command line
* make a proper package
* requirements.txt for pip
* fix long recording crash (might be hardware or software fix)


## To run the plugin: 
### python user.py -p /dev/tty.usbserial-xxxxx --add abhi person Abhishek 
Make sure that the plugin shows up in the active plugins [abhi]

To start collecting data use /start on the openbci command line 

Proper shutdown sequence is to type in /stop 

By default it doesn't use the plotly plugin, to use it supply True value for graph argument 

## Things to install 
Flask, sklearn, yapsy, plotly, scipy


