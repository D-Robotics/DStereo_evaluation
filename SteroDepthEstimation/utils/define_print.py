import os
import sys
from pathlib import Path
from colorama import Fore, Back, Style, init
import datetime
import inspect

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class DPrint:
    color_fore = {
        'RED': Fore.RED,
        'GREEN': Fore.GREEN,
        'BLUE': Fore.BLUE,
        'YELLOW': Fore.YELLOW,
        'MAGENTA': Fore.MAGENTA,
        'CYAN': Fore.CYAN,
        'WHITE': Fore.WHITE
    }

    color_back = {
        'RED': Back.RED,
        'GREEN': Back.GREEN,
        'BLUE': Back.BLUE,
        'YELLOW': Back.YELLOW,
        'MAGENTA': Back.MAGENTA,
        'CYAN': Back.CYAN,
        'WHITE': Back.WHITE 
    }

    color_style = {
        "DIM": Style.DIM, 
        "NORMAL": Style.NORMAL, 
        "BRIGHT": Style.BRIGHT, 
        "RESET": Style.RESET_ALL
        }

    def __init__(self):
        init(autoreset=True)

    @classmethod
    def color_print(cls, message:str, color:str = "") -> None:
        """
        Prints a message with a specified color and an optional timestamp.

        Parameters:
            message (str): The message to be printed.
            color (str, optional): The color of the message. Defaults to "".

        Returns:
            None
        """
        color_codes = {
                    "red": "\033[1;31;40m",
                    "green": "\033[1;32;40m",
                    "yellow": "\033[1;33;40m",
                    "reset": "\033[0m"
                }
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "  " if timestamp else ""
        color_code = color_codes.get(color, "")
        reset_code = color_codes['reset']
        print(f"{timestamp_str}{color_code}{message}{reset_code}")

    @classmethod
    def print_color_message(cls, message:str, color_f:str="", color_b:str="", color_s:str="") -> None:
        """
        Prints a message in a specified color with a timestamp and the current filename.

        Parameters:
            message (str): The message to be printed.
            message (str): The message to be printed.
            color_f (str, optional): The foreground color of the message. Supported colors are 'RED', 'GREEN', 'BLUE', 'YELLOW', 'MAGENTA', 'CYAN', 'WHITE'. Defaults to "".
            color_b (str, optional): The background color of the message. Supported colors are 'RED', 'GREEN', 'BLUE', 'YELLOW', 'MAGENTA', 'CYAN', 'WHITE'. Defaults to "".
            color_s (str, optional): The style of the message. Supported styles are 'BRIGHT', 'DIM', 'NORMAL'. Defaults to "".
        Returns:
            None
        """
        # Initialize Colorama
        init(autoreset=True)

        # Get the color code or default to white
        color_fore = cls.color_fore.get(color_f, Fore.WHITE)
        color_back = cls.color_back.get(color_b, Back.WHITE)
        color_style = cls.color_style.get(color_s, Style.BRIGHT)

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get the current filename
        current_file = os.path.basename(inspect.getframeinfo(inspect.currentframe().f_back).filename)

        # Print the message with timestamp, filename, and specified color
        print(f"{timestamp} - {current_file}: {color_fore}{color_back}{color_style}{message}{Style.RESET_ALL}")



if __name__ == "__main__":
    color_print = DPrint()
    color_print.print_color_message("This is an error message", "RED")
    color_print.print_color_message("This is a success message", "GREEN")
    color_print.print_color_message("This is an informational message", "BLUE")
    color_print.print_color_message("This is a warning message", "YELLOW")