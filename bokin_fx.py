"""Bokin FX: start the app (python bokin_fx.py), or check it (python bokin_fx.py --selftest)."""
import multiprocessing

from fx.app import entry

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the installed app's render workers start as this program
    entry()
