"""CLI for preflight and training against user-supplied frozen expert exports."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jarvis_hpsafe.train import main
if __name__=='__main__':main()
