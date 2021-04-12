from app import app
import os
import sys

if __name__ == "__main__":
        sys.path.insert(1, os.path.realpath(__file__))
        app.run(debug=True)
