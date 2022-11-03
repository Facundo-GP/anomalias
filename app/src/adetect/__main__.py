from .detectors import Detectors
from . import api


def main():
    detectors = Detectors()
    api.start(detectors)


if __name__ == "__main__":
    main()
