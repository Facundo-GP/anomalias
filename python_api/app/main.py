from anomalias.detectors import Detectors
import anomalias.api as api


def main():
    detectors = Detectors()
    api.start(detectors)


if __name__ == "__main__":
    main()