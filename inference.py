"""Command-line entry point; the code lives in geosam2/_propagation.py."""

from geosam2._propagation import main

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
