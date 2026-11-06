"""Train the GTR style embedder without import-time login or fixed devices."""
from gtr_pipeline import main

if __name__ == "__main__":
    main(default_command="train")
