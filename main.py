"""
Main entry point for Neural Dungeon.
Initializes Pygame, opens the game window, and launches the engine.
"""

import sys
import pygame
import config
from game.engine import GameEngine


def main() -> None:
    # Initialize all Pygame submodules
    pygame.init()
    pygame.font.init()

    # Set up display surface
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.WINDOW_TITLE)

    # Launch engine
    engine = GameEngine(screen=screen)
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n[Neural Dungeon] Game interrupted by user. Exiting cleanly.")
    finally:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
