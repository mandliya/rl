import random
from collections import namedtuple
from enum import Enum
from typing import List, Tuple

import pygame
import numpy as np

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
RED = (200, 0, 0)

# Control variables
BLOCK_SIZE = 20
SPEED = 5

pygame.init()
font = pygame.font.SysFont('Arial', 24)

class Direction(Enum):
    """Enum to represent direction of the snake"""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    """Class to represent the Snake Game"""

    def __init__(self, width: int = 640, height: int = 480):
        """Initialize the game with given width and height"""
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game state"""
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        """Place food randomly on the game board"""
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def _move(self):
        """Move the snake in the current direction"""
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def game_over_screen(self):
        """Display game over screen"""
        self.display.fill(BLACK)
        text = font.render("Game Over! Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [self.width / 2 - 200, self.height / 2 - 50])
        text = font.render("Press Enter to play again or Escape to exit", True, WHITE)
        self.display.blit(text, [self.width / 2 - 200, self.height / 2])
        pygame.display.flip()
        # Reset game if Enter is pressed
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.reset()
                        return
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()

    def is_collision(self, point: Point = None) -> bool:
        """Check if there is a collision at the given point"""
        if point is None:
            point = self.head
        
        # Check boundary collision
        if point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0:
            return True
        
        # Check self collision
        if point in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self):
        """Update the game UI"""
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def play_step(self) -> Tuple[int, bool, int]:
        """Play one step of the game"""
        self.frame_iteration += 1

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
        
        # Update head position
        self._move()
        self.snake.insert(0, self.head)

        # Check for game over
        reward = 0
        is_game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            is_game_over = True
            reward = -10
            return reward, is_game_over, self.score
        
        # Update score if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, is_game_over, self.score

def main():
    """Main function to start the game"""
    snake_game = SnakeGame()
    while True:
        reward, is_game_over, score = snake_game.play_step()
        if is_game_over:
            break
    snake_game.game_over_screen()
    pygame.quit()

if __name__ == '__main__':
    main()