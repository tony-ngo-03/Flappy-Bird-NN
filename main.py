import pygame
import random
from pygame.locals import (
    K_SPACE,
    K_ESCAPE,
    KEYDOWN,
    QUIT
)
import neat
import pickle

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
pygame.init()
GAP_SIZE = SCREEN_HEIGHT // 5
clock = pygame.time.Clock()

PIPE_WIDTH = SCREEN_WIDTH // 15

movement_speed = 2
gravity = 1
spawn_rate = 150  # Initialize spawn_rate
jump_speed = -10


class Bird:
    def __init__(self, y_pos, size):
        self.rect = pygame.Rect(SCREEN_WIDTH // 2, y_pos, size, size)
        self.velocity = 0

    def draw(self):
        pygame.draw.rect(screen, (0, 0, 0), self.rect)

    def update(self):
        self.velocity += gravity
        self.rect.move_ip(0, self.velocity)

    def flap(self):
        self.velocity = jump_speed


class Pipe:
    def __init__(self, x_pos):
        random_height = random.randint(0, SCREEN_HEIGHT - GAP_SIZE)
        self.upper_rect = pygame.Rect(x_pos, 0, PIPE_WIDTH, random_height)
        self.lower_rect = pygame.Rect(x_pos, random_height + GAP_SIZE, PIPE_WIDTH,
                                      SCREEN_HEIGHT - random_height - GAP_SIZE)
        self.passed = False

    def draw(self):
        pygame.draw.rect(screen, (0, 0, 0), self.upper_rect)
        pygame.draw.rect(screen, (0, 0, 0), self.lower_rect)

    def collide(self, other):
        return self.upper_rect.colliderect(other) or self.lower_rect.colliderect(other)

    def update(self):
        self.upper_rect.move_ip(-movement_speed, 0)
        self.lower_rect.move_ip(-movement_speed, 0)

    def __str__(self):
        return "Passed: " + str(self.passed) + "Location: " + str(self.upper_rect.topleft)


class FlappyBirdEnvironment:
    def __init__(self):
        self.bird = Bird(SCREEN_HEIGHT // 2, 20)
        self.pipes = [Pipe(SCREEN_WIDTH)]
        self.score = 0
        self.spawn_timer = pygame.time.get_ticks()

    def initialize_game(self):
        self.bird = Bird(SCREEN_HEIGHT // 2, 20)
        self.pipes = [Pipe(SCREEN_WIDTH)]
        self.score = 0

    def get_state(self):
        pipe_to_return = None
        for curr_pipe in self.pipes:
            if not curr_pipe.passed:
                pipe_to_return = curr_pipe
                break
        return [self.bird.rect.y, pipe_to_return.upper_rect.bottomleft[0], pipe_to_return.upper_rect.bottomleft[1],
                pipe_to_return.lower_rect.topleft[0], pipe_to_return.lower_rect.topleft[1]]

    def perform_action(self, output):
        if output >= 0.5:
            self.bird.flap()

    def update_game(self):
        if self.score >= 9999:
            return True
        self.bird.update()
        for curr_pipe in self.pipes:
            curr_pipe.update()
            if curr_pipe.collide(self.bird.rect):
                return True  # Game over if collision occurs
            elif not curr_pipe.passed and curr_pipe.upper_rect.right < self.bird.rect.left and not curr_pipe.upper_rect.colliderect(
                    self.bird.rect):
                curr_pipe.passed = True
                self.score += 1  # Increment score when passing through pipes

        if self.pipes and self.pipes[-1].upper_rect.right < (SCREEN_WIDTH - PIPE_WIDTH * 3):  # Check the last pipe
            self.pipes.append(Pipe(SCREEN_WIDTH))

        if self.pipes and self.pipes[0].upper_rect.right < 0:
            self.pipes.pop(0)

        if self.bird.rect.y > SCREEN_HEIGHT or self.bird.rect.y < 0:
            return True  # Game over if bird goes out of screen

        return False


def display_score(score):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, (255, 0, 0))
    screen.blit(score_text, (10, 10))


# NEAT Initialization
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        flappy_env = FlappyBirdEnvironment()
        while not flappy_env.update_game():
            current_state = flappy_env.get_state()
            output = net.activate(current_state)
            flappy_env.perform_action(output[0])
        genome.fitness = flappy_env.score


def get_winner_net():
    # Load NEAT configuration
    config_path = "config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    # Create NEAT population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Run NEAT algorithm
    winner = population.run(eval_genomes)

    # Use the best genome to play the game
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner_net, f)
        f.close()
    return winner_net


def play_winner(winner_net, this_frame_rate):
    flappy_env = FlappyBirdEnvironment()
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    quit()
        screen.fill((255, 255, 255))
        flappy_env.perform_action(winner_net.activate(flappy_env.get_state())[0])
        game_over = flappy_env.update_game()
        if game_over:
            flappy_env.initialize_game()
        flappy_env.bird.draw()
        for pipe in flappy_env.pipes:
            pipe.draw()
        display_score(flappy_env.score)
        pygame.display.flip()
        clock.tick(this_frame_rate)
