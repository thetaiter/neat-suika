# Import built-ins
import os
import pickle
import subprocess
import time
from datetime import timedelta
from random import randrange

# Import 3rd party libraries
import cv2
import numpy as np
import pyautogui
import pygetwindow
import pytesseract
from scipy import stats

# Import NEAT libraries
from neat import (
    CompleteExtinctionException,
    DefaultGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    Population,
)
from neat.config import Config
from neat.nn import FeedForwardNetwork
from neat.checkpoint import Checkpointer
from neat.reporting import StdOutReporter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEAT_CONFIG = os.path.join(SCRIPT_DIR, "neat.config")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
HIGH_SCORE_FILE = os.path.join(SCRIPT_DIR, "high_score.txt")
HIGH_FITNESS_FILE = os.path.join(SCRIPT_DIR, "high_fitness.txt")
GAME_LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "suika_combination.log")
SUIKA_EXECUTABLE = os.path.join(
    os.path.expanduser("~"),
    "OneDrive",
    "Desktop",
    "Suika Combination",
    "suika_combination.exe",
)


class CustomStdOutReporter(StdOutReporter):
    def start_generation(self, generation):
        self.generation = generation
        print(f"\n ****** Running generation {generation + 1} ****** \n")
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass


# Suika Game Controller class
class SuikaGameController:
    window: pygetwindow.Win32Window
    title: str
    mouse_speed: int
    position: list[int, int]
    offset: list[int, int]
    bucket_size: int
    score: int
    high_score: int
    high_score_file: str
    neat_run_high_score: int
    neat_run_high_fitness: int
    killswitch: bool
    current_level_images: list
    next_level_images: list
    game_loop: str
    gamer_tag: str
    window_error: bool
    generation: int
    neat_run_start: float
    high_fitness: int
    high_fitness_file: str

    def __init__(
        self,
        mouse_speed: int = 1,
        position: list[int, int] = [1272, 645],
        gamer_tag: str = "thetaiter",
    ):
        self.title = "SuikaCombination"
        self.mouse_speed = mouse_speed
        self.position = position
        self.offset = [408, 105]
        self.bucket_size = 480
        self.score = 0

        self.high_score_file = HIGH_SCORE_FILE
        if os.path.isfile(self.high_score_file):
            with open(self.high_score_file, "r") as file:
                self.high_score = int(file.read().strip())
        else:
            self.high_score = 0
            self._write_high_score()

        self.high_fitness_file = HIGH_FITNESS_FILE
        if os.path.isfile(self.high_fitness_file):
            with open(self.high_fitness_file, "r") as file:
                self.high_fitness = int(file.read().strip())
        else:
            self.high_fitness = 0
            self._write_high_fitness()

        self.neat_run_high_score = 0
        self.neat_run_high_fitness = 0

        self.killswitch = False
        self.current_level_images = [
            cv2.imread(os.path.join(IMAGE_DIR, f"level{i+1}_current.png"))
            for i in range(4)
        ]
        self.next_level_images = [
            cv2.imread(os.path.join(IMAGE_DIR, f"level{i+1}_next.png"))
            for i in range(4)
        ]
        self.gamer_tag = gamer_tag
        self.window_error = False
        self.generation = 0
        self.neat_run_start = 0.0

    def _calculate_bonus(self, positions: list, length_weight: float = 0.5, spread_weight: float = 0.5, multiplier: float = 2000.0):
        # Remove outliers from positions
        z_scores = np.abs(stats.zscore(positions))
        filtered_positions = [pos for pos, z in zip(positions, z_scores) if z < 1]

        # Return zero of filtered array is empty
        if not filtered_positions:
            return 0

        # Calculate a score based on the number of positions and how spread out they are
        length_score = len(filtered_positions) / (self.bucket_size + 1)
        spread_score = (max(filtered_positions) - min(filtered_positions)) / self.bucket_size
        final_score = (length_weight * length_score) + (spread_weight * spread_score)

        return round(final_score * multiplier)
    
    def _calculate_fitness(self, turns: int, playtime: int, positions: list = None):
        return self.score - turns - playtime + self._calculate_bonus(positions)

    def _check_window(self):
        if not pygetwindow.getWindowsWithTitle(self.title):
            print("Window was not found. Restarting.")
            self.launch()
            self.start()
            self.window_error = True

    def _click_image(
        self,
        image: str,
        double: bool = False,
        x_offset: int = 0,
        y_offset: int = 0,
        exit_on_failure: bool = True,
    ):
        location = self._locate_image(image)

        if not location:
            if exit_on_failure:
                pyautogui.alert(
                    f"{image}.png was not found in the window titled '{self.title}'.",
                    "Error",
                )
                self.close()
                exit()
            else:
                return False

        center = pyautogui.center(location)
        pyautogui.moveTo(
            center.x + x_offset, center.y + y_offset, duration=1 / self.mouse_speed
        )

        if double:
            pyautogui.doubleClick()
        else:
            pyautogui.click()

        return True

    def _compute_game_state(self):
        game_bucket = self._get_game_bucket_screenshot()

        game_bucket_image = cv2.cvtColor(np.array(game_bucket), cv2.COLOR_RGB2BGR)
        game_bucket_gray = cv2.cvtColor(game_bucket_image, cv2.COLOR_BGR2GRAY)
        _, game_bucket_thresh = cv2.threshold(
            game_bucket_gray, 64, 255, cv2.THRESH_BINARY
        )
        game_bucket_contours, _ = cv2.findContours(
            game_bucket_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        game_bucket_rectangles = np.zeros(1200)
        for i in range(len(game_bucket_contours)):
            if 4 * i < 1200:
                if cv2.contourArea(game_bucket_contours[i]) > 81:
                    x, y, w, h = cv2.boundingRect(game_bucket_contours[i])
                    game_bucket_rectangles[4 * i] = x
                    game_bucket_rectangles[(4 * i) + 1] = y
                    game_bucket_rectangles[(4 * i) + 2] = w
                    game_bucket_rectangles[(4 * i) + 3] = h

        current_item = self._get_next_screenshot(current=True)
        next_item = self._get_next_screenshot()
        current_item_array = np.array(current_item)[:, :, ::-1]
        next_item_array = np.array(next_item)[:, :, ::-1]

        current_number = -1
        for i in range(len(self.current_level_images)):
            difference = cv2.subtract(current_item_array, self.current_level_images[i])
            result = not np.any(difference)

            if result:
                current_number = i / 3.0

        next_number = -1
        for i in range(len(self.next_level_images)):
            difference = cv2.subtract(next_item_array, self.next_level_images[i])
            result = not np.any(difference)

            if result:
                next_number = i / 3.0

        return [current_number, next_number, *game_bucket_rectangles]

    def _extract_score(self, final: bool = False):
        image = self._get_score_screenshot(final)
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text: str = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )

        if text:
            self.score = int(text)
            if final and self.score > self.high_score:
                image.save(os.path.join(SCRIPT_DIR, "high_score.png"))

    def _get_game_bucket_screenshot(self):
        self._check_window()

        screenshot = pyautogui.screenshot(
            region=(
                self.window.box.left + self.offset[0] - 20,
                self.window.box.top + self.offset[1] - 74,
                520,
                693,
            )
        )

        return screenshot

    def _get_next_screenshot(self, current: bool = False):
        self._check_window()

        if current:
            left_offset = 1056
            top_offset = 135
            width = 86
            height = 105
        else:
            left_offset = 1078
            top_offset = 313
            width = 40
            height = 60

        screenshot = pyautogui.screenshot(
            region=(
                self.window.box.left + left_offset,
                self.window.box.top + top_offset,
                width,
                height,
            )
        )

        return screenshot

    def _get_score_screenshot(self, final: bool = False):
        self._check_window()

        if final:
            x_offset = 538
            y_offset = 342
            width = 220
            height = 70
        else:
            x_offset = 90
            y_offset = 388
            width = 220
            height = 70

        screenshot = pyautogui.screenshot(
            region=(
                self.window.box.left + x_offset,
                self.window.box.top + y_offset,
                width,
                height,
            )
        )

        return screenshot

    def _is_game_over(self):
        score_title = self._locate_image("score-title")

        if score_title:
            print("Game Over!")
            return True

        return False

    def _killswitch(self, sensitivity: int = 25):
        mouse_position = pyautogui.position()

        self._check_window()

        if (
            mouse_position.y > self.window.box.top + self.offset[1] + sensitivity
            or mouse_position.y < self.window.box.top + self.offset[1] - sensitivity
        ):
            self.killswitch = True
            print("Killswitch activated!")

        return self.killswitch

    def _load(self, generation: int = None, genome: str = None):
        if generation is None and genome is None:
            print("No checkpoint or genome was provided.")
            return None

        loaded_population = None
        if generation is not None:
            generation = generation - 1
            self.generation = generation

            if generation < 0:
                raise Exception("Generation must be greater than 0")

            loaded_population = Checkpointer.restore_checkpoint(
                os.path.join(SCRIPT_DIR, "checkpoints", f"generation-{generation}")
            )

            return loaded_population

        loaded_genome = None
        with open(
            os.path.join(SCRIPT_DIR, "genomes", f"genome-{genome}"), "rb"
        ) as file:
            loaded_genome = pickle.load(file)

        return loaded_genome

    def _locate_image(self, image: str):
        location = None
        self._check_window()
        location = pyautogui.locateOnWindow(
            os.path.join(IMAGE_DIR, f"{image}.png"), self.title
        )
        return location

    def _move_mouse_to_position(self, x: int, duration: float = None):
        if x > self.bucket_size:
            x = self.bucket_size

        self._check_window()

        pyautogui.moveTo(
            self.window.box.left + self.offset[0] + x,
            self.window.box.top + self.offset[1],
            duration=duration if duration else 1 / self.mouse_speed,
        )

    def _run_neat(self, genomes: list, config: Config):
        print(f"There are {len(genomes)} genomes to run in this generation\n")

        self.generation += 1
        generation_high_score = 0
        generation_high_fitness = -999999999

        count = 1
        genomes_failed = 0
        generation_start = time.time()
        for _, g in genomes:
            if count > 1:
                print()

            divider_length = 48
            print("=" * divider_length)
            print(
                f"Genome {count}/{len(genomes)} ({round((count / len(genomes)) * 100, 2)}%), Generation {self.generation}"
            )
            print("-" * divider_length)

            net = FeedForwardNetwork.create(g, config)

            if count == 1 and self.generation == 1:
                self.launch()
                self.start()

            turns = 0
            runtime = 0
            g.fitness = 0
            positions = []
            start = time.time()
            while not self._is_game_over() and not self._killswitch():
                state = self._compute_game_state()
                output = net.activate(state)

                position = round(output[0] * self.bucket_size)
                self.drop_at_position(position, output[1])
                time.sleep(output[2])

                if position not in positions:
                    positions.append(position)

                turns += 1

                if self.window_error:
                    turns = 0
                    self.window_error = False
                
                self._extract_score()
                runtime = time.time() - start
                g.fitness = self._calculate_fitness(turns, round(runtime), positions)

            if self.killswitch:
                self.close()
                exit()

            self._extract_score(final=True)
            g.fitness = self._calculate_fitness(turns, round(runtime), positions)

            if g.fitness > generation_high_fitness:
                generation_high_fitness = g.fitness

            if (self.neat_run_high_fitness == 0 and count == 1) or g.fitness > self.neat_run_high_fitness:
                self.neat_run_high_fitness = g.fitness

            if self.score > generation_high_score:
                generation_high_score = self.score

            if g.fitness > self.high_fitness:
                self.high_fitness = g.fitness
                self._write_high_fitness()

            if self.check_high_score():
                self.submit_score(self.gamer_tag)
                self._save_genome(g, f"genome-{self.generation}-{count}")

            print("-" * divider_length)
            print(
                f"Time: {timedelta(seconds=runtime)}\tSeconds:       {round(runtime)}"
            )
            print(f"Turns:         {turns}\tBonus:         {self._calculate_bonus(positions)}")
            print(f"Final Score:   {self.score}  \tFitness:       {g.fitness}")
            print(
                f"Generation HS: {generation_high_score}  \tGeneration HF: {generation_high_fitness}"
            )
            print(
                f"NEAT Run HS:   {self.neat_run_high_score}  \tNEAT Run HF:   {self.neat_run_high_fitness}"
            )
            print(
                f"All Time HS:   {self.high_score}  \tAll Time HF:   {self.high_fitness}"
            )
            print("=" * divider_length)

            self.reset(close_window=False)

            count += 1

        print(f"\nAll genomes in generation {self.generation} have been run!")
        print(f"\nGeneration High Score: {generation_high_score}")
        print(f"Generation High Fitness: {generation_high_fitness}")
        print(
            f"Genomes Failed Early: {genomes_failed}/{len(genomes)} ({round((genomes_failed / len(genomes)) * 100, 2)}%)"
        )
        generation_runtime = time.time() - generation_start
        neat_run_runtime = time.time() - self.neat_run_start
        print(f"Generation Runtime: {timedelta(seconds=generation_runtime)}")
        print(f"Cumulative Runtime: {timedelta(seconds=neat_run_runtime)}")
        print("\nComputing statistics and generating next generation...\n")

    def _save_genome(self, genome, filename):
        with open(os.path.join(SCRIPT_DIR, "genomes", f"{filename}.pkl"), "wb") as file:
            pickle.dump(genome, file)

    def _write_high_fitness(self):
        with open(self.high_fitness_file, "w") as file:
            file.write(str(self.high_fitness))

    def _write_high_score(self):
        with open(self.high_score_file, "w") as file:
            file.write(str(self.high_score))

    def check_high_score(self):
        if not self.killswitch:
            if self.score > self.neat_run_high_score:
                self.neat_run_high_score = self.score

            if self.score > self.high_score:
                self.high_score = self.score
                print("You set a new high score!")
                self._write_high_score()
                return True
        return False

    def close(self):
        if pygetwindow.getWindowsWithTitle(self.title):
            self.window.activate()
            self.window.close()

    def drop_at_position(self, pos: int, duration: float = None):
        self._move_mouse_to_position(pos, duration)
        pyautogui.click()

    def drop_at_random_position(self, duration: float = None):
        self.drop_at_position(randrange(self.bucket_size), duration)

    def get_high_score(self):
        return self.high_score

    def get_score(self):
        return self.score

    def launch(self):
        windows = pygetwindow.getWindowsWithTitle(self.title)
        if windows:
            for window in windows:
                print("Existing window with title SuikaGame was found, closing it now")
                self.window: pygetwindow.Win32Window = window
                self.quit()

        print("Launching Suika Combination")
        if not os.path.exists(os.path.join(SCRIPT_DIR, "logs")):
            os.makedirs(os.path.join(SCRIPT_DIR, "logs"))
        subprocess.Popen(
            f"{SUIKA_EXECUTABLE} --verbose",
            stdout=open(GAME_LOG_FILE, "w"),
            stderr=open(GAME_LOG_FILE, "w"),
        )
        time.sleep(2)

        window: pygetwindow.Win32Window = pygetwindow.getWindowsWithTitle(self.title)[0]
        window.moveTo(*self.position)
        self.window = window

    def quit(self):
        print(f"Quitting Suika Combination")

        if self._click_image("return-to-title-button", exit_on_failure=False):
            self._click_image("quit-button", exit_on_failure=False)
        elif self._click_image("exit-button", exit_on_failure=False):
            self._click_image("yes-button", exit_on_failure=False)
            self._click_image("quit-button", exit_on_failure=False)
        else:
            self._click_image("quit-button", exit_on_failure=False)

        time.sleep(2)

        if pygetwindow.getWindowsWithTitle(self.title):
            self.window.close()
            time.sleep(2)

    def replay_genome(self, genome: str):
        # TODO: Do some stuff to re-play the game with a specific genome
        pass

    def reset(self, close_window: bool = False):
        self.score = 0
        self.killswith = False

        if close_window:
            self.quit()
        else:
            self._click_image("retry-button")
            self._move_mouse_to_position(0)

    def run_neat_game_loop(self, generation: int = None, genome: str = None):
        self.game_loop = "neat"

        self.neat_run_start = time.time()

        population = None
        if generation is not None:
            print(f"Loading population from generation {generation}...")
            population = self._load(generation=generation)
        else:
            print("Loading NEAT config")
            config = Config(
                DefaultGenome,
                DefaultReproduction,
                DefaultSpeciesSet,
                DefaultStagnation,
                NEAT_CONFIG,
            )

            print("Generating population")
            population = Population(config)

        for dir in ("checkpoints", "genomes"):
            if not os.path.exists(os.path.join(SCRIPT_DIR, dir)):
                os.makedirs(os.path.join(SCRIPT_DIR, dir))

        checkpointer = Checkpointer(
            generation_interval=1,
            filename_prefix=os.path.join(SCRIPT_DIR, "checkpoints", "generation-"),
        )

        population.add_reporter(CustomStdOutReporter(True))
        population.add_reporter(checkpointer)

        print("Running NEAT game loop")

        self.neat_run_high_score = 0
        self.neat_run_high_fitness = 0
        winner = None
        try:
            winner = population.run(self._run_neat)
        except TypeError:
            if not self.killswitch:
                raise
        except CompleteExtinctionException:
            print("All species went extinct before the fitness goal was achieved.")
            return None

        if winner:
            with open(os.path.join(SCRIPT_DIR, "genomes", "winner.pkl"), "wb") as f:
                pickle.dump(winner, f)

        return winner

    def run_random_game_loop(self, num_retries, use_same_window: bool = True):
        self.game_loop = "random"

        try_number = 1
        while (try_number <= num_retries) if num_retries else True:
            if try_number != 1:
                print()

            print(f"Running random game {try_number}")

            if try_number == 1 or not use_same_window:
                self.launch()
                self.start()

            turns = 0
            start = time.time()
            while not self._is_game_over() and not self._killswitch():
                self.drop_at_random_position(duration=0)
                self._extract_score()
                turns += 1
            end = time.time()

            self._extract_score(final=True)
            print(f"Time: {timedelta(seconds=end-start)}")
            print(f"Turns: {turns}")
            print(f"Final Score: {self.score}")

            if self.killswitch:
                self.close()
                break

            if self.check_high_score():
                self.submit_score(self.gamer_tag)

            self.reset(close_window=not use_same_window)

            try_number += 1

        if num_retries and num_retries > 1 and not self.killswitch:
            print(f"\nHigh Score: {self.high_score}")

    def start(self):
        print("Starting game")
        self._click_image("start-button")
        self._click_image("classic-button")
        self._move_mouse_to_position(0)

    def submit_score(self, name: str):
        print("Submitting score")
        self._click_image("submit-button", double=True, x_offset=-100)
        pyautogui.typewrite(name)
        self._click_image("submit-button")
