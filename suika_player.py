# Import built-ins
import os
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

# Import NEAT libraries
from neat import (
    DefaultGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    Population,
    StatisticsReporter,
    StdOutReporter,
)
from neat.config import Config
from neat.nn import FeedForwardNetwork


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEAT_CONFIG = os.path.join(SCRIPT_DIR, "neat.config")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
HIGH_SCORE_FILE = os.path.join(SCRIPT_DIR, "high_score.txt")
GAME_LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "suika_combination.log")
SUIKA_EXECUTABLE = os.path.join(
    os.path.expanduser("~"),
    "OneDrive",
    "Desktop",
    "Suika Combination",
    "suika_combination.exe",
)


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
    killswitch: bool
    current_level_images: list
    next_level_images: list
    game_loop: str
    gamer_tag: str
    window_error: bool

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
        _, game_bucket_thresh = cv2.threshold(game_bucket_gray, 64, 255, cv2.THRESH_BINARY)
        game_bucket_contours, _ = cv2.findContours(game_bucket_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        game_bucket_rectangles = np.zeros(1200)
        for i in range(len(game_bucket_contours)):
            if 4*i < 1200:
                if cv2.contourArea(game_bucket_contours[i]) > 81:
                    x, y, w, h = cv2.boundingRect(game_bucket_contours[i])
                    game_bucket_rectangles[4*i] = x
                    game_bucket_rectangles[(4*i)+1] = y
                    game_bucket_rectangles[(4*i)+2] = w
                    game_bucket_rectangles[(4*i)+3] = h

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
            height = 50
        else:
            x_offset = 90
            y_offset = 388
            width = 220
            height = 48

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
        count = 1
        for _, g in genomes:
            if count != 1:
                print()

            print(f"Running genome {count}")

            net = FeedForwardNetwork.create(g, config)
            g.fitness = 0

            self.launch()
            self.start()

            turns = 0
            compute_times = []
            activation_times = []
            start = time.time()
            while not self._is_game_over() and not self._killswitch():
                start_compute = time.time()
                state = self._compute_game_state()
                end_compute = time.time()
                output = net.activate(state)
                end_activation = time.time()

                compute_times.append(end_compute - start_compute)
                activation_times.append(end_activation - start_compute)

                position = round(output[0] * self.bucket_size)
                self.drop_at_position(position, output[1])
                time.sleep(output[2])

                if self.window_error:
                    turns = 0
                    self.window_error = False

                turns += 1
                self._extract_score()
                g.fitness = max(0, self.score - (turns * 4))
            end = time.time()

            print(f"Time: {timedelta(seconds=end-start)}")
            print(f"Avg Compute Time: {sum(compute_times)/len(compute_times)}")
            print(f"Avg Activation Time: {sum(activation_times)/len(activation_times)}")

            self._extract_score(final=True)
            g.fitness = max(0, self.score - (turns * 4))

            print(f"Final Score: {self.score}")
            print(f"Turns: {turns}")
            print(f"Fitness: {g.fitness}")

            if self.killswitch:
                self.close()
                exit()

            if self.check_high_score():
                self.submit_score(self.gamer_tag)

            self.reset(close_window=True)

            count += 1

        print("\nAll genomes in this generation have been run!")
        print("\nComputing statistics and generating next generation...\n")

    def _write_high_score(self):
        with open(self.high_score_file, "w") as file:
            file.write(str(self.high_score))

    def check_high_score(self):
        if not self.killswitch and self.score > self.high_score:
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
        subprocess.Popen(
            f"{SUIKA_EXECUTABLE} --verbose",
            stdout=open(GAME_LOG_FILE, "w"),
            stderr=open(GAME_LOG_FILE, "w"),
        )
        time.sleep(1)

        window: pygetwindow.Win32Window = pygetwindow.getWindowsWithTitle(self.title)[0]
        window.moveTo(*self.position)
        self.window = window

    def quit(self):
        print(f"Quitting Suika Combination")

        if self._click_image("return-to-title-button", exit_on_failure=False):
            self._click_image("quit-button")
        elif self._click_image("exit-button", exit_on_failure=False):
            self._click_image("yes-button")
            self._click_image("quit-button")
        else:
            self._click_image("quit-button")

    def reset(self, close_window: bool = False):
        self.score = 0
        self.killswith = False

        if close_window:
            self.quit()
        else:
            print(f"Retrying")
            self._click_image("retry-button")
            self._move_mouse_to_position(0)

    def run_neat_game_loop(self):
        self.game_loop = "neat"

        print("Loading NEAT config...")
        config = Config(
            DefaultGenome,
            DefaultReproduction,
            DefaultSpeciesSet,
            DefaultStagnation,
            NEAT_CONFIG,
        )

        print("Generating population...")
        p = Population(config)
        p.add_reporter(StdOutReporter(True))
        p.add_reporter(StatisticsReporter())

        print("Running NEAT game loop")

        winner = None
        try:
            winner = p.run(self._run_neat)
        except TypeError:
            if not self.killswitch:
                raise

        return winner

    def run_random_game_loop(self, num_retries):
        self.game_loop = "random"

        try_number = 1
        while (try_number <= num_retries) if num_retries else True:
            if try_number != 1:
                print()

            print(f"Running random game {try_number}")

            self.launch()
            self.start()

            turns = 0
            start = time.time()
            while not self._is_game_over() and not self._killswitch():
                self.drop_at_random_position()
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

            self.reset(close_window=True)

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
