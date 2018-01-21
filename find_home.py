from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
# game.load_config("../../scenarios/basic.cfg")

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game.load_config("/home/arnaud/ViZDoom/scenarios/my_way_home.cfg")

# Sets map to start (scenario .wad files can contain many maps).
# game.set_doom_map("map01")
game.set_mode(Mode.PLAYER)

# Sets resolution. Default is 320X240
game.set_screen_resolution(ScreenResolution.RES_320X240)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.RGB24)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of the current episode/level.
game.set_automap_buffer_enabled(True)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# # Adds buttons that will be allowed.
# game.add_available_button(Button.MOVE_LEFT)
# game.add_available_button(Button.MOVE_RIGHT)
# game.add_available_button(Button.ATTACK)
#
# # Adds game variables that will be included in state.
# game.add_available_game_variable(GameVariable.AMMO2)
#
# # Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(800)
#
# # Makes episodes start after 10 tics (~after raising the weapon)
# game.set_episode_start_time(10)
#
# # Makes the window appear (turned on by default)
# game.set_window_visible(True)
#
# # Turns on the sound. (turned off by default)
# game.set_sound_enabled(True)
#
# # Sets the livin reward (for each move) to -1
# game.set_living_reward(-1)
#
# # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
# game.set_mode(Mode.PLAYER)

# Enables engine output to console.
#game.set_console_enabled(True)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()

# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.
actions = [[True, False, False], [False, True, False], [False, False, True]]

# Run this many episodes
episodes = 10

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        # Makes a random action and get remember reward.
        r = game.make_action(choice(actions))

        # Makes a "prolonged" action and skip frames:
        # skiprate = 4
        # r = game.make_action(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        # Prints state's game variables and reward.
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    # Check how the episode went.
    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
