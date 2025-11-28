# gui.py

import sys
import pygame
import numpy as np

from environment import ShoverWorldEnv


MAX_WINDOW_SIZE = 625
HUD_HEIGHT = 100
FPS = 30

COLOR_BG = (20, 20, 20)
COLOR_GRID_BG = (40, 40, 40)
COLOR_EMPTY = (70, 70, 70)
COLOR_BOX = (220, 190, 80)
COLOR_BARRIER = (130, 130, 130)
COLOR_LAVA = (210, 60, 60)
COLOR_AGENT = (80, 210, 130)
COLOR_TEXT = (235, 235, 235)
COLOR_INVALID = (255, 120, 120)
COLOR_BORDER = (25, 25, 25)


def compute_cell_size(n_rows: int, n_cols: int,
                      max_window: int = MAX_WINDOW_SIZE,
                      hud_height: int = HUD_HEIGHT) -> int:
    """
    Compute size : whole grid + HUD fits inside max_window -> minimize
    """
    max_grid_h = max_window - hud_height
    max_grid_w = max_window

    size_h = max_grid_h // max(n_rows, 1)
    size_w = max_grid_w // max(n_cols, 1)
    size = min(size_h, size_w)

    # Avoid too tiny cells
    return max(24, size)


def build_action_from_key(env: ShoverWorldEnv, key: int):
    # Movement
    if key in (pygame.K_UP, pygame.K_w):
        action_type = 1  # Up
        dr, dc = -1, 0
    elif key in (pygame.K_RIGHT, pygame.K_d):
        action_type = 2  # Right
        dr, dc = 0, 1
    elif key in (pygame.K_DOWN, pygame.K_s):
        action_type = 3  # Down
        dr, dc = 1, 0
    elif key in (pygame.K_LEFT, pygame.K_a):
        action_type = 4  # Left
        dr, dc = 0, -1

    # Special actions: 5, 6
    elif key == pygame.K_b:
        action_type = 5  # Barrier Maker
        dr, dc = 0, 0
    elif key == pygame.K_h:
        action_type = 6  # Hellify
        dr, dc = 0, 0
    else:
        return None

    ar, ac = env.agent_pos


    if action_type in (1, 2, 3, 4):
        tr = max(0, min(env.n_rows - 1, ar + dr))
        tc = max(0, min(env.n_cols - 1, ac + dc))
        target_pos = (tr, tc)
    else:
        target_pos = (ar, ac)

    return (target_pos, action_type)


def draw_grid(surface, env: ShoverWorldEnv, info: dict,
              font, small_font, cell_size: int):
    """
    Draw HUD + grid + agent.
    """
    surface.fill(COLOR_BG)

    grid = env.grid
    n_rows, n_cols = grid.shape

    # **** HUD ****
    hud_rect = pygame.Rect(0, 0, n_cols * cell_size, HUD_HEIGHT)
    pygame.draw.rect(surface, COLOR_GRID_BG, hud_rect)

    timestep = info.get("timestep", 0)
    stamina = info.get("stamina", 0.0)
    num_boxes = info.get("number_of_boxes", 0)
    last_valid = info.get("last_action_valid", True)
    chain_k = info.get("chain_length_k", 0)
    lava_k = info.get("lava_destroyed_this_step", 0)
    num_destroyed = info.get("number_destroyed", 0)


    line1 = (
        f"Timestep: {timestep}   "
        f"Stamina: {stamina:.1f}   "
        f"Boxes: {num_boxes}   "
        f"Destroyed: {num_destroyed}"
    )
    line2 = (
        f"Last valid: {last_valid}   "
        f"Chain length: {chain_k}   "
        f"Lava destroyed (this step): {lava_k}"
    )
    line3 = "Controls: Arrows/WASD=move, B=Barrier, H=Hellify, R=reset, Q=quit, Click=move agent"

    text1 = font.render(line1, True, COLOR_TEXT)
    text2_color = COLOR_TEXT if last_valid else COLOR_INVALID
    text2 = font.render(line2, True, text2_color)
    text3 = small_font.render(line3, True, COLOR_TEXT)

    surface.blit(text1, (10, 8))
    surface.blit(text2, (10, 35))
    surface.blit(text3, (10, 65))

    # **** Grid cells ****
    offset_y = HUD_HEIGHT

    for r in range(n_rows):
        for c in range(n_cols):
            val = int(grid[r, c])

            if val == env.EMPTY_VALUE:
                color = COLOR_EMPTY
            elif val == env.BARRIER_VALUE:
                color = COLOR_BARRIER
            elif val == env.LAVA_VALUE:
                color = COLOR_LAVA
            elif env.BOX_MIN <= val <= env.BOX_MAX:
                color = COLOR_BOX
            else:
                color = (150, 60, 200)  # unknown

            cell_rect = pygame.Rect(
                c * cell_size,
                offset_y + r * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(surface, color, cell_rect)
            pygame.draw.rect(surface, COLOR_BORDER, cell_rect, 1)

    # **** Agent ****
    ar, ac = env.agent_pos
    center_x = ac * cell_size + cell_size // 2
    center_y = offset_y + ar * cell_size + cell_size // 2

    pygame.draw.circle(surface, COLOR_AGENT, (center_x, center_y),
                       cell_size // 3)
    agent_glyph = font.render("A", True, (0, 0, 0))
    glyph_rect = agent_glyph.get_rect(center=(center_x, center_y))
    surface.blit(agent_glyph, glyph_rect)


def main():
    env = ShoverWorldEnv(
        render_mode="human",
        n_rows=6,
        n_cols=6,
        initial_stamina=200.0,
        initial_force=10.0,
        unit_force=2.0,
        perf_sq_initial_age=5,
        max_timestep=400,
        map_path=None, 
        # seed=0     #random
    )

    obs, info = env.reset()

    n_rows, n_cols = env.grid.shape
    cell_size = compute_cell_size(n_rows, n_cols)
    width = n_cols * cell_size
    height = HUD_HEIGHT + n_rows * cell_size


    pygame.init()
    pygame.display.set_caption("Shover-World GUI")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # Fonts
    font = pygame.font.SysFont("consolas", 12)
    small_font = pygame.font.SysFont("consolas", 10)

    running = True
    done = False

    while running:
        # ************ Events ****************
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_r:
                    # reset episode
                    obs, info = env.reset()
                    done = False

                else:
                    if not done:
                        action = build_action_from_key(env, event.key)
                        if action is not None:
                            obs, reward, done, info = env.step(action)

            # Mouse -> teleport agent (not to barriers)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # ignore HUD clicks
                if my >= HUD_HEIGHT:
                    grid_y = my - HUD_HEIGHT
                    row = grid_y // cell_size
                    col = mx // cell_size

                    if 0 <= row < env.n_rows and 0 <= col < env.n_cols:
                        if env.grid[row, col] != env.BARRIER_VALUE:
                            env.agent_pos = (row, col)
                            env.previous_selected_position = (row, col)

        # **** Drawing ****
        draw_grid(screen, env, info, font, small_font, cell_size)

        # "DONE" message
        if done:
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            msg = font.render("Episode finished - press R to reset", True, COLOR_TEXT)
            rect = msg.get_rect(center=(width // 2, height // 2))
            screen.blit(msg, rect)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
