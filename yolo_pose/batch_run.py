# batch_run.py — queue multiple pose_velocity.py + analysis.py runs across videos/params

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import json
import curses
import subprocess

# Must stay in sync with pose_velocity.py's MODEL_CODES keys and --yolo-version choices.
MODEL_CHOICES = ["nano", "small", "medium", "large", "xlarge"]
YOLO_VERSION_CHOICES = ["8", "11", "26"]
VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_POSE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------
# Video discovery
# ---------------------------------------------------------
def find_videos(videos_dir):
    matches = []
    for root, _dirs, files in os.walk(videos_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                matches.append(os.path.join(root, f))
    return sorted(matches)


def resolve_videos_dir():
    default_dir = os.path.join(PROJECT_ROOT, "videos")
    videos = find_videos(default_dir) if os.path.isdir(default_dir) else []
    if videos:
        return default_dir, videos

    while True:
        entered = input(f"Videos directory not found at {default_dir}. Enter a videos directory: ").strip()
        if not entered:
            continue
        entered = os.path.abspath(os.path.expanduser(entered))
        if not os.path.isdir(entered):
            print(f"[ERROR] {entered} is not a directory.")
            continue
        videos = find_videos(entered)
        if not videos:
            print(f"[ERROR] No video files ({', '.join(VIDEO_EXTENSIONS)}) found under {entered}.")
            continue
        return entered, videos


# ---------------------------------------------------------
# Video selection — curses checkbox list, with a plain-text fallback
# ---------------------------------------------------------
def select_videos_curses(labels):
    selected = [False] * len(labels)
    cursor = 0
    confirmed = False

    def safe_addstr(stdscr, y, x, text, attr=0):
        max_y, max_x = stdscr.getmaxyx()
        if y < 0 or y >= max_y or x >= max_x:
            return
        # Truncate so the string never touches the terminal's bottom-right cell,
        # which curses can't safely write to without raising curses.error.
        room = max_x - x - (1 if y == max_y - 1 else 0)
        if room <= 0:
            return
        stdscr.addstr(y, x, text[:room], attr)

    def draw(stdscr):
        nonlocal cursor, confirmed
        curses.curs_set(0)
        stdscr.keypad(True)
        top = 0
        while True:
            max_y, _max_x = stdscr.getmaxyx()
            visible_rows = max(1, max_y - 2)  # row 0 is the header, row 1 is blank
            if cursor < top:
                top = cursor
            elif cursor >= top + visible_rows:
                top = cursor - visible_rows + 1

            stdscr.erase()
            safe_addstr(stdscr, 0, 0, "Select videos to queue  (space=toggle, a=all/none, enter=confirm, q=cancel)")
            for row, i in enumerate(range(top, min(top + visible_rows, len(labels)))):
                box = "[x]" if selected[i] else "[ ]"
                prefix = ">" if i == cursor else " "
                line = f"{prefix} {box} {labels[i]}"
                attr = curses.A_REVERSE if i == cursor else 0
                safe_addstr(stdscr, row + 2, 0, line, attr)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(labels)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(labels)
            elif key == ord(" "):
                selected[cursor] = not selected[cursor]
            elif key == ord("a"):
                target = not all(selected)
                for i in range(len(selected)):
                    selected[i] = target
            elif key in (10, 13, curses.KEY_ENTER):
                confirmed = True
                return
            elif key in (27, ord("q")):
                confirmed = False
                return

    curses.wrapper(draw)
    return confirmed, selected


def select_videos_plaintext(labels):
    print("\nAvailable videos:")
    for i, label in enumerate(labels, start=1):
        print(f"  {i}. {label}")

    while True:
        raw = input("Select videos to queue (e.g. 1,3,5 or 1-4 or 'all'): ").strip()
        if not raw:
            continue
        if raw.lower() == "all":
            return list(range(len(labels)))

        indices = set()
        valid = True
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                bounds = part.split("-")
                if len(bounds) != 2 or not all(b.strip().isdigit() for b in bounds):
                    valid = False
                    break
                lo, hi = int(bounds[0]), int(bounds[1])
                if lo > hi:
                    valid = False
                    break
                indices.update(range(lo, hi + 1))
            elif part.isdigit():
                indices.add(int(part))
            else:
                valid = False
                break

        if not valid or not indices or any(i < 1 or i > len(labels) for i in indices):
            print("[ERROR] Invalid selection, try again.")
            continue
        return sorted(i - 1 for i in indices)


def check_name_collisions(selected):
    # pose_velocity.py's output dir is keyed only by video basename (no subfolder),
    # so same-named videos in different folders would silently overwrite each other.
    by_name = {}
    for v in selected:
        name = os.path.splitext(os.path.basename(v))[0].lower()
        by_name.setdefault(name, []).append(v)

    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if not collisions:
        return

    print("[ERROR] Selected videos have colliding filenames — each would write to the same "
          "pose_results/<name>/ directory and overwrite each other:")
    for name, paths in collisions.items():
        print(f"  {name}:")
        for p in paths:
            print(f"    - {p}")
    sys.exit("Rename one of each colliding pair (or deselect it) and re-run.")


def select_videos(videos, videos_dir):
    labels = [os.path.relpath(v, videos_dir) for v in videos]

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            confirmed, selected_mask = select_videos_curses(labels)
            if not confirmed:
                sys.exit("[INFO] Batch run cancelled.")
            chosen = [videos[i] for i, s in enumerate(selected_mask) if s]
            if chosen:
                return chosen
            print("[WARN] No videos selected via checklist; falling back to text prompt.")
        except curses.error:
            print("[WARN] Interactive checklist unavailable in this terminal; falling back to text prompt.")

    chosen_indices = select_videos_plaintext(labels)
    return [videos[i] for i in chosen_indices]


# ---------------------------------------------------------
# Parameter prompts
# ---------------------------------------------------------
def prompt_choice(prompt, choices, default):
    choice_str = "/".join(c if c != default else f"[{c}]" for c in choices)
    while True:
        raw = input(f"{prompt} ({choice_str}): ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"[ERROR] Must be one of: {', '.join(choices)}")


def prompt_fps(default_label="native"):
    while True:
        raw = input(f"FPS to downsample inference to (blank = {default_label}): ").strip()
        if not raw:
            return None
        try:
            fps = float(raw)
        except ValueError:
            print("[ERROR] Enter a positive number or leave blank.")
            continue
        if fps <= 0:
            print("[ERROR] --fps must be a positive number.")
            continue
        return fps


def prompt_bool(prompt, default=False):
    default_label = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} ({default_label}): ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("[ERROR] Enter y or n.")


def prompt_params():
    print("\nRun parameters (applied to every selected video):")
    model = prompt_choice("Model size", MODEL_CHOICES, "large")
    yolo_version = prompt_choice("YOLO version", YOLO_VERSION_CHOICES, "26")
    fps = prompt_fps()
    clean = prompt_bool("Delete rendered images after each run (--clean)", default=False)
    return {"model": model, "yolo_version": yolo_version, "fps": fps, "clean": clean}


# ---------------------------------------------------------
# Job execution
# ---------------------------------------------------------
def track_path_for(video_path, params):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    fps_suffix = f"_fps{params['fps']:g}" if params["fps"] is not None else ""
    output_dir = os.path.join(
        YOLO_POSE_DIR, "pose_results", video_name,
        f"yolo{params['yolo_version']}_{params['model']}{fps_suffix}",
    )
    return os.path.join(output_dir, "barbell_track.json")


def run_step(cmd, description):
    print(f"[BATCH] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=YOLO_POSE_DIR)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed (exit code {result.returncode}).")
        return False
    return True


def validate_track(track_path):
    """Sanity-check the track pose_velocity.py wrote before handing it to analysis.py.

    pose_velocity.py can exit 0 while having skipped video reconstruction (e.g. zero
    rendered frames) or having detected no person in any frame, so a non-zero return
    code alone isn't enough to trust the track is usable.
    """
    if not os.path.exists(track_path):
        return (f"expected track file not found at {track_path} — pose_velocity.py's output "
                "directory naming may have diverged from batch_run.py's track_path_for()")

    with open(track_path) as f:
        track = json.load(f)

    if not track:
        return f"{track_path} is empty — pose_velocity.py produced no frames for this video"

    if not any(entry.get("center") is not None for entry in track):
        return f"{track_path} has no detections (every frame's center is null) — no person was found in this video"

    return None


def run_job(video_path, params):
    cmd = [sys.executable, "pose_velocity.py", params["model"],
           "--yolo-version", params["yolo_version"], "--video", video_path]
    if params["fps"] is not None:
        cmd += ["--fps", str(params["fps"])]
    if params["clean"]:
        cmd.append("--clean")

    if not run_step(cmd, f"pose_velocity.py for {video_path}"):
        return False

    track_path = track_path_for(video_path, params)
    problem = validate_track(track_path)
    if problem:
        print(f"[ERROR] pose_velocity.py for {video_path} produced an unusable track: {problem}")
        return False

    analysis_cmd = [sys.executable, "analysis.py", "--track", track_path]
    if not run_step(analysis_cmd, f"analysis.py for {video_path}"):
        return False

    print(f"[BATCH] Done: {os.path.dirname(track_path)}")
    return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    videos_dir, videos = resolve_videos_dir()
    selected = select_videos(videos, videos_dir)
    if not selected:
        sys.exit("[INFO] No videos selected. Exiting.")
    check_name_collisions(selected)

    params = prompt_params()

    print("\n[BATCH] Queue summary:")
    for v in selected:
        print(f"  - {os.path.relpath(v, videos_dir)}")
    fps_label = f"{params['fps']:g}" if params["fps"] is not None else "native"
    print(f"  model={params['model']}  yolo-version={params['yolo_version']}  "
          f"fps={fps_label}  clean={params['clean']}")

    if not prompt_bool(f"\nRun {len(selected)} job(s) now?", default=True):
        sys.exit("[INFO] Batch run cancelled.")

    completed = []
    for i, video_path in enumerate(selected, start=1):
        print(f"\n[BATCH] Job {i}/{len(selected)}: {video_path}")
        if not run_job(os.path.abspath(video_path), params):
            print(f"[BATCH] Stopping batch after failure on job {i}/{len(selected)}.")
            break
        completed.append(video_path)
        print(f"[BATCH] Completed {i}/{len(selected)}")

    print(f"\n[BATCH] Finished: {len(completed)}/{len(selected)} job(s) completed successfully.")
    for v in completed:
        track_path = track_path_for(os.path.abspath(v), params)
        print(f"  - {v} -> {os.path.dirname(track_path)}")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        sys.exit("\n[ERROR] No input available (stdin closed) — this script requires an interactive terminal.")
    except KeyboardInterrupt:
        sys.exit("\n[INFO] Batch run cancelled.")
