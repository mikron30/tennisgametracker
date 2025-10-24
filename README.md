# Tennis Ball Tracker

An advanced computer vision system for tracking tennis balls in video footage using dual HSV color filtering.

## Features

- **Dual HSV Configuration**: Separate HSV filters for regular court and behind-net areas to handle different lighting conditions
- **Automatic Net Area Detection**: Mark the net area once, and the system remembers it for future runs
- **Smart Ball Selection**: Uses weighted scoring (distance + size similarity) to select the correct ball among multiple candidates
- **Edge Detection**: Robust handling when balls go off-screen and return
- **Interactive HSV Tuner**: Real-time adjustment of HSV filter parameters with dual-window view
- **Frame-by-Frame Analysis**: Manual stepping through frames with 'D' key for detailed tracking verification
- **Persistent Configuration**: All settings saved to JSON for consistency across runs

## How It Works

### Dual HSV Filtering
The system uses two separate HSV color ranges:
- **Regular Court**: For balls in normal lighting conditions
- **Behind Net**: For balls that appear darker when behind the net

When the ball approaches the net area (within 100px margin), the system checks **both** filters simultaneously and selects the best candidate from all detected contours.

### Ball Selection Algorithm
Instead of just selecting the closest ball, the system uses a weighted score:
```
score = distance + (size_ratio × 30)
```
This ensures that balls with similar size to the previous frame are preferred, reducing false positives.

### Net Area Marking
On first run, the system prompts you to mark 6 points clockwise around the net:
1. Left edge of high net (top-left)
2. Middle of high net (top-middle)
3. Right edge of high net (top-right)
4. Right edge of low net (bottom-right)
5. Middle of low net (bottom-middle)
6. Left edge of low net (bottom-left)

These boundaries are saved permanently to `hsv_config.json`.

## Installation

### Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy

### Install Dependencies
```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage
```bash
python interactive_ball_analyzer.py
```

### Controls
- **D**: Advance to next frame
- **Q**: Quit
- **Click**: Mark ball position when tracking is lost
- **ESC/Q in HSV Tuner**: Close tuner and save settings

### First Run Setup
1. The program will skip to frame 144 (configurable)
2. If net area is not marked, you'll be prompted to mark 6 points
3. The ball will start tracking automatically
4. Press 'D' to advance frame by frame

### Manual Recovery
If tracking is lost:
1. The system will open dual HSV tuner windows
2. Adjust the appropriate window (Regular Court or Behind Net) based on ball location
3. Press 'Q' or ESC to close and continue tracking

## Configuration

### hsv_config.json
The system stores all configuration in `hsv_config.json`:
```json
{
    "regular_court": {
        "h_min": 30, "h_max": 88,
        "s_min": 31, "s_max": 255,
        "v_min": 122, "v_max": 255
    },
    "behind_net": {
        "h_min": 76, "h_max": 91,
        "s_min": 33, "s_max": 146,
        "v_min": 62, "v_max": 152
    },
    "net_area_y_min": 338,
    "net_area_y_max": 532,
    "high_net_y_min": 338,
    "high_net_y_max": 424,
    "low_net_y_min": 422,
    "low_net_y_max": 532
}
```

### Customization
- **Start Frame**: Modify `self.cap.set(cv2.CAP_PROP_POS_FRAMES, 144)` in `process_video()`
- **Initial Ball Position**: Update `first_ball_position = (1977, 320)`
- **Search Radius**: Adjust `search_radius` values for faster/slower balls
- **Net Approach Margin**: Change `net_approach_margin = 100` to adjust when dual filtering activates

## Project Structure

```
tennisgametracker/
├── interactive_ball_analyzer.py    # Main tracking application
├── hsv_config.json                 # Configuration file (auto-generated)
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## Technical Details

### Key Components

1. **BallAnalyzer Class**: Main tracking logic
   - `track_ball_in_frame()`: Core tracking algorithm with dual HSV support
   - `select_hsv_for_position()`: Determines which HSV filter to use
   - `mark_net_area()`: Interactive net area marking
   - `_open_interactive_hsv_tuner()`: Dual-window HSV adjustment

2. **Tracking Logic**:
   - Searches in a radius around last known position (120px normal, 200px at edges)
   - Applies both HSV filters when near net area
   - Evaluates candidates using weighted scoring
   - Handles edge cases (off-screen, occlusion, false positives)

3. **Debug Output**:
   - All contours found in search region
   - Candidate evaluation (position, area, distance, size_ratio, score)
   - Selected contour with source filter
   - HSV values at ball position

## Troubleshooting

### Ball Not Detected
- Open HSV tuner and adjust values for current lighting conditions
- Check if ball is in correct size range (3-80 pixels)
- Verify HSV values at ball position match filter ranges

### False Positives
- System may select incorrect objects with similar HSV values
- Click on correct ball to re-initialize tracking
- Adjust HSV ranges to be more restrictive

### Stuck Tracking
- Ball may be stuck on a static object
- Press 'D' several times and system will eventually detect ball has stopped
- Click on actual ball to resume tracking

## Future Enhancements

- Automatic video processing without manual frame stepping
- Multi-ball tracking support
- Export tracking data to CSV/JSON
- Trajectory prediction for off-screen balls
- Machine learning-based ball detection
- Support for different court surfaces/lighting

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with OpenCV and Python for tennis video analysis.
