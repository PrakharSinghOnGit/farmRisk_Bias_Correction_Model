import os
import json

STATS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stats.json")

def update_stats(pipeline_name, duration):
    """
    Updates the running stats (count, last run time, running average)
    for a given pipeline ('forecast' or 'soil_moisture') and prints the log.
    """
    stats = {
        "forecast": {"count": 0, "last_time": 0.0, "avg_time": 0.0},
        "soil_moisture": {"count": 0, "last_time": 0.0, "avg_time": 0.0}
    }
    
    # Load existing stats if file exists
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
                # Ensure structure matches
                for key in stats:
                    if key in loaded:
                        stats[key].update(loaded[key])
        except Exception as e:
            print(f"[STATS WARNING] Could not read stats file: {e}")

    p_stats = stats[pipeline_name]
    old_count = p_stats["count"]
    old_avg = p_stats["avg_time"]

    new_count = old_count + 1
    new_avg = (old_avg * old_count + duration) / new_count

    p_stats["count"] = new_count
    p_stats["last_time"] = duration
    p_stats["avg_time"] = new_avg

    stats[pipeline_name] = p_stats

    # Write updated stats back to file
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[STATS WARNING] Could not write to stats file: {e}")

    # Print log
    print(f"[{pipeline_name.upper()} STATS] Count: {new_count} | Last Time: {duration:.2f}s | Avg Time: {new_avg:.2f}s")
