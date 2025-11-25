from pathlib import Path
from datetime import datetime
import json


class RunManager:
    """Manages run folders and config saving."""
    
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.run_dir = self._create_run_folder()
        
    def _create_run_folder(self) -> Path:
        """Create a new run_x folder with incremented number."""
        existing_runs = list(self.base_dir.glob("run_*"))
        
        if not existing_runs:
            run_number = 1
        else:
            # Extract numbers from existing run folders
            numbers = []
            for run_path in existing_runs:
                try:
                    num = int(run_path.name.split("_")[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    continue
            
            run_number = max(numbers) + 1 if numbers else 1
        
        run_dir = self.base_dir / f"run_{run_number}"
        run_dir.mkdir(exist_ok=True)

        prediction_dir = run_dir / "prediction_samples"
        prediction_dir.mkdir(exist_ok=True)
        
        print(f"Created run folder: {run_dir}")
        return run_dir
    
    def save_config(self, config: dict):
        """Save configuration to a text file."""
        config_path = self.run_dir / "config.txt"
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in config.items():
                f.write(f"{key:30s}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Configuration saved to: {config_path}")
        
        # Also save as JSON for programmatic access
        json_path = self.run_dir / "config.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the run directory."""
        return self.run_dir / filename
    
    def __str__(self):
        return str(self.run_dir)
