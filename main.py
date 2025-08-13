from train import train_from_config
from dashboard import run_dashboard_app
import threading

if __name__ == "__main__":

    dashboard_thread = threading.Thread(target=run_dashboard_app, daemon=True)

    dashboard_thread.start()

    train_from_config("config.yaml")