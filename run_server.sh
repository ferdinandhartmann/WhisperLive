# Activate virtual environment
# .\venv_whisperlive\Scripts\Activate.ps1
source venv_whisperlive/bin/activate

# Select which GPU to use (0-based). This makes the process see only GPU #2.
export CUDA_VISIBLE_DEVICES=2

# Run WhisperLive server 
python run_server.py --port 9090 --backend faster_whisper --max_clients 1
