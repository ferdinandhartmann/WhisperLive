# Activate virtual environment
# .\venv_whisperlive\Scripts\Activate.ps1
source venv_whisperlive/bin/activate

# Run WhisperLive server 
python run_server.py --port 9090 --backend faster_whisper --max_clients 1
