# Titani

Consumers Python per il sistema Ermete/Iris, gestiti con `uv`.

## Consumer disponibili

- **Teia**: riceve snapshot (`frame_available`), scarica il file, lo invia a una API LLM e pubblica sul websocket una descrizione testuale (`snapshot_description`).
- **Ceo**: riceve audio WebRTC e lo elabora con una pipeline VAD pensata per `mlx-audio` (con fallback RMS).
- **Crio**: riceve audio WebRTC e fa loopback basilare della traccia (pipeline torch/torchaudio placeholder).

## Setup rapido

```bash
uv sync
```

Esecuzione:

```bash
uv run teia
uv run ceo
uv run crio
```

## Variabili ambiente principali

- `ERMETE_WS` (default: `wss://alveare.metallize.it:8080/v1/ws?role=consumer`)
- `ERMETE_HTTP_BASE` (default: `https://alveare.metallize.it:8080`)
- `ERMETE_PSK_HEADER` (default: `X-Ermete-PSK`)
- `ERMETE_PSK`
- `FRAMES_OUT_DIR` (default: `./frames_downloaded`)

Solo **Teia**:

- `LLM_API_KEY` (oppure `OPENAI_API_KEY`)
- `LLM_MODEL` (default: `gpt-4.1-mini`)
- `LLM_BASE_URL` (default: `https://api.openai.com/v1`)
