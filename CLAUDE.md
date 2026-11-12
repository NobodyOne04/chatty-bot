# chatty-bot

A tiny Keras seq2seq LSTM chatbot with a 3-exchange memory, trained on synthetic chats.

## Commands
```sh
uv sync
uv run jupyter lab chatty_bot.ipynb                     # train and play in the notebook
uv run --env-file .env python telegram_bot.py           # answer in Telegram
```

## Layout
- `chatty_bot.ipynb`: everything about the model: data, vocab, model, training, inference
- `telegram_bot.py`: runs the notebook's code cells (up to the one defining `decoder_model`)
  and answers Telegram chats; first run trains and saves `model/`, later runs load it
- `model/`: trained weights + vocab, git-ignored, re-created by training
- `.env`: `TELEGRAM_BOT_TOKEN`, `ALLOWED_CHAT_IDS` (comma-separated), git-ignored

## Conventions
- The model code lives only in the notebook. `telegram_bot.py` finds cells by their text
  (`model.fit(`, `model.save(`, `decoder_model =`): keep those markers when editing it.
- Notebook cells before `decoder_model` must run headless (no `display()`, no `input()`).
