# chatty-bot 🤖

A tiny, silly chatbot built from scratch in **Keras**: a word-level
**seq2seq LSTM with attention** and a short conversation memory.
It speaks English and Russian: write in either and it answers in the same one.

```
🧑 hi                        🤖 hi there, human!
🧑 my name is vlad           🤖 nice to meet you, vlad!
🧑 i love pizza              🤖 pizza? yummy!
🧑 what's my name ?          🤖 you are vlad, of course!
🧑 tell me a joke            🤖 why did the robot go on holiday? to recharge its batteries!
🧑 another joke              🤖 what do you call a sleeping bull? a bulldozer!
🧑 what do i like to eat ?   🤖 hmm... i don't remember. tell me again?   ← pizza fell out of its 3-turn memory
```

## Run

```sh
uv sync
uv run python -m ipykernel install --user --name chatty-bot-keras --display-name "Python (chatty-bot-keras)"
uv run jupyter lab chatty_bot.ipynb
```

Run all cells. Training takes about 20 minutes on a laptop CPU. Then chat with
`bot.say("...")` or the interactive loop at the end.

## Telegram

`telegram_bot.py` lets the same model answer in Telegram. Each chat has its
own 3-exchange memory. It runs on your machine and only answers while it runs.

1. Make a bot with `@BotFather` and copy its token.
2. Create `.env` (git ignores it):
   ```
   TELEGRAM_BOT_TOKEN=<token>
   ALLOWED_CHAT_IDS=<your chat id>,<a friend's chat id>
   ```
3. `uv run --env-file .env python telegram_bot.py`

The first start trains the model (about 20 minutes) and saves it to `model/`;
later starts load it in seconds. Chats not in `ALLOWED_CHAT_IDS` get no answer,
and their ID is printed so you can add them.

## How it works

Everything is in [`chatty_bot.ipynb`](chatty_bot.ipynb).

- **Data.** About 10,000 synthetic conversations, half English, half Russian: greetings, jokes, telling the
  bot your name/food/pet/city, then asking about them later. The right answer
  to "what's my name?" is computed from what's inside the bot's memory window,
  so the data teaches it both to remember and to admit when it has forgotten.
- **Memory.** The model's input is the last **3 exchanges + your new message**,
  not just your message.
- **Model.** Embedding → LSTM encoder → LSTM decoder → dot-product attention
  over the context → Dense → softmax over a ~600-word vocabulary.
  Trained with teacher forcing; replies are generated greedily, word by word.
- **Results** on unseen conversations (exact match of the whole reply):

  | chit-chat | jokes in order | acknowledging a fact | recalling a fact | "i don't remember" |
  |---|---|---|---|---|
  | 100% | 100% | 100% | 100% | 100% |

  The notebook also draws an attention heatmap. When the bot writes your
  name, attention jumps to where you said it a few turns earlier.

## Limits (it's silly on purpose)

- It only knows the ~600 words it was trained on (English and Russian). Unknown words become `<unk>`,
  so names, foods, pets and cities must come from its lists (`print(NAMES)`).
- Its whole personality is ~40 canned replies. It is a pattern learner, not a
  language model.
- Memory is exactly 3 exchanges.
