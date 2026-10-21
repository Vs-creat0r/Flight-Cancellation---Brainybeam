# Put this app on Streamlit (free) — 6 steps

You only need 3 files: **app.py**, **requirements.txt**, **flights_dataset.csv**
(The app trains the models itself when it opens — no .pkl files needed.)

## Run on your own computer first (optional test)
1. Put the 3 files in one folder.
2. Open a terminal in that folder and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
   It opens in your browser. First load takes ~15 seconds (it trains the models once).

## Put it online for free
3. Make a **public** repo on GitHub and upload the 3 files
   (github.com → New repository → upload files → Commit).
4. Go to **share.streamlit.io** and sign in with GitHub.
5. Click **New app** → pick your repo → branch **main** → main file **app.py** → **Deploy**.
6. Wait ~3 minutes. You get a public link like `https://yourname-flights.streamlit.app`.
   Share that link with anyone.

## If GitHub blocks the CSV (too big)
The CSV is ~2.3 MB, well under GitHub's 25 MB upload limit, so it will be fine.
