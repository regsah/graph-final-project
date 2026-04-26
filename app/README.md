# App Prototype

This folder contains the first GUI prototype for the repository.

## What it is

A framework-free static prototype that demonstrates the agreed product flow:

- model selection,
- title search with partial matches,
- explicit article disambiguation,
- top-20 ranked recommendations in the center,
- and a right-side excerpt panel for the currently selected recommendation.

## What it is not

- not connected to the real Python recommendation pipeline yet,
- not connected to Ollama,
- not generating a learning tree yet,
- and not a final frontend stack decision.

## Files

- `index.html`
  - main page structure
- `styles.css`
  - visual design and layout
- `app.js`
  - mock data + interaction logic

## How to view it

Run the local app server from the repo root:

`python app/server.py`

Then open:

`http://127.0.0.1:8000`

The server hosts both the frontend files and the local API that reads the
repository's cached graph and embedding artifacts.

## Current design decision

The layout follows the clarified direction:

- left panel: search, model selection, disambiguation
- center: ranked recommendation output
- right panel: selected article excerpt / details

## Next likely step

After this version, the next likely step is adding a thin structuring layer for
the future LLM-based learning tree while keeping the retrieval API stable.
