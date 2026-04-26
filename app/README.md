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

Open `app/index.html` in a browser.

Because this first pass is static and dependency-free, no install step is required.

## Current design decision

The layout follows the clarified direction:

- left panel: search, model selection, disambiguation
- center: ranked recommendation output
- right panel: selected article excerpt / details

## Next likely step

After this prototype is approved, the next step is to replace the mock data in
`app.js` with a real API or local bridge to the repository's recommendation logic.
