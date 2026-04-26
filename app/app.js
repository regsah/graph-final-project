const ui = {
  appStatus: document.getElementById("app-status"),
  modelSelect: document.getElementById("model-select"),
  modelDescription: document.getElementById("model-description"),
  queryInput: document.getElementById("query-input"),
  searchButton: document.getElementById("search-button"),
  clearButton: document.getElementById("clear-query"),
  suggestionsList: document.getElementById("suggestions-list"),
  disambiguationBlock: document.getElementById("disambiguation-block"),
  disambiguationList: document.getElementById("disambiguation-list"),
  selectedArticleCard: document.getElementById("selected-article-card"),
  selectedArticleTitle: document.getElementById("selected-article-title"),
  selectedArticleMeta: document.getElementById("selected-article-meta"),
  activeModelPill: document.getElementById("active-model-pill"),
  activeSourcePill: document.getElementById("active-source-pill"),
  resultsTitle: document.getElementById("results-title"),
  resultsMeta: document.getElementById("results-meta"),
  rankedViewButton: document.getElementById("ranked-view-button"),
  pathViewButton: document.getElementById("path-view-button"),
  resultsCount: document.getElementById("results-count"),
  resultsList: document.getElementById("results-list"),
  learningPathPanel: document.getElementById("learning-path-panel"),
  emptyState: document.getElementById("empty-state"),
  previewKicker: document.getElementById("preview-kicker"),
  previewTitle: document.getElementById("preview-title"),
  previewLabel: document.getElementById("preview-label"),
  previewRank: document.getElementById("preview-rank"),
  previewScore: document.getElementById("preview-score"),
  previewCanonical: document.getElementById("preview-canonical"),
  previewNodeId: document.getElementById("preview-node-id"),
  previewDegree: document.getElementById("preview-degree"),
  previewLinkPattern: document.getElementById("preview-link-pattern"),
  previewExcerpt: document.getElementById("preview-excerpt")
};

const state = {
  models: [],
  selectedModel: "hybrid-corrected",
  viewMode: "ranked",
  selectedArticle: null,
  selectedResultId: null,
  suggestions: [],
  highlightedSuggestionIndex: -1,
  results: [],
  learningPath: null,
  rawStreamingOutput: "",
  organizeStream: null,
  isLoadingSearch: false,
  isLoadingResults: false,
  isLoadingPath: false
};

let searchDebounce = null;
let latestSearchRequest = 0;

function setStatus(message, tone = "default") {
  ui.appStatus.textContent = message;
  ui.appStatus.dataset.tone = tone;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function init() {
  attachEvents();
  resetPreview();
  updateViewButtons();
  renderCurrentView();

  try {
    const payload = await fetchJson("/api/models");
    state.models = payload.models;
    renderModelOptions();
    updateModelUI();
    await refreshSuggestions("");
    setStatus("Connected to the local WikiCS app server and ready for real recommendations.");
  } catch (error) {
    setStatus(
      "The app could not reach the local server. Start it with `python app/server.py`, then refresh this page.",
      "error"
    );
    ui.resultsMeta.textContent = error.message;
  }
}

function attachEvents() {
  ui.modelSelect.addEventListener("change", async (event) => {
    state.selectedModel = event.target.value;
    updateModelUI();
    if (state.selectedArticle) {
      await runRecommendations(state.selectedArticle.title);
    }
  });

  ui.queryInput.addEventListener("input", () => {
    const value = ui.queryInput.value;
    if (
      state.selectedArticle &&
      normalizeTitle(value) !== normalizeTitle(state.selectedArticle.title)
    ) {
      state.selectedArticle = null;
      state.selectedResultId = null;
      ui.selectedArticleCard.classList.add("is-hidden");
      ui.activeSourcePill.textContent = "None selected";
    }

    ui.disambiguationBlock.classList.add("is-hidden");
    window.clearTimeout(searchDebounce);
    searchDebounce = window.setTimeout(() => {
      refreshSuggestions(value);
    }, 120);
  });

  ui.queryInput.addEventListener("keydown", async (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const direction = event.key === "ArrowDown" ? 1 : -1;
      if (state.suggestions.length) {
        event.preventDefault();
        moveSuggestionHighlight(direction);
      }
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      if (
        state.highlightedSuggestionIndex >= 0 &&
        state.highlightedSuggestionIndex < state.suggestions.length
      ) {
        const article = state.suggestions[state.highlightedSuggestionIndex];
        ui.queryInput.value = article.display_title || formatTitle(article.title);
        selectStartingArticle(article);
        return;
      }
      await handleQuerySubmit();
    }
  });

  ui.searchButton.addEventListener("click", async () => {
    await handleQuerySubmit();
  });

  ui.clearButton.addEventListener("click", resetSelection);
  ui.rankedViewButton.addEventListener("click", async () => {
    await switchView("ranked");
  });
  ui.pathViewButton.addEventListener("click", async () => {
    await switchView("learning-path");
  });
}

function renderModelOptions() {
  ui.modelSelect.innerHTML = state.models
    .map(
      (model) =>
        `<option value="${model.id}" ${model.id === state.selectedModel ? "selected" : ""}>${model.name}</option>`
    )
    .join("");
}

function updateModelUI() {
  const model = state.models.find((item) => item.id === state.selectedModel);
  if (!model) {
    return;
  }
  ui.modelDescription.textContent = model.description;
  ui.activeModelPill.textContent = model.name;
}

async function refreshSuggestions(query) {
  const requestId = ++latestSearchRequest;
  state.isLoadingSearch = true;
  try {
    const payload = await fetchJson(`/api/search?q=${encodeURIComponent(query)}&limit=8`);
    if (requestId !== latestSearchRequest) {
      return;
    }
    state.suggestions = payload.results;
    state.highlightedSuggestionIndex = query.trim() && state.suggestions.length ? 0 : -1;
    renderSuggestions();
  } catch (error) {
    if (requestId !== latestSearchRequest) {
      return;
    }
    state.highlightedSuggestionIndex = -1;
    ui.suggestionsList.innerHTML = `<li class="helper-text">${error.message}</li>`;
  } finally {
    if (requestId === latestSearchRequest) {
      state.isLoadingSearch = false;
    }
  }
}

function renderSuggestions() {
  if (!state.suggestions.length) {
    state.highlightedSuggestionIndex = -1;
    ui.suggestionsList.innerHTML = `<li class="helper-text">No matching article titles found.</li>`;
    return;
  }

  ui.suggestionsList.innerHTML = state.suggestions
    .map(
      (item, index) => `
        <li>
          <button class="suggestion-item ${index === state.highlightedSuggestionIndex ? "is-highlighted" : ""}" type="button" data-title="${escapeHtml(item.title)}">
            ${escapeHtml(item.display_title || formatTitle(item.title))}
            <small>${escapeHtml(item.label)}</small>
          </button>
        </li>
      `
    )
    .join("");

  document.querySelectorAll(".suggestion-item").forEach((button) => {
    button.addEventListener("click", () => {
      const article = state.suggestions.find((item) => item.title === button.dataset.title);
      if (article) {
        ui.queryInput.value = article.display_title || formatTitle(article.title);
        selectStartingArticle(article);
      }
    });
  });
}

function moveSuggestionHighlight(direction) {
  if (!state.suggestions.length) {
    state.highlightedSuggestionIndex = -1;
    return;
  }

  if (state.highlightedSuggestionIndex < 0) {
    state.highlightedSuggestionIndex = direction > 0 ? 0 : state.suggestions.length - 1;
  } else {
    state.highlightedSuggestionIndex =
      (state.highlightedSuggestionIndex + direction + state.suggestions.length) % state.suggestions.length;
  }

  renderSuggestions();
}

async function handleQuerySubmit() {
  const query = ui.queryInput.value.trim();
  if (!query) {
    return;
  }

  const normalizedQuery = normalizeTitle(query);
  const exact = state.suggestions.find(
    (item) => normalizeTitle(item.title) === normalizedQuery
  );
  if (exact) {
    selectStartingArticle(exact);
    return;
  }

  try {
    const payload = await fetchJson(`/api/search?q=${encodeURIComponent(query)}&limit=8`);
    const candidates = payload.results;

    const exactCandidate = candidates.find(
      (item) => normalizeTitle(item.title) === normalizedQuery
    );
    if (exactCandidate) {
      selectStartingArticle(exactCandidate);
      return;
    }

    if (!candidates.length) {
      ui.disambiguationBlock.classList.remove("is-hidden");
      ui.disambiguationList.innerHTML =
        '<li class="helper-text">No matching article titles were found in the dataset.</li>';
      return;
    }

    ui.disambiguationBlock.classList.remove("is-hidden");
    ui.disambiguationList.innerHTML =
      '<li class="helper-text">Matching titles are already listed in Suggestions.</li>';
  } catch (error) {
    ui.disambiguationBlock.classList.remove("is-hidden");
    ui.disambiguationList.innerHTML = `<li class="helper-text">${error.message}</li>`;
  }
}

function selectStartingArticle(article) {
  state.selectedArticle = article;
  state.selectedResultId = null;

  ui.selectedArticleCard.classList.remove("is-hidden");
  ui.selectedArticleTitle.textContent = article.display_title || formatTitle(article.title);
  ui.selectedArticleMeta.textContent = `${article.label} article selected as the recommendation source.`;
  ui.activeSourcePill.textContent = article.display_title || formatTitle(article.title);
  ui.disambiguationBlock.classList.add("is-hidden");

  runRecommendations(article.title);
}

async function runRecommendations(title) {
  state.isLoadingResults = true;
  state.learningPath = null;
  ui.resultsTitle.textContent = `Loading top results for ${title}...`;
  ui.resultsMeta.textContent = "Querying the cached repository artifacts.";
  ui.resultsList.innerHTML = "";
  ui.learningPathPanel.innerHTML = "";
  ui.emptyState.style.display = "none";
  ui.resultsCount.textContent = "Loading";
  resetPreview();

  try {
    const payload = await fetchJson(
      `/api/recommend?title=${encodeURIComponent(title)}&model=${encodeURIComponent(state.selectedModel)}&top_k=20`
    );
    state.selectedArticle = payload.source;
    state.results = payload.results;
    ui.resultsMeta.textContent = `Resolved source article: ${formatTitle(payload.resolved_title)}`;
    renderResults();
    if (state.results.length) {
      selectResult(state.results[0].id);
    }
    if (state.viewMode === "learning-path") {
      await loadLearningPath();
    } else {
      renderCurrentView();
    }
  } catch (error) {
    state.results = [];
    state.learningPath = null;
    ui.resultsTitle.textContent = "Could not load recommendations";
    ui.resultsMeta.textContent = error.message;
    ui.resultsCount.textContent = "0 items";
    ui.resultsList.innerHTML = "";
    ui.learningPathPanel.innerHTML = "";
    ui.emptyState.style.display = "block";
  } finally {
    state.isLoadingResults = false;
  }
}

function renderResults() {
  const sourceTitle = state.selectedArticle
    ? state.selectedArticle.display_title || formatTitle(state.selectedArticle.title)
    : "Start by selecting an article";
  ui.resultsTitle.textContent = state.selectedArticle
    ? `Top 20 for ${sourceTitle}`
    : "Start by selecting an article";
  ui.resultsCount.textContent = `${state.results.length} items`;

  if (!state.results.length) {
    ui.resultsList.innerHTML = "";
    ui.emptyState.style.display = "block";
    return;
  }

  ui.emptyState.style.display = "none";
  ui.resultsList.innerHTML = state.results
    .map(
      (item) => `
        <li>
          <button class="result-card ${item.id === state.selectedResultId ? "is-active" : ""}" type="button" data-result-id="${item.id}">
            <div class="result-rank">${item.rank}</div>
            <div class="result-content">
              <div class="result-meta-row">
                <span class="result-label">${escapeHtml(item.label)}</span>
              </div>
              <h4>${escapeHtml(item.display_title || formatTitle(item.title))}</h4>
              <p>${escapeHtml(item.excerpt)}</p>
            </div>
            <div class="score-chip">${Number(item.score).toFixed(4)}</div>
          </button>
        </li>
      `
    )
    .join("");

  document.querySelectorAll(".result-card").forEach((button) => {
    button.addEventListener("click", () => {
      selectResult(Number(button.dataset.resultId));
    });
  });
}

async function switchView(mode) {
  state.viewMode = mode;
  updateViewButtons();
  renderCurrentView();

  if (mode === "learning-path" && state.selectedArticle && !state.learningPath && state.results.length) {
    await loadLearningPath();
  }
}

function updateViewButtons() {
  ui.rankedViewButton.classList.toggle("is-active", state.viewMode === "ranked");
  ui.pathViewButton.classList.toggle("is-active", state.viewMode === "learning-path");
}

function renderCurrentView() {
  const hasResults = state.results.length > 0;
  const hasPath = Boolean(state.learningPath);

  if (!hasResults) {
    ui.resultsList.classList.add("is-hidden");
    ui.learningPathPanel.classList.add("is-hidden");
    ui.emptyState.style.display = "block";
    return;
  }

  ui.emptyState.style.display = "none";
  if (state.viewMode === "ranked") {
    ui.resultsList.classList.remove("is-hidden");
    ui.learningPathPanel.classList.add("is-hidden");
  } else {
    ui.resultsList.classList.add("is-hidden");
    ui.learningPathPanel.classList.remove("is-hidden");
    if (!hasPath && !state.isLoadingPath) {
      ui.learningPathPanel.innerHTML = `
        <div class="path-state-card">
          <h3>Learning path not generated yet</h3>
          <p>Switching to this view will organize the current ranked results into accordion sections.</p>
        </div>
      `;
    }
  }
}

async function loadLearningPath() {
  if (!state.selectedArticle) {
    return;
  }

  if (state.organizeStream) {
    state.organizeStream.close();
    state.organizeStream = null;
  }

  state.isLoadingPath = true;
  state.rawStreamingOutput = "";
  renderCurrentView();
  ui.learningPathPanel.innerHTML = `
    <div class="ai-output-card">
      <div class="ai-output-header">
        <div>
          <p class="section-label">Temporary Debug</p>
          <h3>AI Organizer Output</h3>
        </div>
        <span class="ai-output-badge">Streaming</span>
      </div>
      <p class="muted-copy">
        This temporary box shows the organizer text as it is being generated. It will disappear once the structured learning path is ready.
      </p>
      <pre id="streaming-ai-output" class="ai-output-pre">Waiting for local model output...</pre>
    </div>
  `;

  await new Promise((resolve) => {
    const url = `/api/organize-stream?title=${encodeURIComponent(state.selectedArticle.title)}&model_id=${encodeURIComponent(state.selectedModel)}&top_k=20`;
    const stream = new EventSource(url);
    state.organizeStream = stream;

    stream.addEventListener("token", (event) => {
      const payload = JSON.parse(event.data);
      state.rawStreamingOutput += payload.chunk || "";
      const outputEl = document.getElementById("streaming-ai-output");
      if (outputEl) {
        outputEl.textContent = state.rawStreamingOutput || "Waiting for local model output...";
      }
    });

    stream.addEventListener("complete", (event) => {
      const payload = JSON.parse(event.data);
      state.learningPath = payload;
      state.isLoadingPath = false;
      state.organizeStream = null;
      stream.close();
      if (payload.warning) {
        ui.resultsMeta.textContent = payload.warning;
      } else if (payload.organizer_model) {
        ui.resultsMeta.textContent = `Learning path organized with ${payload.organizer_model}.`;
      }
      renderLearningPath();
      renderCurrentView();
      resolve();
    });

    stream.addEventListener("fatal", (event) => {
      const payload = JSON.parse(event.data);
      state.learningPath = null;
      state.isLoadingPath = false;
      state.organizeStream = null;
      stream.close();
      ui.learningPathPanel.innerHTML = `
        <div class="path-state-card">
          <h3>Learning path unavailable</h3>
          <p>${escapeHtml(payload.error || "Unknown error.")}</p>
        </div>
      `;
      ui.resultsMeta.textContent = payload.error || "Learning path unavailable.";
      renderCurrentView();
      resolve();
    });

    stream.onerror = () => {
      state.organizeStream = null;
      state.isLoadingPath = false;
      stream.close();
      if (!state.learningPath) {
        ui.learningPathPanel.innerHTML = `
          <div class="path-state-card">
            <h3>Learning path unavailable</h3>
            <p>The organizer stream stopped before completion.</p>
          </div>
        `;
        ui.resultsMeta.textContent = "The organizer stream stopped before completion.";
      }
      renderCurrentView();
      resolve();
    };
  });
}

function renderLearningPath() {
  if (!state.learningPath) {
    return;
  }

  const sections = state.learningPath.sections || [];
  const totalItems = sections.reduce((sum, section) => sum + section.items.length, 0);
  ui.resultsCount.textContent = `${totalItems} items`;

  const sectionsBlock = sections
    .map(
      (section, index) => `
        <details class="path-section" ${index < 2 ? "open" : ""}>
          <summary class="path-section-summary">
            <div>
              <span class="path-section-kicker">${escapeHtml(section.title)}</span>
              <strong>${section.items.length} item${section.items.length === 1 ? "" : "s"}</strong>
            </div>
            <span class="path-section-toggle">Expand</span>
          </summary>
          <p class="path-section-description">${escapeHtml(section.summary)}</p>
          <div class="path-items">
            ${
              section.items.length
                ? section.items
                    .map(
                      (item) => `
                        <button class="path-item ${item.id === state.selectedResultId ? "is-active" : ""}" type="button" data-result-id="${item.id}">
                          <div class="path-item-top">
                            <span class="path-item-rank">#${item.rank}</span>
                            <span class="path-item-label">${escapeHtml(item.label)}</span>
                          </div>
                          <h4>${escapeHtml(item.display_title || formatTitle(item.title))}</h4>
                          <p class="path-item-why">${escapeHtml(item.why)}</p>
                        </button>
                      `
                    )
                    .join("")
                : `<div class="path-empty-section">No items placed in this section for this run.</div>`
            }
          </div>
        </details>
      `
    )
    .join("");

  ui.learningPathPanel.innerHTML = sectionsBlock;

  document.querySelectorAll(".path-item").forEach((button) => {
    button.addEventListener("click", () => {
      const item = getLearningPathItemById(Number(button.dataset.resultId));
      if (item) {
        selectPreviewItem(item);
      }
    });
  });
}

function getLearningPathItemById(resultId) {
  if (!state.learningPath) {
    return null;
  }
  for (const section of state.learningPath.sections || []) {
    const item = section.items.find((entry) => entry.id === resultId);
    if (item) {
      return item;
    }
  }
  return null;
}

function selectPreviewItem(item) {
  state.selectedResultId = item.id;
  fillPreview(item);
  renderResults();
  if (state.viewMode === "learning-path") {
    renderLearningPath();
  }
}

function selectResult(resultId) {
  state.selectedResultId = resultId;
  const item = state.results.find((entry) => entry.id === resultId);
  if (!item) {
    return;
  }
  fillPreview(item);
  renderResults();
  if (state.viewMode === "learning-path") {
    renderLearningPath();
  }
}

function fillPreview(item) {
  ui.previewKicker.textContent = state.selectedArticle
    ? `From ${state.selectedArticle.display_title || formatTitle(state.selectedArticle.title)}`
    : "Selected recommendation";
  ui.previewTitle.textContent = item.display_title || formatTitle(item.title);
  ui.previewLabel.textContent = item.label;
  ui.previewRank.textContent = `Rank ${item.rank}`;
  ui.previewScore.textContent = `Score ${Number(item.score).toFixed(4)}`;
  ui.previewCanonical.textContent = item.canonical_title || item.title;
  ui.previewNodeId.textContent = String(item.id);
  ui.previewDegree.textContent = `in ${item.in_degree} / out ${item.out_degree} / total ${item.total_degree}`;
  ui.previewLinkPattern.textContent = formatLinkPattern(item);
  ui.previewExcerpt.textContent = item.excerpt;
}

function resetPreview() {
  ui.previewKicker.textContent = "Nothing selected";
  ui.previewTitle.textContent = "Choose a recommended article";
  ui.previewLabel.textContent = "Preview";
  ui.previewRank.textContent = "Rank -";
  ui.previewScore.textContent = "Score -";
  ui.previewCanonical.textContent = "-";
  ui.previewNodeId.textContent = "-";
  ui.previewDegree.textContent = "-";
  ui.previewLinkPattern.textContent = "-";
  ui.previewExcerpt.textContent =
    "The right panel will show a compact title, score, and representative article excerpt once a result is selected from the center column.";
}

function resetSelection() {
  state.selectedArticle = null;
  state.selectedResultId = null;
  state.highlightedSuggestionIndex = -1;
  state.results = [];
  state.learningPath = null;
  state.rawStreamingOutput = "";
  state.isLoadingPath = false;
  if (state.organizeStream) {
    state.organizeStream.close();
    state.organizeStream = null;
  }
  ui.queryInput.value = "";
  ui.selectedArticleCard.classList.add("is-hidden");
  ui.activeSourcePill.textContent = "None selected";
  ui.resultsTitle.textContent = "Start by selecting an article";
  ui.resultsMeta.textContent = "The app will use the repository's cached graph and embedding artifacts.";
  ui.resultsCount.textContent = "0 items";
  ui.resultsList.innerHTML = "";
  ui.learningPathPanel.innerHTML = "";
  ui.emptyState.style.display = "block";
  ui.disambiguationBlock.classList.add("is-hidden");
  resetPreview();
  renderCurrentView();
  refreshSuggestions("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTitle(value) {
  return String(value).replaceAll("_", " ");
}

function normalizeTitle(value) {
  return formatTitle(value).replace(/\s+/g, " ").trim().toLowerCase();
}

function formatLinkPattern(item) {
  if (item.linked_from_source && item.linked_to_source) {
    return "bidirectional with source";
  }
  if (item.linked_from_source) {
    return "source links to this article";
  }
  if (item.linked_to_source) {
    return "this article links back to source";
  }
  return "not a direct edge in either direction";
}

init();
