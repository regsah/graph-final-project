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
  resultsCount: document.getElementById("results-count"),
  resultsList: document.getElementById("results-list"),
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
  selectedArticle: null,
  selectedResultId: null,
  suggestions: [],
  highlightedSuggestionIndex: -1,
  results: [],
  isLoadingSearch: false,
  isLoadingResults: false
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

async function init() {
  attachEvents();
  resetPreview();

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
    if (state.selectedArticle && value !== state.selectedArticle.title) {
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
  ui.resultsTitle.textContent = `Loading top results for ${title}...`;
  ui.resultsMeta.textContent = "Querying the cached repository artifacts.";
  ui.resultsList.innerHTML = "";
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
  } catch (error) {
    state.results = [];
    ui.resultsTitle.textContent = "Could not load recommendations";
    ui.resultsMeta.textContent = error.message;
    ui.resultsCount.textContent = "0 items";
    ui.resultsList.innerHTML = "";
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

function selectResult(resultId) {
  state.selectedResultId = resultId;
  const item = state.results.find((entry) => entry.id === resultId);
  if (!item) {
    return;
  }

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

  renderResults();
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
  ui.queryInput.value = "";
  ui.selectedArticleCard.classList.add("is-hidden");
  ui.activeSourcePill.textContent = "None selected";
  ui.resultsTitle.textContent = "Start by selecting an article";
  ui.resultsMeta.textContent = "The app will use the repository's cached graph and embedding artifacts.";
  ui.resultsCount.textContent = "0 items";
  ui.resultsList.innerHTML = "";
  ui.emptyState.style.display = "block";
  ui.disambiguationBlock.classList.add("is-hidden");
  resetPreview();
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
