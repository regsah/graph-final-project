const models = [
  {
    id: "bge-only",
    name: "BGE-M3 Only",
    description: "Semantic-only retrieval using article content similarity."
  },
  {
    id: "n2v-uncorrected",
    name: "Node2Vec (uncorrected)",
    description: "Pure graph structure without degree damping."
  },
  {
    id: "n2v-corrected",
    name: "Node2Vec (corrected)",
    description: "Pure structure with popularity damping from the power-law correction."
  },
  {
    id: "hybrid-corrected",
    name: "BGE-M3 + Node2Vec (corrected)",
    description: "Hybrid recommendation model blending semantics with corrected structural signal."
  },
  {
    id: "gcn-bge-fusion",
    name: "GCN + BGE-M3 (rank fusion)",
    description: "Ranking-level fusion between graph-learned and text-based similarity."
  }
];

const articles = {
  Programming_language: {
    title: "Programming_language",
    category: "Languages",
    excerpt:
      "A programming language is a formal system for expressing computation. In practice it acts as both an engineering tool and a conceptual layer, shaping how programmers think about abstraction, execution, and correctness."
  },
  Compiler: {
    title: "Compiler",
    category: "Languages",
    excerpt:
      "A compiler translates source code from a high-level programming language into another target language, often machine code, while also performing analysis and optimization over program structure."
  },
  Interpreter: {
    title: "Interpreter_(computing)",
    category: "Languages",
    excerpt:
      "An interpreter executes a program directly, usually statement by statement or expression by expression, making it central to interactive environments and dynamic language workflows."
  },
  Programming_paradigm: {
    title: "Programming_paradigm",
    category: "Languages",
    excerpt:
      "Programming paradigms define styles of computation such as imperative, functional, logic, or object-oriented programming, and they influence both syntax and conceptual problem decomposition."
  },
  Type_system: {
    title: "Type_system",
    category: "Languages",
    excerpt:
      "A type system constrains what operations can be applied to values, making it one of the main mechanisms for expressing safety, abstraction, and language design tradeoffs."
  },
  Syntax_programming: {
    title: "Syntax_(programming_languages)",
    category: "Languages",
    excerpt:
      "Programming language syntax defines the legal structure of programs, including expression forms, declarations, and composition rules that make code readable and compilable."
  },
  Natural_language_processing: {
    title: "Natural_language_processing",
    category: "AI",
    excerpt:
      "Natural language processing studies how computers represent, analyze, and generate human language, combining linguistics, machine learning, and statistical modeling."
  },
  Machine_translation: {
    title: "Machine_translation",
    category: "AI",
    excerpt:
      "Machine translation focuses on transforming text from one language into another while preserving meaning, structure, and context across linguistic systems."
  },
  Computational_linguistics: {
    title: "Computational_linguistics",
    category: "AI",
    excerpt:
      "Computational linguistics connects formal language analysis with computation, offering theoretical and practical foundations for NLP systems."
  },
  Language_model: {
    title: "Language_model",
    category: "AI",
    excerpt:
      "A language model estimates the probability of token sequences and underpins text prediction, generation, retrieval support, and many contemporary NLP systems."
  },
  Question_answering: {
    title: "Question_answering",
    category: "AI",
    excerpt:
      "Question answering systems are designed to retrieve or generate answers to user queries, often combining retrieval, reasoning, and language understanding."
  },
  Operating_system: {
    title: "Operating_system",
    category: "Systems",
    excerpt:
      "An operating system manages hardware resources and provides services for software execution, acting as the core coordination layer between applications and the machine."
  },
  Process_computing: {
    title: "Process_(computing)",
    category: "Systems",
    excerpt:
      "A process is a running program instance with its own state, memory context, and execution flow, making it a basic unit of scheduling and isolation in operating systems."
  },
  Computer_multitasking: {
    title: "Computer_multitasking",
    category: "Systems",
    excerpt:
      "Computer multitasking refers to the management of multiple tasks over time, usually through scheduling policies that maximize responsiveness or throughput."
  },
  Real_time_os: {
    title: "Real-time_operating_system",
    category: "Systems",
    excerpt:
      "A real-time operating system is engineered to meet timing guarantees, making it critical in embedded, industrial, and safety-sensitive computing environments."
  },
  Database: {
    title: "Database",
    category: "Data",
    excerpt:
      "A database is an organized collection of data designed for efficient storage, retrieval, and update, usually with a data model that structures access patterns."
  },
  Relational_database: {
    title: "Relational_database",
    category: "Data",
    excerpt:
      "A relational database stores information in tables with well-defined relationships, making it one of the dominant paradigms in data management and query processing."
  },
  Database_design: {
    title: "Database_design",
    category: "Data",
    excerpt:
      "Database design covers schema structure, normalization, constraints, and workload-aware modeling decisions that shape correctness and performance."
  },
  Cloud_computing: {
    title: "Cloud_computing",
    category: "Distributed",
    excerpt:
      "Cloud computing provides on-demand computing resources over networks, emphasizing elasticity, service abstraction, distributed infrastructure, and operational scale."
  },
  Cloud_storage: {
    title: "Cloud_storage",
    category: "Distributed",
    excerpt:
      "Cloud storage is a service model where data is stored across distributed infrastructure and accessed remotely with scalability and replication guarantees."
  },
  Software_as_a_service: {
    title: "Software_as_a_service",
    category: "Distributed",
    excerpt:
      "Software as a service packages software delivery as an online service, shifting infrastructure and maintenance burdens away from the end user."
  },
  World_Wide_Web: {
    title: "World_Wide_Web",
    category: "Web",
    excerpt:
      "The World Wide Web is a linked information system built on documents, URLs, protocols, and browsers, and it remains one of the clearest examples of large-scale hypertext."
  },
  Web_server: {
    title: "Web_server",
    category: "Web",
    excerpt:
      "A web server accepts HTTP requests and returns web resources, making it a foundational building block for websites, APIs, and distributed web systems."
  },
  URL: {
    title: "URL",
    category: "Web",
    excerpt:
      "A URL identifies the location of a resource and the mechanism for accessing it, functioning as one of the basic address systems of the web."
  }
};

const recommendationLibrary = {
  Programming_language: {
    "hybrid-corrected": [
      ["Compiler", 0.769],
      ["Syntax_programming", 0.722],
      ["Programming_paradigm", 0.731],
      ["Type_system", 0.708],
      ["Interpreter", 0.704],
      ["Language_model", 0.624],
      ["Database_design", 0.612],
      ["Operating_system", 0.607],
      ["Web_server", 0.596],
      ["Question_answering", 0.592],
      ["Relational_database", 0.588],
      ["Computer_multitasking", 0.583],
      ["Cloud_computing", 0.579],
      ["World_Wide_Web", 0.573],
      ["Natural_language_processing", 0.566],
      ["Computational_linguistics", 0.561],
      ["Machine_translation", 0.556],
      ["Process_computing", 0.552],
      ["Cloud_storage", 0.548],
      ["Software_as_a_service", 0.541]
    ],
    "bge-only": [
      ["Compiler", 0.769],
      ["Programming_paradigm", 0.731],
      ["Syntax_programming", 0.722],
      ["Type_system", 0.714],
      ["Interpreter", 0.708],
      ["Language_model", 0.641],
      ["Natural_language_processing", 0.612],
      ["Question_answering", 0.606],
      ["Machine_translation", 0.598],
      ["Computational_linguistics", 0.594],
      ["Operating_system", 0.581],
      ["Database", 0.576],
      ["Database_design", 0.57],
      ["Relational_database", 0.567],
      ["Cloud_computing", 0.56],
      ["World_Wide_Web", 0.554],
      ["Web_server", 0.549],
      ["Cloud_storage", 0.541],
      ["Process_computing", 0.538],
      ["Software_as_a_service", 0.534]
    ],
    "n2v-uncorrected": [
      ["Compiler", 0.842],
      ["Interpreter", 0.821],
      ["Programming_paradigm", 0.798],
      ["Type_system", 0.773],
      ["Syntax_programming", 0.741],
      ["Operating_system", 0.723],
      ["Process_computing", 0.712],
      ["Database", 0.694],
      ["Relational_database", 0.688],
      ["Web_server", 0.683],
      ["World_Wide_Web", 0.671],
      ["Computer_multitasking", 0.666],
      ["Real_time_os", 0.654],
      ["Cloud_computing", 0.648],
      ["Database_design", 0.641],
      ["Cloud_storage", 0.635],
      ["Language_model", 0.612],
      ["Natural_language_processing", 0.607],
      ["Question_answering", 0.601],
      ["Software_as_a_service", 0.595]
    ],
    "n2v-corrected": [
      ["Compiler", 0.764],
      ["Programming_paradigm", 0.742],
      ["Type_system", 0.724],
      ["Interpreter", 0.718],
      ["Syntax_programming", 0.708],
      ["Database_design", 0.652],
      ["Operating_system", 0.641],
      ["Relational_database", 0.632],
      ["World_Wide_Web", 0.624],
      ["Cloud_computing", 0.619],
      ["Language_model", 0.611],
      ["Question_answering", 0.604],
      ["Web_server", 0.596],
      ["Natural_language_processing", 0.589],
      ["Computer_multitasking", 0.583],
      ["Cloud_storage", 0.579],
      ["Machine_translation", 0.573],
      ["Computational_linguistics", 0.57],
      ["Process_computing", 0.561],
      ["Software_as_a_service", 0.556]
    ],
    "gcn-bge-fusion": [
      ["Compiler", 0.782],
      ["Programming_paradigm", 0.746],
      ["Interpreter", 0.731],
      ["Type_system", 0.719],
      ["Syntax_programming", 0.711],
      ["Operating_system", 0.643],
      ["Database_design", 0.634],
      ["World_Wide_Web", 0.628],
      ["Relational_database", 0.621],
      ["Language_model", 0.617],
      ["Natural_language_processing", 0.61],
      ["Question_answering", 0.603],
      ["Process_computing", 0.596],
      ["Cloud_computing", 0.589],
      ["Web_server", 0.584],
      ["Computer_multitasking", 0.579],
      ["Machine_translation", 0.572],
      ["Computational_linguistics", 0.568],
      ["Cloud_storage", 0.562],
      ["Software_as_a_service", 0.555]
    ]
  },
  Natural_language_processing: {
    "hybrid-corrected": [
      ["Machine_translation", 0.731],
      ["Computational_linguistics", 0.707],
      ["Language_model", 0.697],
      ["Question_answering", 0.678],
      ["Programming_language", 0.566],
      ["Compiler", 0.554],
      ["Interpreter", 0.549],
      ["Programming_paradigm", 0.544],
      ["Type_system", 0.539],
      ["Database", 0.531],
      ["Database_design", 0.528],
      ["Relational_database", 0.522],
      ["Cloud_computing", 0.519],
      ["Operating_system", 0.516],
      ["World_Wide_Web", 0.51],
      ["Web_server", 0.506],
      ["Cloud_storage", 0.501],
      ["Software_as_a_service", 0.497],
      ["Process_computing", 0.492],
      ["Computer_multitasking", 0.488]
    ],
    "bge-only": [
      ["Machine_translation", 0.731],
      ["Computational_linguistics", 0.707],
      ["Language_model", 0.697],
      ["Question_answering", 0.678],
      ["Programming_language", 0.612],
      ["Compiler", 0.587],
      ["Programming_paradigm", 0.579],
      ["Type_system", 0.574],
      ["Interpreter", 0.571],
      ["Database", 0.54],
      ["Cloud_computing", 0.528],
      ["World_Wide_Web", 0.519],
      ["Operating_system", 0.515],
      ["Web_server", 0.51],
      ["Database_design", 0.507],
      ["Relational_database", 0.503],
      ["Interpreter", 0.499],
      ["Cloud_storage", 0.497],
      ["Software_as_a_service", 0.491],
      ["Process_computing", 0.487]
    ],
    "n2v-uncorrected": [
      ["Machine_translation", 0.802],
      ["Computational_linguistics", 0.786],
      ["Language_model", 0.758],
      ["Question_answering", 0.742],
      ["Programming_language", 0.626],
      ["Compiler", 0.617],
      ["Database", 0.604],
      ["World_Wide_Web", 0.595],
      ["Web_server", 0.591],
      ["Cloud_computing", 0.586],
      ["Operating_system", 0.58],
      ["Database_design", 0.571],
      ["Relational_database", 0.567],
      ["Cloud_storage", 0.562],
      ["Software_as_a_service", 0.558],
      ["Programming_paradigm", 0.552],
      ["Type_system", 0.548],
      ["Interpreter", 0.544],
      ["Process_computing", 0.539],
      ["Computer_multitasking", 0.533]
    ],
    "n2v-corrected": [
      ["Machine_translation", 0.744],
      ["Computational_linguistics", 0.728],
      ["Language_model", 0.702],
      ["Question_answering", 0.689],
      ["Programming_language", 0.588],
      ["Compiler", 0.577],
      ["Database", 0.556],
      ["Cloud_computing", 0.548],
      ["World_Wide_Web", 0.542],
      ["Database_design", 0.536],
      ["Operating_system", 0.531],
      ["Relational_database", 0.528],
      ["Web_server", 0.524],
      ["Programming_paradigm", 0.519],
      ["Type_system", 0.516],
      ["Interpreter", 0.512],
      ["Cloud_storage", 0.508],
      ["Software_as_a_service", 0.501],
      ["Process_computing", 0.496],
      ["Computer_multitasking", 0.492]
    ],
    "gcn-bge-fusion": [
      ["Machine_translation", 0.756],
      ["Computational_linguistics", 0.732],
      ["Language_model", 0.709],
      ["Question_answering", 0.694],
      ["Programming_language", 0.598],
      ["Compiler", 0.584],
      ["World_Wide_Web", 0.566],
      ["Database", 0.562],
      ["Cloud_computing", 0.553],
      ["Operating_system", 0.547],
      ["Web_server", 0.542],
      ["Database_design", 0.539],
      ["Relational_database", 0.534],
      ["Type_system", 0.526],
      ["Programming_paradigm", 0.523],
      ["Interpreter", 0.519],
      ["Cloud_storage", 0.514],
      ["Software_as_a_service", 0.507],
      ["Process_computing", 0.501],
      ["Computer_multitasking", 0.496]
    ]
  },
  Operating_system: {
    "hybrid-corrected": [
      ["Process_computing", 0.76],
      ["Computer_multitasking", 0.743],
      ["Real_time_os", 0.728],
      ["Programming_language", 0.607],
      ["Compiler", 0.598],
      ["Interpreter", 0.587],
      ["Type_system", 0.575],
      ["Database", 0.57],
      ["Cloud_computing", 0.564],
      ["Web_server", 0.558],
      ["World_Wide_Web", 0.553],
      ["Database_design", 0.549],
      ["Relational_database", 0.545],
      ["Cloud_storage", 0.54],
      ["Software_as_a_service", 0.533],
      ["Programming_paradigm", 0.528],
      ["Language_model", 0.518],
      ["Natural_language_processing", 0.514],
      ["Question_answering", 0.509],
      ["Machine_translation", 0.503]
    ],
    "bge-only": [
      ["Process_computing", 0.76],
      ["Computer_multitasking", 0.743],
      ["Real_time_os", 0.728],
      ["Programming_language", 0.581],
      ["Compiler", 0.574],
      ["Interpreter", 0.568],
      ["Database", 0.561],
      ["Cloud_computing", 0.556],
      ["Web_server", 0.551],
      ["World_Wide_Web", 0.548],
      ["Type_system", 0.543],
      ["Database_design", 0.538],
      ["Relational_database", 0.535],
      ["Programming_paradigm", 0.529],
      ["Cloud_storage", 0.524],
      ["Software_as_a_service", 0.519],
      ["Language_model", 0.508],
      ["Natural_language_processing", 0.504],
      ["Question_answering", 0.501],
      ["Machine_translation", 0.497]
    ],
    "n2v-uncorrected": [
      ["Process_computing", 0.819],
      ["Computer_multitasking", 0.797],
      ["Real_time_os", 0.776],
      ["Compiler", 0.713],
      ["Programming_language", 0.709],
      ["Interpreter", 0.698],
      ["Database", 0.682],
      ["Database_design", 0.671],
      ["Relational_database", 0.663],
      ["Web_server", 0.657],
      ["World_Wide_Web", 0.651],
      ["Cloud_computing", 0.642],
      ["Type_system", 0.638],
      ["Programming_paradigm", 0.631],
      ["Cloud_storage", 0.624],
      ["Software_as_a_service", 0.618],
      ["Language_model", 0.592],
      ["Natural_language_processing", 0.587],
      ["Question_answering", 0.579],
      ["Machine_translation", 0.572]
    ],
    "n2v-corrected": [
      ["Process_computing", 0.742],
      ["Computer_multitasking", 0.728],
      ["Real_time_os", 0.71],
      ["Programming_language", 0.641],
      ["Compiler", 0.632],
      ["Interpreter", 0.621],
      ["Database", 0.611],
      ["Database_design", 0.603],
      ["Relational_database", 0.598],
      ["Web_server", 0.589],
      ["World_Wide_Web", 0.582],
      ["Cloud_computing", 0.576],
      ["Type_system", 0.571],
      ["Programming_paradigm", 0.566],
      ["Cloud_storage", 0.559],
      ["Software_as_a_service", 0.554],
      ["Language_model", 0.526],
      ["Natural_language_processing", 0.521],
      ["Question_answering", 0.517],
      ["Machine_translation", 0.511]
    ],
    "gcn-bge-fusion": [
      ["Process_computing", 0.754],
      ["Computer_multitasking", 0.737],
      ["Real_time_os", 0.723],
      ["Programming_language", 0.653],
      ["Compiler", 0.642],
      ["Interpreter", 0.634],
      ["Database", 0.616],
      ["Database_design", 0.609],
      ["Relational_database", 0.603],
      ["Web_server", 0.597],
      ["World_Wide_Web", 0.589],
      ["Cloud_computing", 0.584],
      ["Type_system", 0.579],
      ["Programming_paradigm", 0.573],
      ["Cloud_storage", 0.567],
      ["Software_as_a_service", 0.561],
      ["Language_model", 0.533],
      ["Natural_language_processing", 0.529],
      ["Question_answering", 0.522],
      ["Machine_translation", 0.518]
    ]
  }
};

const ui = {
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
  resultsCount: document.getElementById("results-count"),
  resultsList: document.getElementById("results-list"),
  emptyState: document.getElementById("empty-state"),
  previewTitle: document.getElementById("preview-title"),
  previewRank: document.getElementById("preview-rank"),
  previewScore: document.getElementById("preview-score"),
  previewExcerpt: document.getElementById("preview-excerpt")
};

const state = {
  selectedModel: models[3].id,
  selectedArticleKey: "",
  selectedResultKey: "",
  results: []
};

const searchableTitles = Object.keys(articles)
  .map((key) => ({
    key,
    title: articles[key].title
  }))
  .sort((a, b) => a.title.localeCompare(b.title));

function init() {
  renderModelOptions();
  attachEvents();
  updateModelUI();
  renderSuggestions("");
}

function renderModelOptions() {
  ui.modelSelect.innerHTML = models
    .map(
      (model) =>
        `<option value="${model.id}" ${model.id === state.selectedModel ? "selected" : ""}>${model.name}</option>`
    )
    .join("");
}

function attachEvents() {
  ui.modelSelect.addEventListener("change", (event) => {
    state.selectedModel = event.target.value;
    updateModelUI();
    if (state.selectedArticleKey) {
      runRecommendations(state.selectedArticleKey);
    }
  });

  ui.queryInput.addEventListener("input", () => {
    renderSuggestions(ui.queryInput.value);
    ui.disambiguationBlock.classList.add("is-hidden");
  });

  ui.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleQuerySubmit();
    }
  });

  ui.searchButton.addEventListener("click", handleQuerySubmit);
  ui.clearButton.addEventListener("click", resetSelection);
}

function updateModelUI() {
  const model = models.find((item) => item.id === state.selectedModel);
  ui.modelDescription.textContent = model.description;
  ui.activeModelPill.textContent = model.name;
}

function renderSuggestions(rawQuery) {
  const query = rawQuery.trim().toLowerCase();
  if (!query) {
    ui.suggestionsList.innerHTML = searchableTitles
      .slice(0, 6)
      .map(
        (item) => `
          <li>
            <button class="suggestion-item" type="button" data-article-key="${item.key}">
              ${item.title}
              <small>${articles[item.key].category}</small>
            </button>
          </li>
        `
      )
      .join("");
  } else {
    const matches = searchableTitles
      .filter((item) => item.title.toLowerCase().includes(query))
      .slice(0, 8);

    ui.suggestionsList.innerHTML = matches.length
      ? matches
          .map(
            (item) => `
              <li>
                <button class="suggestion-item" type="button" data-article-key="${item.key}">
                  ${item.title}
                  <small>${articles[item.key].category}</small>
                </button>
              </li>
            `
          )
          .join("")
      : `<li class="helper-text">No partial title matches yet.</li>`;
  }

  bindSuggestionButtons();
}

function bindSuggestionButtons() {
  document.querySelectorAll(".suggestion-item").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.articleKey;
      selectStartingArticle(key);
      ui.queryInput.value = articles[key].title;
    });
  });
}

function handleQuerySubmit() {
  const query = ui.queryInput.value.trim();
  if (!query) {
    return;
  }

  const exact = searchableTitles.find(
    (item) => item.title.toLowerCase() === query.toLowerCase()
  );

  if (exact) {
    selectStartingArticle(exact.key);
    return;
  }

  const candidates = searchableTitles
    .filter((item) => item.title.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 8);

  if (!candidates.length) {
    ui.disambiguationBlock.classList.remove("is-hidden");
    ui.disambiguationList.innerHTML =
      '<li class="helper-text">No matching article titles found in the prototype dataset.</li>';
    return;
  }

  ui.disambiguationBlock.classList.remove("is-hidden");
  ui.disambiguationList.innerHTML = candidates
    .map(
      (item) => `
        <li>
          <button class="candidate-item" type="button" data-article-key="${item.key}">
            ${item.title}
            <small>${articles[item.key].category}</small>
          </button>
        </li>
      `
    )
    .join("");

  document.querySelectorAll(".candidate-item").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.articleKey;
      ui.queryInput.value = articles[key].title;
      selectStartingArticle(key);
    });
  });
}

function selectStartingArticle(articleKey) {
  state.selectedArticleKey = articleKey;
  state.selectedResultKey = "";

  ui.selectedArticleCard.classList.remove("is-hidden");
  ui.selectedArticleTitle.textContent = articles[articleKey].title;
  ui.selectedArticleMeta.textContent = `${articles[articleKey].category} article selected as the recommendation source.`;
  ui.activeSourcePill.textContent = articles[articleKey].title;
  ui.disambiguationBlock.classList.add("is-hidden");

  runRecommendations(articleKey);
}

function runRecommendations(articleKey) {
  const articleRecommendations =
    recommendationLibrary[articleKey]?.[state.selectedModel] ||
    recommendationLibrary[articleKey]?.["hybrid-corrected"] ||
    [];

  state.results = articleRecommendations.map(([resultKey, score], index) => ({
    resultKey,
    score,
    rank: index + 1,
    article: articles[resultKey]
  }));

  renderResults();
  if (state.results.length) {
    selectResult(state.results[0].resultKey);
  } else {
    resetPreview();
  }
}

function renderResults() {
  const sourceTitle = state.selectedArticleKey
    ? articles[state.selectedArticleKey].title
    : "Start by selecting an article";

  ui.resultsTitle.textContent = state.selectedArticleKey
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
          <button class="result-card ${item.resultKey === state.selectedResultKey ? "is-active" : ""}" type="button" data-result-key="${item.resultKey}">
            <div class="result-rank">${item.rank}</div>
            <div class="result-content">
              <h4>${item.article.title}</h4>
              <p>${item.article.excerpt}</p>
            </div>
            <div class="score-chip">${item.score.toFixed(3)}</div>
          </button>
        </li>
      `
    )
    .join("");

  document.querySelectorAll(".result-card").forEach((button) => {
    button.addEventListener("click", () => {
      selectResult(button.dataset.resultKey);
    });
  });
}

function selectResult(resultKey) {
  state.selectedResultKey = resultKey;
  const item = state.results.find((entry) => entry.resultKey === resultKey);
  if (!item) {
    return;
  }

  ui.previewTitle.textContent = item.article.title;
  ui.previewRank.textContent = `Rank ${item.rank}`;
  ui.previewScore.textContent = `Score ${item.score.toFixed(3)}`;
  ui.previewExcerpt.textContent = item.article.excerpt;

  renderResults();
}

function resetPreview() {
  ui.previewTitle.textContent = "Choose a recommended article";
  ui.previewRank.textContent = "Rank -";
  ui.previewScore.textContent = "Score -";
  ui.previewExcerpt.textContent =
    "The right panel will show a compact title, score, and representative article excerpt once a result is selected from the center column.";
}

function resetSelection() {
  state.selectedArticleKey = "";
  state.selectedResultKey = "";
  state.results = [];
  ui.queryInput.value = "";
  ui.selectedArticleCard.classList.add("is-hidden");
  ui.activeSourcePill.textContent = "None selected";
  ui.resultsTitle.textContent = "Start by selecting an article";
  ui.resultsCount.textContent = "0 items";
  ui.resultsList.innerHTML = "";
  ui.emptyState.style.display = "block";
  ui.disambiguationBlock.classList.add("is-hidden");
  resetPreview();
  renderSuggestions("");
}

init();
